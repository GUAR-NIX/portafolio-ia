"""
Evaluación de los tres enfoques sobre el conjunto test (320 reseñas etiquetadas a mano).

    python -m resenas.evaluar            # usa la caché de embeddings y las predicciones grabadas del LLM

* Clásico: se entrena con las ~54 mil reseñas no etiquetadas (estrellas) y su umbral se elige en dev.
* Embeddings: se entrena con dev (cláusulas ↔ citas) + 3000 reseñas con estrellas; C y umbral por validación
  cruzada en dev (5 particiones por reseña).
* LLM: sin entrenamiento; prompt escrito con la guía de anotación antes de ver el test.

Salida: reports/evaluacion.json, reports/evaluacion_detalle.csv y las figuras de evaluación.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .enfoque_clasico import RUTA_MODELO, ModeloClasico
from .enfoque_embeddings import (RUTA_CACHE, CacheEmbeddings, ModeloEmbeddings, analizar_lote,
                                 etiquetas_clausulas, preparar_textos)
from .enfoque_clasico import etiqueta_estrellas
from .etiquetas import cargar_muestra, como_dict, leer_jsonl
from .metricas import acuerdo, evaluar
from .preparar import cargar_resenas, muestra_debil
from .texto import texto_completo

ARCHIVO_COSTOS = config.DIR_REPORTES / "costos.json"
PRED_LLM = config.DIR_REPORTES / "predicciones_llm.jsonl"
PRED_TODAS = config.DIR_REPORTES / "predicciones_test.jsonl"


def _a_dict(pred: dict) -> dict:
    return {"global": pred["sentimiento_global"], "aspectos": {m["aspecto"]: m["polaridad"] for m in pred["aspectos"]}}


def cargar_etiquetas():
    muestra = cargar_muestra()
    dev = {d["id"]: d for d in leer_jsonl(config.ETIQUETAS_DEV)}
    test = {d["id"]: d for d in leer_jsonl(config.ETIQUETAS_TEST)}
    es = {d["id"]: d for d in leer_jsonl(config.ETIQUETAS_ES)}
    return muestra, dev, test, es


# --------------------------------------------------------------------------- clásico
def correr_clasico(resenas: pd.DataFrame, muestra, dev, objetivo: list[str]) -> tuple[dict, dict]:
    """Predicciones del clásico para las reseñas `objetivo` (test + español); umbral elegido en dev."""
    if RUTA_MODELO.exists():
        modelo = ModeloClasico.cargar(RUTA_MODELO)
        print("Clásico: modelo cargado de", RUTA_MODELO.name)
    else:
        t0 = time.perf_counter()
        modelo = ModeloClasico().entrenar(resenas[~resenas["id"].isin(muestra)])   # sin dev, test ni español
        print(f"Clásico: entrenado con {len(resenas) - len(muestra):,} reseñas en {time.perf_counter() - t0:.0f} s")
    verdad_dev = {i: como_dict(d) for i, d in dev.items()}
    pares_dev = [(muestra[i]["titulo_resena"], muestra[i]["texto"]) for i in dev]
    mejor = (-1, None)
    for u in [0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.01]:
        modelo.umbral_clausula = u
        pred = {i: _a_dict(p) for i, p in zip(dev, modelo.analizar_lote(pares_dev))}
        f = evaluar(verdad_dev, pred, bootstrap=0)["f1_aspecto_polaridad"]
        mejor = max(mejor, (f, u))
    modelo.umbral_clausula = mejor[1]
    print(f"  umbral de cláusula elegido en dev: {mejor[1]} (F1 aspecto+polaridad dev {mejor[0]:.3f})")
    modelo.guardar(RUTA_MODELO)
    pares = [(muestra[i]["titulo_resena"], muestra[i]["texto"]) for i in objetivo]
    t0 = time.perf_counter()
    preds = modelo.analizar_lote(pares)
    seg = (time.perf_counter() - t0) / len(pares)
    return dict(zip(objetivo, preds)), {"segundos_por_resena": seg, "umbral_clausula": mejor[1]}


# --------------------------------------------------------------------------- embeddings
def _datos_embeddings(cache, muestra, dev_ids, dev, debil):
    resenas_dev = [muestra[i] for i in dev_ids]
    cl, Y, pols = etiquetas_clausulas(resenas_dev, dev)
    X_cl = cache.matriz(cl)
    textos_debil = [texto_completo(t, x) for t, x in zip(debil["titulo_resena"], debil["texto"])]
    X_debil = cache.matriz(textos_debil)
    y_debil = [etiqueta_estrellas(e) for e in debil["estrellas"]]
    _, completos_dev = preparar_textos(resenas_dev)
    X_glob = np.vstack([X_debil, cache.matriz(completos_dev)])
    y_glob = y_debil + [dev[i]["sentimiento_global"] for i in dev_ids]
    return X_cl, Y, pols, X_debil, y_debil, X_glob, y_glob


def correr_embeddings(muestra, dev, objetivo: list[str], debil) -> tuple[dict, dict]:
    cache = CacheEmbeddings(RUTA_CACHE)
    dev_ids = list(dev)
    rng = np.random.default_rng(config.SEMILLA)
    pliegues = np.array_split(rng.permutation(dev_ids), 5)
    rejilla = [(C, u) for C in (0.5, 2.0, 8.0, 32.0) for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    puntajes = {k: [] for k in rejilla}
    for k, val in enumerate(pliegues):
        val = list(val)
        entren = [i for i in dev_ids if i not in set(val)]
        datos = _datos_embeddings(cache, muestra, entren, dev, debil)
        verdad = {i: como_dict(dev[i]) for i in val}
        for C in (0.5, 2.0, 8.0, 32.0):
            m = ModeloEmbeddings(C_aspecto=C).entrenar(*datos)
            for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                m.umbral = u
                pred = {i: _a_dict(p) for i, p in zip(val, analizar_lote(m, cache, [muestra[i] for i in val]))}
                puntajes[(C, u)].append(evaluar(verdad, pred, bootstrap=0)["f1_aspecto_polaridad"])
    (C, u), f = max(((k, float(np.mean(v))) for k, v in puntajes.items()), key=lambda x: x[1])
    print(f"Embeddings: C={C}, umbral={u} por validación cruzada en dev (F1 aspecto+polaridad {f:.3f})")
    modelo = ModeloEmbeddings(C_aspecto=C, umbral=u).entrenar(*_datos_embeddings(cache, muestra, dev_ids, dev, debil))
    t0 = time.perf_counter()
    preds = analizar_lote(modelo, cache, [muestra[i] for i in objetivo])
    seg_clf = (time.perf_counter() - t0) / len(objetivo)
    cls, _ = preparar_textos([muestra[i] for i in objetivo])
    textos_por_resena = float(np.mean([len(c) + 1 for c in cls]))
    import joblib
    joblib.dump(modelo, config.DIR_DATOS / "modelo_embeddings.joblib")
    return dict(zip(objetivo, preds)), {"C_aspecto": C, "umbral": u, "f1_cv_dev": f,
                                    "segundos_clasificadores": seg_clf, "textos_por_resena": textos_por_resena}


def medir_costo_embeddings(resenas: pd.DataFrame, n: int = 96) -> dict:
    """Tiempo real de bge-m3 en esta PC con textos no cacheados (necesita Ollama; se guarda en costos.json)."""
    from .llm import embeber
    textos = [texto_completo(t, x) for t, x in resenas.sample(n, random_state=7)[["titulo_resena", "texto"]].values]
    embeber(textos[:4])     # calentar el modelo
    t0 = time.perf_counter()
    embeber(textos)
    return {"segundos_por_texto": (time.perf_counter() - t0) / n, "n": n}


# --------------------------------------------------------------------------- LLM
def cargar_llm(objetivo) -> tuple[dict, dict]:
    filas = {d["id"]: d for d in leer_jsonl(PRED_LLM)} if PRED_LLM.exists() else {}
    faltan = [i for i in objetivo if i not in filas]
    if faltan:
        print(f"LLM: faltan {len(faltan)} de {len(objetivo)} reseñas; se evalúan solo las que hay")
    preds = {i: filas[i] for i in objetivo if i in filas}
    if not preds:
        return {}, {}
    seg = np.array([p["segundos"] for p in preds.values()])
    tok = np.array([p["tokens_salida"] for p in preds.values()])
    n_asp = sum(len(p["aspectos"]) for p in preds.values())
    costo = {"segundos_por_resena": float(seg.mean()), "mediana_segundos": float(np.median(seg)),
             "p90_segundos": float(np.percentile(seg, 90)), "tokens_salida_medios": float(tok.mean()),
             "errores": sum(bool(p["error"]) for p in preds.values()),
             "citas_no_literales": sum(p["citas_no_literales"] for p in preds.values()),
             "menciones": n_asp, "n": len(preds)}
    return preds, costo


# --------------------------------------------------------------------------- todo
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    muestra, dev, test, es = cargar_etiquetas()
    resenas = cargar_resenas()
    debil = muestra_debil(resenas)
    verdad = {i: como_dict(d) for i, d in test.items()}
    objetivo = list(test) + list(es)

    costos = json.loads(ARCHIVO_COSTOS.read_text()) if ARCHIVO_COSTOS.exists() else {}
    p_cla, c_cla = correr_clasico(resenas, muestra, dev, objetivo)
    p_emb, c_emb = correr_embeddings(muestra, dev, objetivo, debil)
    p_llm, c_llm = cargar_llm(list(test))
    p_llm_es, _ = cargar_llm(list(es))
    p_llm.update(p_llm_es)
    if "embeddings" not in costos:
        try:
            costos["embeddings"] = medir_costo_embeddings(resenas)
            ARCHIVO_COSTOS.write_text(json.dumps(costos, indent=1))
        except Exception as e:  # noqa: BLE001  (sin Ollama: se usa lo grabado)
            print("No se pudo medir el costo de bge-m3:", e)
    seg_texto = costos.get("embeddings", {}).get("segundos_por_texto", float("nan"))
    c_emb["segundos_por_texto"] = seg_texto
    c_emb["segundos_por_resena"] = seg_texto * c_emb["textos_por_resena"] + c_emb["segundos_clasificadores"]

    ids_llm = [i for i in test if i in p_llm]
    resultados = {}
    for nombre, preds, costo in [("clasico", p_cla, c_cla), ("embeddings", p_emb, c_emb), ("llm", p_llm, c_llm)]:
        pd_ = {i: _a_dict(p) for i, p in preds.items()}
        r = evaluar(verdad, pd_, ids=ids_llm)
        r["costo"] = costo
        r["por_estrellas"] = {e: {k: v for k, v in evaluar(verdad, pd_, ids=[i for i in ids_llm
                                  if muestra[i]["estrellas"] == e], bootstrap=0).items() if k != "por_aspecto"}
                              for e in range(1, 6)}
        resultados[nombre] = r
        print(f"{nombre:>10}: F1 aspectos {r['f1_aspectos']:.3f} · F1 aspecto+pol {r['f1_aspecto_polaridad']:.3f}"
              f" · polaridad {r['exactitud_polaridad']:.3f} · global {r['exactitud_global']:.3f}"
              f" · {costo['segundos_por_resena']:.3g} s/reseña")

    # Parte en español: 80 reseñas escritas en español (ningún enfoque se entrenó ni ajustó con español)
    verdad_es = {i: como_dict(d) for i, d in es.items()}
    ids_es = [i for i in es if i in p_llm]
    espanol = {}
    for nombre, preds in ([("clasico", p_cla), ("embeddings", p_emb), ("llm", p_llm)] if ids_es else []):
        r = evaluar(verdad_es, {i: _a_dict(preds[i]) for i in ids_es}, ids=ids_es)
        r.pop("por_aspecto")
        espanol[nombre] = r
        print(f"{'español ' + nombre:>18}: F1 aspectos {r['f1_aspectos']:.3f} · polaridad {r['exactitud_polaridad']:.3f}"
              f" · global {r['exactitud_global']:.3f}")

    p1 = {i: como_dict(d) for i, d in test.items()}
    p2 = {d["id"]: como_dict(d) for d in leer_jsonl(config.ETIQUETAS_TEST_PASADA2)}
    salida = {"n_test": len(ids_llm), "n_dev": len(dev), "enfoques": resultados, "acuerdo_anotacion": acuerdo(p1, p2),
              "soporte_test": {a: sum(a in verdad[i]["aspectos"] for i in ids_llm) for a in config.ASPECTOS},
              "espanol": {"n": len(ids_es), "enfoques": espanol}}
    (config.DIR_REPORTES / "evaluacion.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    filas = []
    with open(PRED_TODAS, "w", encoding="utf-8") as fh:
        for i in ids_llm:
            reg = {"id": i, "verdad": test[i], "clasico": p_cla[i], "embeddings": p_emb[i],
                   "llm": {k: p_llm[i][k] for k in ("sentimiento_global", "aspectos")}}
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
            for a in config.ASPECTOS:
                fila = {"id": i, "estrellas": muestra[i]["estrellas"], "aspecto": a, "verdad": verdad[i]["aspectos"].get(a)}
                for nombre, preds in [("clasico", p_cla), ("embeddings", p_emb), ("llm", p_llm)]:
                    fila[nombre] = _a_dict(preds[i])["aspectos"].get(a)
                if any(fila[k] for k in ("verdad", "clasico", "embeddings", "llm")):
                    filas.append(fila)
    pd.DataFrame(filas).to_csv(config.DIR_REPORTES / "evaluacion_detalle.csv", index=False)
    from .graficos import figuras_evaluacion
    figuras_evaluacion(salida)
    print("→ reports/evaluacion.json")


if __name__ == "__main__":
    main()
