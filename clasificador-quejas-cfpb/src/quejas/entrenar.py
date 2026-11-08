"""Pipeline completo de IA_05_clasificador_quejas.

  1. Carga el corpus CFPB con particiones temporales (train antiguo, val y test recientes)
  2. Asegura los embeddings bge-m3 (data/embeddings_bge_m3.npz)
  3. Entrena TF-IDF + LR (25.722 quejas), bge-m3 + LR (1.800), Zero-Shot bge-m3 y calibra por temperatura
  4. Reproduce (o consulta) qwen2.5:7b zero-shot y few-shot sobre el conjunto común de prueba
  5. Evalúa TODOS los enfoques sobre las mismas quejas: IC bootstrap, diferencias pareadas, cascada,
     deriva temporal, urgencia, calibración, señales de confianza del LLM y transferencia al español
  6. Genera figuras, reports/resultados.json y el artefacto para la API y la app

Con QUEJAS_PROVEEDOR=grabado (por defecto) no llama a Ollama: usa solo lo grabado.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config, datos, evaluacion, graficos
from .grabar import elegir_ejemplos_few_shot
from .llm import CacheEmbeddings, GrabadoraLLM
from .modelos import (
    ModeloEmbeddingsSupervisado,
    ModeloLLM,
    ModeloTFIDF,
    ModeloUrgenciaTFIDF,
    ModeloZeroShotEmbeddings,
    ajustar_temperatura,
    cascada,
    curva_aprendizaje_few_shot,
    umbral_por_presupuesto,
    urgencia_por_reglas,
)

FRACCION_AL_LLM = 0.25  # presupuesto de la cascada: 25 % de las quejas (las de menor confianza) van al LLM

N_TFIDF = "1. TF-IDF + LR (25.722 ej.)"
N_TFIDF_P = "1a. TF-IDF + LR (1.800 ej.)"
N_ZS = "2. Zero-Shot prototipos bge-m3"
N_EMB = "3. bge-m3 + LR (1.800 ej.)"
N_LLM0 = "4. LLM zero-shot (qwen2.5:7b)"
N_LLM1 = "5. LLM few-shot (qwen2.5:7b, 6 ej.)"
N_CASC0 = "6. Cascada TF-IDF → LLM zero-shot"
N_CASC1 = "7. Cascada TF-IDF → LLM few-shot"


def prefijo_grabado(textos: list[str], grabadora: GrabadoraLLM, modos=("cero", "pocos")) -> int:
    """Mayor prefijo (múltiplo de 6, balanceado por mesa) con respuesta grabada en todos los modos."""
    n = 0
    for t in textos:
        if all(grabadora.obtener(t, m) is not None for m in modos):
            n += 1
        else:
            break
    return n - n % len(config.CATEGORIAS)


def _pred(probs: np.ndarray) -> np.ndarray:
    return np.asarray([config.CATEGORIAS[i] for i in probs.argmax(axis=1)], dtype=str)


def _bootstrap_dif_independiente(y_a, p_a, y_b, p_b, n_boot: int = 500) -> tuple[float, float]:
    """IC 95 % de F1(A) − F1(B) cuando A y B son conjuntos distintos (remuestreo independiente)."""
    rng = np.random.default_rng(config.SEMILLA)
    y_a, p_a, y_b, p_b = (np.asarray(v, dtype=str) for v in (y_a, p_a, y_b, p_b))
    difs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        difs.append(
            f1_score(y_a[ia], p_a[ia], average="macro", zero_division=0)
            - f1_score(y_b[ib], p_b[ib], average="macro", zero_division=0)
        )
    return float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def estadistica_confianza_autodeclarada() -> dict:
    """Experimento descartado: pedirle al LLM un campo `confianza` en el JSON (casi siempre responde 0,99).
    Sus 54 respuestas están en data/experimento_confianza_autodeclarada.jsonl."""
    ruta = config.RAIZ / "data" / "experimento_confianza_autodeclarada.jsonl"
    if not ruta.exists():
        return {}
    confs = [json.loads(l)["confianza"] for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "n_respuestas": len(confs),
        "n_con_0_99": int(sum(c >= 0.99 for c in confs)),
        "valores_distintos": sorted(set(float(c) for c in confs)),
    }


def main() -> None:
    t0 = time.perf_counter()
    for carpeta in (config.MODELOS, config.REPORTES, config.FIGURAS):
        carpeta.mkdir(parents=True, exist_ok=True)
    permitir_ollama = os.environ.get("QUEJAS_PROVEEDOR", "grabado") == "ollama"

    print("1/9 · Cargando corpus CFPB con particiones temporales y quejas traducidas al español...")
    df = datos.cargar_corpus_cfpb()
    df_es = datos.cargar_quejas_espanol()
    resumen_fuente = json.loads(config.RESUMEN_FUENTE.read_text(encoding="utf-8"))
    res_cat = datos.resumen_por_categoria(df)
    res_cat.to_csv(config.REPORTES / "resumen_categorias_cfpb.csv", index=False)
    particiones = pd.crosstab(df["categoria"], df["particion"]).reindex(config.CATEGORIAS)
    particiones.to_csv(config.REPORTES / "particiones_temporales.csv")

    tr = df[df["particion"] == "train"].reset_index(drop=True)
    tr_emb = df[df["orden_train_emb"] >= 0].sort_values("orden_train_emb").reset_index(drop=True)
    val = df[df["particion"] == "val"].reset_index(drop=True)
    val_emb = df[df["orden_val_emb"] >= 0].sort_values("orden_val_emb").reset_index(drop=True)
    te = df[df["particion"] == "test"].reset_index(drop=True)
    te_vieja = df[df["particion"] == "test_misma_epoca"].reset_index(drop=True)
    comun_todo = df[df["orden_comun"] >= 0].sort_values("orden_comun").reset_index(drop=True)

    ejemplos = elegir_ejemplos_few_shot(df)
    grabadora = GrabadoraLLM(config.GRABACION_LLM)
    n_comun = len(comun_todo) if permitir_ollama else prefijo_grabado(comun_todo["texto_corto"].tolist(), grabadora)
    if n_comun == 0:
        raise RuntimeError("No hay respuestas grabadas del LLM: ejecuta python -m quejas.grabar llm")
    co = comun_todo.head(n_comun).reset_index(drop=True)
    print(f"      train={len(tr):,} · train_emb={len(tr_emb):,} · val={len(val):,} · val_emb={len(val_emb)} · "
          f"test={len(te):,} · test_misma_epoca={len(te_vieja):,} · test_comun={len(co)} · español={len(df_es)}")

    print("2/9 · Embeddings bge-m3 (caché)...")
    cache = CacheEmbeddings(config.CACHE_EMBEDDINGS)
    emb = lambda d: cache.matriz(d["texto_corto"].tolist(), permitir_ollama=permitir_ollama)  # noqa: E731
    X_tr_emb, X_val_emb, X_co, X_es = emb(tr_emb), emb(val_emb), emb(co), emb(df_es)

    print("3/9 · Entrenando y calibrando (escalado de temperatura en validación)...")
    m_tf = ModeloTFIDF().ajustar(tr["texto_limpio_tfidf"].tolist(), tr["categoria"].to_numpy())
    m_tf_p = ModeloTFIDF().ajustar(tr_emb["texto_limpio_tfidf"].tolist(), tr_emb["categoria"].to_numpy())
    m_zs = ModeloZeroShotEmbeddings(temperatura_escala=16.0).ajustar_prototipos(cache, permitir_ollama=permitir_ollama)
    m_emb = ModeloEmbeddingsSupervisado(C=4.0).ajustar_vectores(X_tr_emb, tr_emb["categoria"].to_numpy())
    y_val = val_emb["categoria"].to_numpy()
    m_tf.temperatura_ = ajustar_temperatura(m_tf.logits(val_emb["texto_limpio_tfidf"].tolist()), y_val)
    m_tf_p.temperatura_ = ajustar_temperatura(m_tf_p.logits(val_emb["texto_limpio_tfidf"].tolist()), y_val)
    m_emb.temperatura_ = ajustar_temperatura(m_emb.logits(X_val_emb), y_val)

    print(f"4/9 · LLM qwen2.5:7b zero-shot y few-shot sobre {len(co)} quejas comunes y {len(df_es)} en español...")
    llm0 = ModeloLLM(grabadora, "cero", ejemplos)
    llm1 = ModeloLLM(grabadora, "pocos", ejemplos)
    r0 = llm0.predecir_lote(co["texto_corto"].tolist(), permitir_ollama)
    r1 = llm1.predecir_lote(co["texto_corto"].tolist(), permitir_ollama)
    hay_es = all(grabadora.obtener(t, m) is not None for t in df_es["texto_corto"] for m in ("cero", "pocos"))
    if hay_es or permitir_ollama:
        r0_es = llm0.predecir_lote(df_es["texto_corto"].tolist(), permitir_ollama)
        r1_es = llm1.predecir_lote(df_es["texto_corto"].tolist(), permitir_ollama)

    print("5/9 · Evaluando en el conjunto común (mismas quejas para todos)...")
    y_co = co["categoria"].to_numpy()
    txt_co = co["texto_limpio_tfidf"].tolist()
    P = {
        N_TFIDF: m_tf.predecir_proba(txt_co),
        N_TFIDF_P: m_tf_p.predecir_proba(txt_co),
        N_ZS: m_zs.predecir_proba_desde_vectores(X_co),
        N_EMB: m_emb.predecir_proba_desde_vectores(X_co),
        N_LLM0: r0["probs"],
        N_LLM1: r1["probs"],
    }
    ms_tf = m_tf.ms_por_queja_
    ms_emb = cache.segundos_por_texto * 1000.0
    ms_llm0, ms_llm1 = float(r0["segundos"].mean() * 1000.0), float(r1["segundos"].mean() * 1000.0)  # solo el conjunto común
    conf_val_tf = m_tf.predecir_proba(val_emb["texto_limpio_tfidf"].tolist()).max(axis=1)
    tau_casc = umbral_por_presupuesto(conf_val_tf, FRACCION_AL_LLM)
    P[N_CASC0], al_llm = cascada(P[N_TFIDF], P[N_LLM0], tau_casc)
    P[N_CASC1], _ = cascada(P[N_TFIDF], P[N_LLM1], tau_casc)
    frac_llm = float(al_llm.mean())
    latencias = {
        N_TFIDF: ms_tf, N_TFIDF_P: m_tf_p.ms_por_queja_, N_ZS: ms_emb, N_EMB: ms_emb,
        N_LLM0: ms_llm0, N_LLM1: ms_llm1,
        N_CASC0: ms_tf + frac_llm * ms_llm0, N_CASC1: ms_tf + frac_llm * ms_llm1,
    }
    tabla = pd.DataFrame([evaluacion.evaluar_enfoque(n, y_co, p, latencias[n], "test_comun") for n, p in P.items()])
    tabla.to_csv(config.REPORTES / "comparacion_enfoques.csv", index=False)

    pares = [
        (N_TFIDF, N_LLM0), (N_TFIDF, N_LLM1), (N_LLM1, N_LLM0), (N_EMB, N_LLM0), (N_EMB, N_LLM1),
        (N_TFIDF, N_EMB), (N_EMB, N_TFIDF_P), (N_CASC1, N_TFIDF), (N_CASC0, N_TFIDF), (N_TFIDF, N_ZS),
    ]
    difs = pd.DataFrame([
        {"enfoque_a": a, "enfoque_b": b, **evaluacion.diferencia_pareada_f1(y_co, _pred(P[a]), _pred(P[b]))}
        for a, b in pares
    ])
    difs.to_csv(config.REPORTES / "diferencias_pareadas.csv", index=False)

    # Curva completa de la cascada sobre el conjunto común (descriptiva: el punto operativo se fijó en validación)
    filas_casc = []
    conf_co = P[N_TFIDF].max(axis=1)
    for frac in np.linspace(0.0, 1.0, 21):
        u = float(np.quantile(conf_co, frac)) if frac < 1 else 1.01
        for nombre, pl in (("zero-shot", P[N_LLM0]), ("few-shot", P[N_LLM1])):
            pc, m = cascada(P[N_TFIDF], pl, u)
            filas_casc.append({
                "llm": nombre, "pct_al_llm": float(m.mean() * 100.0),
                "f1_macro": float(f1_score(y_co, _pred(pc), average="macro")),
                "exactitud": float(np.mean(_pred(pc) == y_co)),
            })
    curva_casc = pd.DataFrame(filas_casc)
    curva_casc.to_csv(config.REPORTES / "curva_cascada.csv", index=False)

    print("6/9 · Curva few-shot, matrices de confusión y señales de confianza del LLM...")
    curva_fs = pd.DataFrame(curva_aprendizaje_few_shot(
        X_tr_emb, tr_emb["texto_limpio_tfidf"].tolist(), tr_emb["categoria"].to_numpy(),
        X_co, txt_co, y_co,
    ))
    curva_fs.to_csv(config.REPORTES / "curva_few_shot.csv", index=False)

    df_por_cat = evaluacion.metricas_por_categoria({n: P[n] for n in (N_TFIDF, N_EMB, N_LLM0, N_LLM1)}, y_co)
    df_por_cat.to_csv(config.REPORTES / "metricas_por_categoria.csv", index=False)
    cm_tf = evaluacion.tabla_confusion(y_co, P[N_TFIDF])
    cm_llm0 = evaluacion.tabla_confusion(y_co, P[N_LLM0])
    cm_llm1 = evaluacion.tabla_confusion(y_co, P[N_LLM1])
    cm_tf.to_csv(config.REPORTES / "matriz_confusion_tfidf.csv")
    cm_llm0.to_csv(config.REPORTES / "matriz_confusion_llm_zero_shot.csv")
    cm_llm1.to_csv(config.REPORTES / "matriz_confusion_llm_few_shot.csv")
    confusiones = []
    for nombre, cm in ((N_TFIDF, cm_tf), (N_LLM0, cm_llm0), (N_LLM1, cm_llm1)):
        errores = int(cm.to_numpy().sum() - np.trace(cm.to_numpy()))
        for real in config.CATEGORIAS:
            for pred in config.CATEGORIAS:
                if real != pred and cm.loc[real, pred] > 0:
                    confusiones.append({"enfoque": nombre, "real": real, "predicha": pred,
                                        "n": int(cm.loc[real, pred]), "pct_de_errores": float(cm.loc[real, pred] / errores * 100.0)})
    pd.DataFrame(confusiones).sort_values(["enfoque", "n"], ascending=[True, False]).to_csv(
        config.REPORTES / "confusiones_principales.csv", index=False)

    acierto0 = _pred(P[N_LLM0]) == y_co
    acierto1 = _pred(P[N_LLM1]) == y_co
    acuerdo_modos = _pred(P[N_LLM0]) == _pred(P[N_LLM1])
    senales = []
    for nombre_llm, acierto, pl in ((N_LLM0, acierto0, P[N_LLM0]), (N_LLM1, acierto1, P[N_LLM1])):
        for senal, valor in (
            ("Probabilidad de la ficha (logprobs del LLM)", pl.max(axis=1)),
            ("Acuerdo zero-shot = few-shot", acuerdo_modos.astype(float)),
            ("Acuerdo LLM = TF-IDF", (_pred(pl) == _pred(P[N_TFIDF])).astype(float)),
            ("Probabilidad calibrada de TF-IDF", P[N_TFIDF].max(axis=1)),
        ):
            senales.append({
                "llm": nombre_llm, "senal": senal,
                "auroc_acierto_vs_error": evaluacion.auroc_senal(acierto, valor),
                "pct_con_senal_maxima": float(np.mean(valor >= 0.99) * 100.0),
            })
    senales = pd.DataFrame(senales)
    senales.to_csv(config.REPORTES / "senales_confianza_llm.csv", index=False)
    acuerdo = {
        "pct_acuerdo_zero_few": float(acuerdo_modos.mean() * 100.0),
        "exactitud_few_si_acuerdan": float(acierto1[acuerdo_modos].mean()),
        "exactitud_few_si_discrepan": float(acierto1[~acuerdo_modos].mean()) if (~acuerdo_modos).any() else float("nan"),
        "n_discrepan": int((~acuerdo_modos).sum()),
    }

    print("7/9 · Deriva temporal, calibración y abstención (umbral elegido en validación)...")
    filas_deriva = []
    pred_vieja = m_tf.predecir(te_vieja["texto_limpio_tfidf"].tolist())
    pred_nueva = m_tf.predecir(te["texto_limpio_tfidf"].tolist())
    for etiqueta, d, pr in (("misma época que train (2015–T1 2017, reserva aleatoria)", te_vieja, pred_vieja),
                            ("recientes (T4 2017 – 2018)", te, pred_nueva)):
        y_d = d["categoria"].to_numpy()
        lo, hi = evaluacion.bootstrap_ic_f1_macro(y_d, np.asarray(pr))
        filas_deriva.append({"conjunto": etiqueta, "n_quejas": len(d), "f1_macro": float(f1_score(y_d, pr, average="macro")),
                             "f1_ic95_bajo": lo, "f1_ic95_alto": hi, "exactitud": float(np.mean(np.asarray(pr) == y_d))})
    deriva = pd.DataFrame(filas_deriva)
    dif_deriva = _bootstrap_dif_independiente(te["categoria"], pred_nueva, te_vieja["categoria"], pred_vieja)
    deriva.to_csv(config.REPORTES / "deriva_temporal.csv", index=False)
    # Macro-F1 por trimestre de recepción (val + test, todos posteriores al corte de entrenamiento)
    posteriores = pd.concat([val, te], ignore_index=True)
    posteriores["pred"] = m_tf.predecir(posteriores["texto_limpio_tfidf"].tolist())
    posteriores["trimestre"] = posteriores["fecha_recepcion"].dt.to_period("Q").astype(str)
    por_trim = []
    for q, g in posteriores.groupby("trimestre"):
        if len(g) >= 300:
            por_trim.append({"trimestre": q, "n_quejas": len(g),
                             "f1_macro": float(f1_score(g["categoria"], g["pred"], average="macro")),
                             "exactitud": float(np.mean(g["categoria"] == g["pred"]))})
    por_trim = pd.DataFrame(por_trim)
    por_trim.to_csv(config.REPORTES / "deriva_por_trimestre.csv", index=False)
    f1_clase_deriva = pd.DataFrame({
        "categoria": config.CATEGORIAS,
        "f1_misma_epoca": f1_score(te_vieja["categoria"], pred_vieja, labels=config.CATEGORIAS, average=None),
        "f1_recientes": f1_score(te["categoria"], pred_nueva, labels=config.CATEGORIAS, average=None),
    })
    f1_clase_deriva.to_csv(config.REPORTES / "deriva_f1_por_mesa.csv", index=False)

    calib = []
    confiab = []
    for nombre, p_sin, p_con in (
        (N_TFIDF, m_tf.predecir_proba(txt_co, calibrado=False), P[N_TFIDF]),
        (N_EMB, m_emb.predecir_proba_desde_vectores(X_co, calibrado=False), P[N_EMB]),
    ):
        temp = m_tf.temperatura_ if nombre == N_TFIDF else m_emb.temperatura_
        calib.append({"enfoque": nombre, "temperatura": temp,
                      "brier_sin_calibrar": evaluacion.brier_multiclase(y_co, p_sin),
                      "brier_calibrado": evaluacion.brier_multiclase(y_co, p_con),
                      "ece_sin_calibrar": evaluacion.ece(y_co, p_sin), "ece_calibrado": evaluacion.ece(y_co, p_con),
                      "confianza_media_sin_calibrar": float(p_sin.max(axis=1).mean()),
                      "confianza_media_calibrado": float(p_con.max(axis=1).mean()),
                      "exactitud": float(np.mean(_pred(p_con) == y_co))})
        confiab.append(evaluacion.tabla_confiabilidad(y_co, p_sin, f"{nombre} · sin calibrar"))
        confiab.append(evaluacion.tabla_confiabilidad(y_co, p_con, f"{nombre} · calibrado"))
    for nombre in (N_LLM0, N_LLM1):
        calib.append({"enfoque": nombre, "temperatura": float("nan"),
                      "brier_sin_calibrar": evaluacion.brier_multiclase(y_co, P[nombre]), "brier_calibrado": float("nan"),
                      "ece_sin_calibrar": evaluacion.ece(y_co, P[nombre]), "ece_calibrado": float("nan"),
                      "confianza_media_sin_calibrar": float(P[nombre].max(axis=1).mean()), "confianza_media_calibrado": float("nan"),
                      "exactitud": float(np.mean(_pred(P[nombre]) == y_co))})
        confiab.append(evaluacion.tabla_confiabilidad(y_co, P[nombre], f"{nombre} · logprobs"))
    calib = pd.DataFrame(calib)
    calib.to_csv(config.REPORTES / "calibracion.csv", index=False)
    confiab = pd.concat(confiab, ignore_index=True)
    confiab.to_csv(config.REPORTES / "curvas_confiabilidad.csv", index=False)

    p_val_tf = m_tf.predecir_proba(val_emb["texto_limpio_tfidf"].tolist())
    p_val_emb = m_emb.predecir_proba_desde_vectores(X_val_emb)
    abst = []
    for nombre, p_val, p_test in ((N_TFIDF, p_val_tf, P[N_TFIDF]), (N_EMB, p_val_emb, P[N_EMB])):
        u = evaluacion.umbral_optimo_costo(y_val, p_val)
        abst.append({"enfoque": nombre, "elegido_en": f"val_emb (n={len(val_emb)})", "evaluado_en": f"test_comun (n={len(co)})",
                     **evaluacion.aplicar_umbral_abstencion(y_co, p_test, u)})
    abst = pd.DataFrame(abst)
    abst.to_csv(config.REPORTES / "abstencion_umbral_validacion.csv", index=False)
    tau_abst = float(abst.loc[abst["enfoque"] == N_TFIDF, "umbral"].iloc[0])
    curva_abs = evaluacion.curva_abstencion_triaje(y_co, P[N_TFIDF], nombre_modelo=N_TFIDF)
    curva_abs.to_csv(config.REPORTES / "curva_abstencion_triaje.csv", index=False)

    print("8/9 · Urgencia (segunda tarea) y transferencia al español...")
    m_urg = ModeloUrgenciaTFIDF().ajustar(tr["texto_limpio_tfidf"].tolist(), tr["urgente"].to_numpy())
    m_urg.elegir_umbral(val["texto_limpio_tfidf"].tolist(), val["urgente"].to_numpy())
    urg = []
    p_urg_te = m_urg.proba(te["texto_limpio_tfidf"].tolist())
    urg.append(evaluacion.evaluar_binario("Reglas (palabras clave)", te["urgente"], urgencia_por_reglas(te["texto_queja"].tolist()), conjunto="test_reciente"))
    urg.append(evaluacion.evaluar_binario("TF-IDF + LR (umbral en val)", te["urgente"], p_urg_te >= m_urg.umbral_, p_urg_te, "test_reciente"))
    p_urg_co = m_urg.proba(txt_co)
    urg.append(evaluacion.evaluar_binario("Reglas (palabras clave)", co["urgente"], urgencia_por_reglas(co["texto_queja"].tolist()), conjunto="test_comun"))
    urg.append(evaluacion.evaluar_binario("TF-IDF + LR (umbral en val)", co["urgente"], p_urg_co >= m_urg.umbral_, p_urg_co, "test_comun"))
    urg.append(evaluacion.evaluar_binario(N_LLM0, co["urgente"], r0["urgente"], r0["p_urgente"], "test_comun"))
    urg.append(evaluacion.evaluar_binario(N_LLM1, co["urgente"], r1["urgente"], r1["p_urgente"], "test_comun"))
    urg = pd.DataFrame(urg)
    urg.to_csv(config.REPORTES / "urgencia.csv", index=False)
    pred_urg_te = p_urg_te >= m_urg.umbral_
    reglas_te = urgencia_por_reglas(te["texto_queja"].tolist())
    urg_tipo = []
    for tipo, g in te[te["urgente"]].groupby("tipo_urgencia"):
        urg_tipo.append({"tipo_urgencia": tipo, "n": len(g),
                         "exhaustividad_tfidf": float(pred_urg_te[g.index].mean()),
                         "exhaustividad_reglas": float(reglas_te[g.index].mean())})
    urg_tipo = pd.DataFrame(urg_tipo)
    urg_tipo.to_csv(config.REPORTES / "urgencia_por_tipo.csv", index=False)

    espanol = None
    pct_al_llm_es = {}
    if hay_es or permitir_ollama:
        en_pares = comun_todo.assign(id_queja=comun_todo["id_queja"].astype(str)).set_index("id_queja").loc[df_es["id_queja"].astype(str)].reset_index()
        X_en_pares = emb(en_pares)
        y_es = df_es["categoria"].to_numpy()
        P_es = {
            N_TFIDF: (m_tf.predecir_proba(df_es["texto_limpio_tfidf"].tolist()), m_tf.predecir_proba(en_pares["texto_limpio_tfidf"].tolist())),
            N_ZS: (m_zs.predecir_proba_desde_vectores(X_es), m_zs.predecir_proba_desde_vectores(X_en_pares)),
            N_EMB: (m_emb.predecir_proba_desde_vectores(X_es), m_emb.predecir_proba_desde_vectores(X_en_pares)),
            N_LLM0: (r0_es["probs"], llm0.predecir_lote(en_pares["texto_corto"].tolist(), permitir_ollama)["probs"]),
            N_LLM1: (r1_es["probs"], llm1.predecir_lote(en_pares["texto_corto"].tolist(), permitir_ollama)["probs"]),
        }
        casc_es, al_llm_es = cascada(P_es[N_TFIDF][0], P_es[N_LLM1][0], tau_casc)
        casc_en, al_llm_en = cascada(P_es[N_TFIDF][1], P_es[N_LLM1][1], tau_casc)
        P_es[N_CASC1] = (casc_es, casc_en)
        pct_al_llm_es = {"espanol": float(al_llm_es.mean() * 100.0), "ingles": float(al_llm_en.mean() * 100.0)}
        filas_es = []
        for nombre, (p_es, p_en) in P_es.items():
            a_es, a_en = _pred(p_es) == y_es, _pred(p_en) == y_es
            lo, hi = evaluacion.bootstrap_ic(lambda a: float(np.mean(a)), a_es)
            filas_es.append({"enfoque": nombre, "n_quejas": len(y_es),
                             "exactitud_ingles": float(a_en.mean()), "exactitud_espanol": float(a_es.mean()),
                             "exactitud_espanol_ic95_bajo": lo, "exactitud_espanol_ic95_alto": hi,
                             "aciertos_ingles": int(a_en.sum()), "aciertos_espanol": int(a_es.sum()),
                             "f1_macro_espanol": float(f1_score(y_es, _pred(p_es), average="macro"))})
        espanol = pd.DataFrame(filas_es)
        espanol.to_csv(config.REPORTES / "transferencia_espanol.csv", index=False)

    print("9/9 · Figuras, artefacto y resultados.json...")
    graficos.fig_01_datos_y_particiones(res_cat, particiones, resumen_fuente)
    graficos.fig_02_comparacion_enfoques(tabla)
    graficos.fig_03_curva_few_shot(curva_fs, tabla, N_TFIDF, N_LLM0, N_LLM1)
    graficos.fig_04_deriva_temporal(deriva, por_trim, f1_clase_deriva)
    graficos.fig_05_matrices_confusion(cm_tf, cm_llm1, N_TFIDF, N_LLM1)
    graficos.fig_06_cascada(curva_casc, tabla, frac_llm, N_TFIDF, N_CASC0, N_CASC1)
    graficos.fig_07_confiabilidad(confiab, calib)
    graficos.fig_08_senales_llm(senales)
    graficos.fig_09_urgencia(urg)
    if espanol is not None:
        graficos.fig_10_espanol(espanol)

    metadata = {
        "nombre_modelo": "clasificador_quejas_cfpb",
        "version": "2.0.0",
        "fuente_datos": "CFPB Consumer Complaint Database (milesbutler/consumer_complaints, Parquet)",
        "categorias": config.CATEGORIAS,
        "etiquetas_categoria": config.ETIQUETAS_CATEGORIA,
        "modelos_ollama": {"embeddings": config.MODELO_EMB, "llm": config.MODELO_LLM},
        "tamanos_particion": {
            "corpus_total": int(len(df)),
            "train_tfidf": int(len(tr)),
            "train_embeddings": int(len(tr_emb)),
            "val": int(len(val)),
            "val_emb": int(len(val_emb)),
            "test_reciente": int(len(te)),
            "test_misma_epoca": int(len(te_vieja)),
            "test_comun": int(len(co)),
            "test_espanol": int(len(df_es)),
        },
        "fechas": {
            "train_hasta": "2017-03-31",
            "val": "2017-04-01 a 2017-09-30",
            "test_desde": "2017-10-01",
            "fecha_min": resumen_fuente["fecha_min"],
            "fecha_max": resumen_fuente["fecha_max"],
        },
        "umbral_abstencion": round(tau_abst, 3),
        "umbral_cascada": round(tau_casc, 4),
        "fraccion_al_llm_objetivo": FRACCION_AL_LLM,
        "umbral_urgencia": round(m_urg.umbral_, 3),
        "temperaturas": {"tfidf": round(m_tf.temperatura_, 4), "bge_m3_lr": round(m_emb.temperatura_, 4)},
        "supuestos_operativos": {
            "costo_mal_enrutamiento_usd": config.COSTO_MAL_ENRUTAMIENTO_USD,
            "costo_triaje_humano_usd": config.COSTO_TRIAJE_HUMANO_USD,
        },
        "comparacion_enfoques": tabla.to_dict(orient="records"),
    }
    joblib.dump({
        "modelo_tfidf": m_tf,
        "modelo_emb_sup": m_emb,
        "modelo_emb_zs": m_zs,
        "modelo_urgencia": m_urg,
        "umbral_optimo": tau_abst,
        "umbral_cascada": tau_casc,
        "ejemplos_few_shot": ejemplos,
        "metadata": metadata,
    }, config.MODELOS / "clasificador_quejas.joblib")
    (config.MODELOS / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    resultados = {
        "metadata": metadata,
        "resumen_fuente": resumen_fuente,
        "particiones_temporales": particiones.reset_index().to_dict(orient="records"),
        "resumen_categorias_cfpb": res_cat.to_dict(orient="records"),
        "comparacion_enfoques": tabla.to_dict(orient="records"),
        "diferencias_pareadas": difs.to_dict(orient="records"),
        "cascada": {"umbral": tau_casc, "pct_al_llm_test": frac_llm * 100.0, "curva": curva_casc.to_dict(orient="records")},
        "curva_few_shot": curva_fs.to_dict(orient="records"),
        "metricas_por_categoria": df_por_cat.to_dict(orient="records"),
        "confusiones_principales": pd.read_csv(config.REPORTES / "confusiones_principales.csv").to_dict(orient="records"),
        "senales_confianza_llm": senales.to_dict(orient="records"),
        "acuerdo_zero_few": acuerdo,
        "confianza_autodeclarada": estadistica_confianza_autodeclarada(),
        "llm_respuestas_invalidas": int((~r0["validas"]).sum() + (~r1["validas"]).sum()),
        "deriva_temporal": deriva.to_dict(orient="records"),
        "deriva_diferencia_ic95": list(dif_deriva),
        "deriva_por_trimestre": por_trim.to_dict(orient="records"),
        "deriva_f1_por_mesa": f1_clase_deriva.to_dict(orient="records"),
        "calibracion": calib.to_dict(orient="records"),
        "abstencion_umbral_validacion": abst.to_dict(orient="records"),
        "urgencia": urg.to_dict(orient="records"),
        "urgencia_por_tipo": urg_tipo.to_dict(orient="records"),
        "transferencia_espanol": espanol.to_dict(orient="records") if espanol is not None else [],
        "espanol_pct_al_llm_en_cascada": pct_al_llm_es,
    }
    (config.REPORTES / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Listo en {time.perf_counter() - t0:.1f} s (test_comun = {len(co)} quejas).")


if __name__ == "__main__":
    main()
