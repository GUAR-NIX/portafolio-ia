"""
Evaluación de la generación con qwen2.5:7b (reanudable: cada respuesta se agrega a reports/respuestas_llm.jsonl).

    RAG_GRABAR=1 python -m rag_laboral.evaluar_generacion --correr     # con Ollama: ~1 min por pregunta en CPU
    python -m rag_laboral.evaluar_generacion                           # solo métricas (lee las respuestas guardadas)

Métricas:
* cita exacta: la respuesta cita al menos un artículo correcto (preguntas respondibles);
* abstención indebida: dice que la ley no lo cubre cuando sí lo cubre;
* rechazo correcto: en las trampas responde que la ley cargada no lo cubre;
* citas inventadas: identificadores que el modelo citó sin que estuvieran en su contexto (se eliminan antes de mostrar);
* umbral del re-ranker: se simula «si el mejor puntaje < τ, rechazar sin llamar al LLM» con las mismas respuestas.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config
from .evaluar_recuperacion import cargar_preguntas, mismo_articulo
from .generar import Asistente
from .llm import ErrorLLM

SALIDA = config.REPORTS / "respuestas_llm.jsonl"


def correr(conjuntos=("principal", "nuevas"), limite: int | None = None) -> None:
    q = cargar_preguntas()
    q = q[q.conjunto.isin(conjuntos)]
    hechas = set()
    if SALIDA.exists():
        for ln in SALIDA.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    hechas.add(json.loads(ln)["id"])
                except json.JSONDecodeError:
                    pass
    pendientes = q[~q.id.isin(hechas)]
    if limite:
        pendientes = pendientes.head(limite)
    asistente = Asistente()
    print(f"{len(hechas)} ya respondidas, {len(pendientes)} pendientes", flush=True)
    for n, (_, r) in enumerate(pendientes.iterrows(), 1):
        t0 = time.perf_counter()
        try:
            resp = asistente.responder(r.pregunta).model_dump()
            resp["error"] = None
        except ErrorLLM as e:
            resp = {"pregunta": r.pregunta, "error": str(e)}
        resp.update(id=r.id, segundos_total=round(time.perf_counter() - t0, 2))
        with open(SALIDA, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(pendientes)}] {r.id} {resp.get('segundos_llm', 0):.0f}s citas={resp.get('articulos_citados')}",
              flush=True)


def cargar_respuestas() -> pd.DataFrame:
    filas = [json.loads(ln) for ln in SALIDA.read_text(encoding="utf-8").splitlines() if ln.strip()]
    d = pd.DataFrame(filas).drop_duplicates("id", keep="last")
    q = cargar_preguntas()[["id", "conjunto", "tipo", "tema", "correctos", "respuesta_referencia"]]
    return q.merge(d, on="id", how="inner")


def evaluar_fila(r, umbral: float | None = None) -> dict:
    rechaza_umbral = umbral is not None and r.puntaje_max_rerank is not None and r.puntaje_max_rerank < umbral
    cubierto = bool(r.cubierto_por_la_ley) and not rechaza_umbral
    citas = [] if not cubierto else list(r.articulos_citados)
    if r.tipo == "respondible":
        return {"cita_correcta": any(mismo_articulo(c) in r.correctos for c in citas), "abstencion_indebida": not cubierto,
                "correcto_en_contexto": any(mismo_articulo(c) in r.correctos for c in r.contexto)}
    return {"rechazo_correcto": not cubierto}


def metricas(d: pd.DataFrame, umbral: float | None = None) -> dict:
    ev = pd.DataFrame([evaluar_fila(r, umbral) for r in d.itertuples()], index=d.index)
    resp = d.tipo == "respondible"
    out = {"umbral": umbral, "respondibles": int(resp.sum()), "trampas": int((~resp).sum())}
    if resp.any():
        e = ev[resp]
        out.update(
            cita_exacta=round(float(e.cita_correcta.mean()), 4),
            cita_exacta_si_estaba_en_contexto=round(float(e[e.correcto_en_contexto].cita_correcta.mean()), 4),
            abstencion_indebida=round(float(e.abstencion_indebida.mean()), 4),
        )
    if (~resp).any():
        e = ev[~resp]
        out["rechazo_correcto"] = round(float(e.rechazo_correcto.mean()), 4)
        for sub in ("trampa_cercana", "trampa_lejana"):
            m = d.tipo == sub
            if m.any():
                out[f"rechazo_{sub}"] = round(float(ev[m].rechazo_correcto.mean()), 4)
    return out


def resumen() -> dict:
    d = cargar_respuestas()
    errores = d[d.error.notna()]
    d = d[d.error.isna()].copy()
    d["articulos_citados"] = d["articulos_citados"].map(list)
    res = {"modelo": config.MODELO, "respuestas": len(d), "errores": len(errores),
           "citas_inventadas_total": int(d.citas_eliminadas.map(len).sum()),
           "respuestas_con_cita_inventada": int((d.citas_eliminadas.map(len) > 0).sum()),
           "segundos_llm": {"mediana": round(float(d[d.segundos_llm > 0].segundos_llm.median()), 1),
                            "p90": round(float(d[d.segundos_llm > 0].segundos_llm.quantile(0.9)), 1)},
           "por_conjunto": {}, "umbral": {}}
    # Umbral del re-ranker: se elige en el conjunto principal y se reporta en el de preguntas nuevas
    principal = d[d.conjunto == "principal"]
    candidatos = sorted(set(np.round(principal.puntaje_max_rerank.dropna(), 3)))
    mejor, mejor_valor = None, -1.0
    for u in [None, *candidatos]:
        m = metricas(principal, u)
        valor = m["cita_exacta"] + m["rechazo_correcto"] * 0.5   # una trampa vale media pregunta respondible
        if valor > mejor_valor + 1e-9:
            mejor, mejor_valor = u, valor
    res["umbral"] = {"elegido_en": "principal", "valor": mejor,
                     "criterio": "max(cita_exacta + 0.5 * rechazo_correcto) en el conjunto principal"}
    for c, g in d.groupby("conjunto"):
        res["por_conjunto"][c] = {"sin_umbral": metricas(g), "con_umbral": metricas(g, mejor)}
    # Juicio a mano: 30 respuestas al azar del principal (semilla 7) y las 20 respondibles de «nuevas»
    muestra = config.REPORTS / "fidelidad_muestra.csv"
    if muestra.exists():
        f = pd.read_csv(muestra)
        f["grupo"] = f.conjunto.str.split(" ").str[0]
        res["fidelidad_muestra"] = {}
        for grupo, g in [("total", f), *f.groupby("grupo")]:
            res["fidelidad_muestra"][grupo] = {
                "n": len(g), **{k: int(v) for k, v in g.juicio.value_counts().items()},
                "fieles": round(float(g.juicio.isin(["fiel_completa", "fiel_incompleta"]).mean()), 4),
                "con_afirmacion_falsa": round(float((g.juicio == "no_fiel").mean()), 4)}
    (config.REPORTS / "generacion.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    d.drop(columns=["correctos"]).to_csv(config.REPORTS / "generacion_detalle.csv", index=False, encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--correr", action="store_true", help="llama al LLM para las preguntas que faltan")
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    if a.correr:
        correr(limite=a.limite)
    print(json.dumps(resumen(), ensure_ascii=False, indent=2))
