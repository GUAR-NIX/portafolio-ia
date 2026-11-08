"""
Evaluación de la recuperación por pieza: BM25, vectorial, híbrido (RRF) e híbrido + re-ranking, sin sinónimos, con la lista inicial (v1) y con la corregida (v2).

    python -m rag_laboral.evaluar_recuperacion            → reports/recuperacion.json y reports/recuperacion_detalle.csv
    python -m rag_laboral.evaluar_recuperacion --ic       → recalcula solo los IC bootstrap desde recuperacion_detalle.csv

Métricas por pregunta respondible (artículo correcto = cualquiera de `articulos_correctos`):
* Recall@k: ¿algún artículo correcto está entre los k primeros?
* MRR@20: 1 / posición del primer artículo correcto (0 si no está en los 20 primeros).
Los IC 95 % de Recall@5 y MRR@20 salen de un bootstrap de preguntas (percentiles, 2 000 remuestreos, semilla fija); la
comparación entre métodos es pareada (mismas preguntas), porque un método y otro fallan en preguntas parecidas.
Con Ollama abierto y RAG_GRABAR=1 los embeddings de consultas y los puntajes del re-ranker se guardan en data/indice/:
después corre igual sin Ollama ni modelo descargado (RAG_PROVEEDOR=grabado).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config
from .recuperar import Buscador

METODOS = ["bm25", "vectorial", "hibrido", "rerank"]
PROFUNDIDAD = 20
REMUESTREOS, SEMILLA = 2000, 2026
COMPARACIONES = (("rerank", "vectorial"), ("rerank", "hibrido"), ("vectorial", "hibrido"))
VARIANTES_SINONIMOS = (False, "v1", "v2")


def cargar_preguntas() -> pd.DataFrame:
    q = pd.read_csv(config.PREGUNTAS, dtype=str).fillna("")
    q["correctos"] = q["articulos_correctos"].map(lambda s: [x for x in s.split("|") if x])
    return q


def mismo_articulo(i: str) -> str:
    """El PDF trae dos «Art. 459» idénticos: el índice los guarda como CT-459 y CT-459#2; al calificar son el mismo."""
    return i.split("#")[0]


def posicion(ids: list[str], correctos: list[str]) -> int | None:
    for n, i in enumerate(ids, 1):
        if mismo_articulo(i) in correctos:
            return n
    return None


def _por_pregunta(posiciones) -> tuple[np.ndarray, np.ndarray]:
    """(acierto en los 5 primeros, 1/posición si cae en los 20 primeros) para cada pregunta."""
    p = np.array([x if x is not None and not pd.isna(x) else 10 ** 6 for x in posiciones], dtype=float)
    return (p <= 5).astype(float), np.where(p <= 20, 1 / p, 0.0)


def _ic(valores: np.ndarray, idx: np.ndarray) -> list[float]:
    medias = valores[idx].mean(axis=1)
    return [round(float(np.percentile(medias, 2.5)), 4), round(float(np.percentile(medias, 97.5)), 4)]


def _indices(n: int) -> np.ndarray:
    return np.random.default_rng(SEMILLA).integers(0, n, size=(REMUESTREOS, n))


def metricas(posiciones: list[int | None], bootstrap: bool = True) -> dict:
    p = [x if x is not None and not pd.isna(x) else 10 ** 6 for x in posiciones]
    n = len(p)
    res = {
        "n": n,
        "recall@1": round(sum(x <= 1 for x in p) / n, 4),
        "recall@5": round(sum(x <= 5 for x in p) / n, 4),
        "recall@10": round(sum(x <= 10 for x in p) / n, 4),
        "recall@20": round(sum(x <= 20 for x in p) / n, 4),
        "mrr@20": round(float(np.mean([1 / x if x <= 20 else 0 for x in p])), 4),
    }
    if bootstrap:
        r5, rr = _por_pregunta(posiciones)
        idx = _indices(n)
        res["ic95_recall@5"], res["ic95_mrr@20"] = _ic(r5, idx), _ic(rr, idx)
    return res


def diferencias_pareadas(d: pd.DataFrame, sinonimos: str = "v2") -> list[dict]:
    """Diferencia de Recall@5 y MRR@20 entre pares de métodos sobre las mismas preguntas, con IC 95 % bootstrap."""
    filas = []
    for c, g in d[(d.tipo == "respondible") & (d.sinonimos == sinonimos)].groupby("conjunto"):
        pos = {m: g[g.metodo == m].set_index("id").posicion for m in METODOS}
        for a, b in COMPARACIONES:
            ids = pos[a].index.intersection(pos[b].index)
            ra, ma = _por_pregunta(pos[a].loc[ids].tolist())
            rb, mb = _por_pregunta(pos[b].loc[ids].tolist())
            idx = _indices(len(ids))
            for nombre, x, y in (("recall@5", ra, rb), ("mrr@20", ma, mb)):
                dif = x - y
                filas.append({"conjunto": c, "sinonimos": sinonimos, "a": a, "b": b, "metrica": nombre, "n": len(ids),
                              "diferencia": round(float(dif.mean()), 4), "ic95": _ic(dif, idx)})
    return filas


def recalcular_ic() -> dict:
    """Agrega los IC a reports/recuperacion.json desde recuperacion_detalle.csv, sin volver a buscar."""
    d = pd.read_csv(config.REPORTS / "recuperacion_detalle.csv")
    ruta = config.REPORTS / "recuperacion.json"
    resumen = json.loads(ruta.read_text(encoding="utf-8"))
    for fila in resumen["resultados"]:
        g = d[(d.tipo == "respondible") & (d.conjunto == fila["conjunto"]) & (d.metodo == fila["metodo"])
              & (d.sinonimos == fila["sinonimos"])]
        m = metricas(g.posicion.tolist())
        assert m["recall@5"] == fila["recall@5"] and m["mrr@20"] == fila["mrr@20"], fila
        fila["ic95_recall@5"], fila["ic95_mrr@20"] = m["ic95_recall@5"], m["ic95_mrr@20"]
    resumen["bootstrap"] = {"remuestreos": REMUESTREOS, "semilla": SEMILLA, "unidad": "pregunta"}
    resumen["diferencias_pareadas"] = diferencias_pareadas(d)
    ruta.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


def evaluar(buscador: Buscador | None = None, conjuntos=("principal", "nuevas")) -> dict:
    b = buscador or Buscador()
    q = cargar_preguntas()
    q = q[q["conjunto"].isin(conjuntos)]
    filas = []
    for _, r in q.iterrows():
        for sin in VARIANTES_SINONIMOS:
            for m in METODOS:
                if m == "rerank" and sin == "v1":   # el cross-encoder en CPU es lo caro (~1,3 pares/s): v1 vs v2 se
                    continue                        # compara en los otros tres métodos
                t0 = time.perf_counter()
                hits = b.buscar(r.pregunta, m, PROFUNDIDAD, sinonimos=sin)
                seg = time.perf_counter() - t0
                ids = [h.id for h in hits]
                filas.append({"id": r.id, "conjunto": r.conjunto, "tipo": r.tipo, "tema": r.tema, "metodo": m,
                              "sinonimos": sin or "no", "posicion": posicion(ids, r.correctos) if r.correctos else None,
                              "top5": "|".join(ids[:5]), "puntaje_top1": round(hits[0].puntaje, 5) if hits else None,
                              "segundos": round(seg, 4)})
        print(f"\r{r.id}", end="", flush=True)
    print()
    d = pd.DataFrame(filas)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(config.REPORTS / "recuperacion_detalle.csv", index=False, encoding="utf-8")
    resumen = {"preguntas": {c: int(((q.conjunto == c) & (q.tipo == "respondible")).sum()) for c in conjuntos},
               "corpus_articulos": len(b.articulos), "resultados": []}
    for (c, m, s), g in d[d.tipo == "respondible"].groupby(["conjunto", "metodo", "sinonimos"]):
        fila = {"conjunto": c, "metodo": m, "sinonimos": s, **metricas(list(g.posicion.where(g.posicion.notna(), None))),
                "ms_por_consulta": round(1000 * g.segundos.mean(), 1)}
        resumen["resultados"].append(fila)
    resumen["bootstrap"] = {"remuestreos": REMUESTREOS, "semilla": SEMILLA, "unidad": "pregunta"}
    resumen["diferencias_pareadas"] = diferencias_pareadas(d)
    (config.REPORTS / "recuperacion.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    import sys
    res = recalcular_ic() if "--ic" in sys.argv[1:] else evaluar()
    t = pd.DataFrame(res["resultados"])
    print(t.to_string(index=False))
