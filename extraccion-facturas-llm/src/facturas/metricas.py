"""
Comparación de una extracción con la verdad.

* Campo correcto:
    - textos: iguales tras normalizar mayúsculas, tildes, puntuación y espacios (el ticket imprime "MARÍA", la verdad "María")
    - identificadores: mismos dígitos
    - montos: diferencia ≤ 0,005
    - null: correcto solo si la verdad también es null (devolver null cuando el dato existe es un error, e
      inventar un dato que no está en la factura también)
* Ítems: emparejamiento 1 a 1 (codicioso por similitud de descripción). Un ítem predicho es correcto si su
  descripción se parece ≥ 0,85 a la de un ítem real y cantidad, precio unitario y total coinciden.
  Precisión = correctos / predichos, exhaustividad = correctos / reales, F1.
* Factura perfecta: todos los campos correctos y F1 de ítems = 1.
* Intervalos de confianza 95 %: bootstrap de facturas (percentiles) para la exactitud de campos y el F1 de ítems,
  e intervalo de Wilson para la proporción de facturas perfectas (con n pequeño el bootstrap de una proporción en
  0 % o 100 % daría un intervalo de ancho cero).
"""
from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from .esquema import CAMPOS, CAMPOS_MONTO, Factura

UMBRAL_DESC = 0.85
REMUESTREOS, SEMILLA = 2000, 2026


def norm(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().casefold()
    s = re.sub(r"[^\w\s]", " ", s)                  # "CIA. LTDA" = "Cía. Ltda." (el OCR pierde puntos)
    return re.sub(r"\s+", " ", s).strip()


def campo_correcto(campo: str, pred, verdad) -> bool:
    if verdad is None or pred is None:
        return verdad is None and pred is None
    if campo in CAMPOS_MONTO:
        return abs(float(pred) - float(verdad)) <= 0.005
    if campo == "fecha_emision":
        return str(pred) == str(verdad)
    return norm(pred) == norm(verdad)


def similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def comparar_items(pred: list[dict], verdad: list[dict]) -> dict:
    pares = sorted(((similitud(p["descripcion"], v["descripcion"]), i, j)
                    for i, p in enumerate(pred) for j, v in enumerate(verdad)), reverse=True)
    usados_p, usados_v, correctos, montos_ok = set(), set(), 0, 0
    for s, i, j in pares:
        if s < UMBRAL_DESC or i in usados_p or j in usados_v:
            continue
        usados_p.add(i)
        usados_v.add(j)
        p, v = pred[i], verdad[j]
        ok = all(abs(float(p[k]) - float(v[k])) <= 0.005 for k in ("cantidad", "precio_unitario", "total"))
        correctos += ok
    n_p, n_v = len(pred), len(verdad)
    prec = correctos / n_p if n_p else (1.0 if not n_v else 0.0)
    rec = correctos / n_v if n_v else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"items_pred": n_p, "items_real": n_v, "items_ok": correctos, "items_precision": prec,
            "items_recall": rec, "items_f1": f1}


def comparar(pred: Factura | dict | None, verdad: dict) -> dict:
    """Una fila por factura: un booleano por campo + métricas de ítems + factura_perfecta."""
    if pred is None:
        fila = {c: verdad[c] is None for c in CAMPOS}
        fila.update(comparar_items([], verdad["items"]))
    else:
        p = pred.model_dump(mode="json") if isinstance(pred, Factura) else pred
        fila = {c: campo_correcto(c, p.get(c), verdad[c]) for c in CAMPOS}
        fila.update(comparar_items(p.get("items") or [], verdad["items"]))
    fila["campos_ok"] = sum(fila[c] for c in CAMPOS)
    fila["factura_perfecta"] = fila["campos_ok"] == len(CAMPOS) and fila["items_f1"] == 1.0
    return fila


def f1_micro(filas: list[dict]) -> dict:
    ok = sum(f["items_ok"] for f in filas)
    p = sum(f["items_pred"] for f in filas)
    r = sum(f["items_real"] for f in filas)
    prec, rec = (ok / p if p else 0.0), (ok / r if r else 0.0)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def wilson(exitos: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = exitos / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    mitad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centro - mitad), 4), round(min(1.0, centro + mitad), 4)]


def _si(v) -> int:
    return int(v == "True" or v is True or v == 1)


def ic95(filas: list[dict]) -> dict:
    """IC 95 % de exactitud de campos, F1 de ítems (micro) y facturas perfectas para un grupo de facturas.
    Acepta las filas de `comparar` o las del CSV de detalle (los booleanos pueden venir como texto)."""
    n = len(filas)
    if not n:
        return {}
    ok = np.array([int(float(f["campos_ok"])) for f in filas])
    i_ok = np.array([float(f["items_ok"]) for f in filas])
    i_p = np.array([float(f["items_pred"]) for f in filas])
    i_r = np.array([float(f["items_real"]) for f in filas])
    perf = np.array([_si(f["factura_perfecta"]) for f in filas])
    idx = np.random.default_rng(SEMILLA).integers(0, n, size=(REMUESTREOS, n))
    exac = ok[idx].sum(1) / (n * len(CAMPOS))
    a, p, r = i_ok[idx].sum(1), i_p[idx].sum(1), i_r[idx].sum(1)
    prec, rec = np.divide(a, p, out=np.zeros(len(a)), where=p > 0), np.divide(a, r, out=np.zeros(len(a)), where=r > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(a)), where=(prec + rec) > 0)
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"exactitud_campos": pct(exac), "items_f1": pct(f1), "facturas_perfectas": wilson(int(perf.sum()), n)}
