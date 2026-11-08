"""
Métricas de ABSA. Cada reseña se representa como {"global": pol, "aspectos": {aspecto: pol}}.

* **Detección de aspectos**: precisión, exhaustividad y F1 sobre pares (reseña, aspecto), micro y por aspecto.
* **Aspecto + polaridad**: F1 sobre tríos (reseña, aspecto, polaridad); un acierto exige ambas cosas.
* **Exactitud de polaridad**: sobre los aspectos que el método detectó bien (aísla la polaridad de la detección).
* **Sentimiento global**: exactitud y F1 macro.
* Intervalos de confianza al 95 % por bootstrap de reseñas.
"""
from __future__ import annotations

import numpy as np

from . import config


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "exhaustividad": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _conteos(verdad: dict, pred: dict, ids: list[str]) -> dict:
    c = {"tp": 0, "fp": 0, "fn": 0, "tp_pol": 0, "fp_pol": 0, "fn_pol": 0, "pol_ok": 0, "glob_ok": 0}
    for i in ids:
        v, p = verdad[i]["aspectos"], pred.get(i, {"aspectos": {}})["aspectos"]
        comunes = v.keys() & p.keys()
        c["tp"] += len(comunes)
        c["fp"] += len(p.keys() - v.keys())
        c["fn"] += len(v.keys() - p.keys())
        ok = sum(v[a] == p[a] for a in comunes)
        c["pol_ok"] += ok
        c["tp_pol"] += ok
        c["fp_pol"] += len(p) - ok
        c["fn_pol"] += len(v) - ok
        c["glob_ok"] += verdad[i]["global"] == pred.get(i, {}).get("global")
    return c


def _resumen(c: dict, n: int) -> dict:
    det = _prf(c["tp"], c["fp"], c["fn"])
    tri = _prf(c["tp_pol"], c["fp_pol"], c["fn_pol"])
    return {"f1_aspectos": det["f1"], "precision_aspectos": det["precision"],
            "exhaustividad_aspectos": det["exhaustividad"], "f1_aspecto_polaridad": tri["f1"],
            "exactitud_polaridad": c["pol_ok"] / c["tp"] if c["tp"] else 0.0,
            "exactitud_global": c["glob_ok"] / n if n else 0.0}


def f1_macro_global(verdad: dict, pred: dict, ids: list[str]) -> float:
    f1s = []
    for clase in config.POLARIDADES:
        tp = sum(verdad[i]["global"] == clase and pred.get(i, {}).get("global") == clase for i in ids)
        fp = sum(verdad[i]["global"] != clase and pred.get(i, {}).get("global") == clase for i in ids)
        fn = sum(verdad[i]["global"] == clase and pred.get(i, {}).get("global") != clase for i in ids)
        f1s.append(_prf(tp, fp, fn)["f1"])
    return float(np.mean(f1s))


def evaluar(verdad: dict, pred: dict, ids: list[str] | None = None, bootstrap: int = 1000,
            semilla: int = 0) -> dict:
    """Métricas globales con IC 95 % (bootstrap por reseña) y F1 por aspecto."""
    ids = list(ids or verdad)
    res = _resumen(_conteos(verdad, pred, ids), len(ids))
    res["f1_macro_global"] = f1_macro_global(verdad, pred, ids)
    if bootstrap:
        rng = np.random.default_rng(semilla)
        muestras = {k: [] for k in res if k != "f1_macro_global"}
        arr = np.array(ids)
        for _ in range(bootstrap):
            b = list(rng.choice(arr, len(arr), replace=True))
            r = _resumen(_conteos(verdad, pred, b), len(b))
            for k in muestras:
                muestras[k].append(r[k])
        res["ic95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in muestras.items()}
    res["por_aspecto"] = por_aspecto(verdad, pred, ids)
    res["f1_macro_aspectos"] = float(np.mean([d["f1"] for d in res["por_aspecto"].values() if d["soporte"] >= 5]))
    return res


def por_aspecto(verdad: dict, pred: dict, ids: list[str]) -> dict:
    salida = {}
    for a in config.ASPECTOS:
        tp = fp = fn = pol_ok = 0
        for i in ids:
            v, p = verdad[i]["aspectos"], pred.get(i, {"aspectos": {}})["aspectos"]
            if a in v and a in p:
                tp += 1
                pol_ok += v[a] == p[a]
            elif a in p:
                fp += 1
            elif a in v:
                fn += 1
        d = _prf(tp, fp, fn)
        d["soporte"] = tp + fn
        d["exactitud_polaridad"] = pol_ok / tp if tp else None
        salida[a] = d
    return salida


def kappa(a: list, b: list) -> float:
    """Kappa de Cohen entre dos listas de etiquetas."""
    a, b = list(a), list(b)
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    clases = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in clases)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def acuerdo(pasada1: dict, pasada2: dict) -> dict:
    """Acuerdo entre dos pasadas de etiquetado sobre las mismas reseñas."""
    ids = sorted(pasada1.keys() & pasada2.keys())
    c = _conteos(pasada1, pasada2, ids)
    det = _prf(c["tp"], c["fp"], c["fn"])
    kappas = {}
    for asp in config.ASPECTOS:
        x = [asp in pasada1[i]["aspectos"] for i in ids]
        y = [asp in pasada2[i]["aspectos"] for i in ids]
        if sum(x) + sum(y) >= 4:
            kappas[asp] = kappa(x, y)
    comunes = [(pasada1[i]["aspectos"][a], pasada2[i]["aspectos"][a])
               for i in ids for a in pasada1[i]["aspectos"].keys() & pasada2[i]["aspectos"].keys()]
    return {
        "n_resenas": len(ids),
        "f1_aspectos_entre_pasadas": det["f1"],
        "kappa_aspectos_medio": float(np.mean(list(kappas.values()))),
        "kappa_por_aspecto": kappas,
        "acuerdo_polaridad": float(np.mean([x == y for x, y in comunes])) if comunes else None,
        "kappa_polaridad": kappa([x for x, _ in comunes], [y for _, y in comunes]),
        "kappa_global": kappa([pasada1[i]["global"] for i in ids], [pasada2[i]["global"] for i in ids]),
        "acuerdo_global": float(np.mean([pasada1[i]["global"] == pasada2[i]["global"] for i in ids])),
    }
