"""Métricas de clasificación multiclase, intervalos de confianza por bootstrap,
matriz de confusión y simulación de enrutamiento selectivo con abstención (triaje humano)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


N_BOOT = 1000


def bootstrap_ic(
    metrica,
    *arrays: np.ndarray,
    n_boot: int = N_BOOT,
    semilla: int = config.SEMILLA,
) -> tuple[float, float]:
    """IC 95 % por bootstrap de percentiles: remuestrea filas (quejas) con reemplazo."""
    rng = np.random.default_rng(semilla)
    n = len(arrays[0])
    valores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        valores[b] = metrica(*(a[idx] for a in arrays))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def _f1_macro(y_r: np.ndarray, y_p: np.ndarray) -> float:
    return float(f1_score(y_r, y_p, average="macro", zero_division=0))


def _exactitud(y_r: np.ndarray, y_p: np.ndarray) -> float:
    return float(np.mean(y_r == y_p))


def bootstrap_ic_f1_macro(y_real: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT, semilla: int = config.SEMILLA) -> tuple[float, float]:
    return bootstrap_ic(_f1_macro, np.asarray(y_real), np.asarray(y_pred), n_boot=n_boot, semilla=semilla)


def diferencia_pareada_f1(
    y_real: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = N_BOOT, semilla: int = config.SEMILLA
) -> dict[str, float]:
    """Macro-F1(A) − Macro-F1(B) sobre LAS MISMAS quejas, con IC 95 % por bootstrap pareado."""
    y_r, a, b = (np.asarray(v, dtype=str) for v in (y_real, pred_a, pred_b))
    dif = _f1_macro(y_r, a) - _f1_macro(y_r, b)
    lo, hi = bootstrap_ic(lambda y, pa, pb: _f1_macro(y, pa) - _f1_macro(y, pb), y_r, a, b, n_boot=n_boot, semilla=semilla)
    return {"diferencia": dif, "ic95_bajo": lo, "ic95_alto": hi, "distinguible": bool(lo > 0 or hi < 0)}


def ece(y_real: list[str] | np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error de la confianza máxima (bins de igual ancho)."""
    y_idx = np.array([config.CATEGORIAS.index(str(c)) for c in y_real])
    conf = probs.max(axis=1)
    acierto = (probs.argmax(axis=1) == y_idx).astype(float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def tabla_confiabilidad(y_real: list[str] | np.ndarray, probs: np.ndarray, nombre: str, n_bins: int = 10) -> pd.DataFrame:
    """Puntos de la curva de confiabilidad: confianza media vs. exactitud observada por bin."""
    y_idx = np.array([config.CATEGORIAS.index(str(c)) for c in y_real])
    conf = probs.max(axis=1)
    acierto = (probs.argmax(axis=1) == y_idx).astype(float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    filas = []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            filas.append({
                "modelo": nombre,
                "bin_inferior": float(lo),
                "bin_superior": float(hi),
                "n": int(m.sum()),
                "confianza_media": float(conf[m].mean()),
                "exactitud": float(acierto[m].mean()),
            })
    return pd.DataFrame(filas)


def auroc_senal(acierto: np.ndarray, senal: np.ndarray) -> float:
    """AUROC de una señal de confianza para separar aciertos de errores (0,5 = no sirve)."""
    acierto = np.asarray(acierto, dtype=int)
    if acierto.min() == acierto.max():
        return float("nan")
    return float(roc_auc_score(acierto, np.asarray(senal, dtype=float)))


def evaluar_binario(nombre: str, y_real: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None, conjunto: str = "") -> dict:
    """Precisión, exhaustividad y F1 de la clase positiva (urgente) con IC 95 % por bootstrap."""
    y = np.asarray(y_real, dtype=int)
    p = np.asarray(pred, dtype=int)
    f1 = float(f1_score(y, p, zero_division=0))
    rec = float(recall_score(y, p, zero_division=0))
    prec = float(precision_score(y, p, zero_division=0))
    f1_lo, f1_hi = bootstrap_ic(lambda a, b: f1_score(a, b, zero_division=0), y, p)
    rec_lo, rec_hi = bootstrap_ic(lambda a, b: recall_score(a, b, zero_division=0), y, p)
    fila = {
        "enfoque": nombre,
        "conjunto": conjunto,
        "n_quejas": int(len(y)),
        "n_urgentes": int(y.sum()),
        "precision": prec,
        "exhaustividad": rec,
        "exhaustividad_ic95_bajo": rec_lo,
        "exhaustividad_ic95_alto": rec_hi,
        "f1": f1,
        "f1_ic95_bajo": f1_lo,
        "f1_ic95_alto": f1_hi,
        "pct_marcadas": float(p.mean() * 100.0),
        "pr_auc": float(average_precision_score(y, score)) if score is not None else float("nan"),
    }
    return fila


def brier_multiclase(y_real: list[str] | np.ndarray, probs: np.ndarray) -> float:
    """Calcula el Brier score multiclase medio: (1/N) sum_i sum_k (p_ik - y_ik)^2."""
    n = len(y_real)
    Y = np.zeros((n, len(config.CATEGORIAS)), dtype=float)
    for i, cat in enumerate(y_real):
        Y[i, config.CATEGORIAS.index(str(cat))] = 1.0
    return float(np.mean(np.sum((probs - Y) ** 2, axis=1)))


def evaluar_enfoque(
    nombre: str,
    y_real: list[str] | np.ndarray,
    probs: np.ndarray,
    ms_por_queja: float,
    conjunto: str = "test_en",
) -> dict[str, float | str]:
    y_r = np.asarray(y_real, dtype=str)
    y_p = np.asarray([config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)], dtype=str)

    acc = float(accuracy_score(y_r, y_p))
    f1_m = float(f1_score(y_r, y_p, average="macro", zero_division=0))
    ic_bajo, ic_alto = bootstrap_ic_f1_macro(y_r, y_p)
    acc_bajo, acc_alto = bootstrap_ic(_exactitud, y_r, y_p)
    brier = brier_multiclase(y_r, probs)

    # Top-2 accuracy (¿está la mesa correcta entre las 2 más probables?)
    orden_top2 = np.argsort(-probs, axis=1)[:, :2]
    aciertos_top2 = [
        config.CATEGORIAS.index(str(yr)) in orden_top2[i] for i, yr in enumerate(y_r)
    ]
    top2_acc = float(np.mean(aciertos_top2))

    # Costo operativo esperado por cada 1.000 quejas sin abstención
    tasa_error = 1.0 - acc
    costo_retrabajo_1k = tasa_error * 1000.0 * config.COSTO_MAL_ENRUTAMIENTO_USD

    return {
        "enfoque": nombre,
        "conjunto": conjunto,
        "n_quejas": int(len(y_r)),
        "exactitud": acc,
        "exactitud_ic95_bajo": acc_bajo,
        "exactitud_ic95_alto": acc_alto,
        "f1_macro": f1_m,
        "f1_ic95_bajo": ic_bajo,
        "f1_ic95_alto": ic_alto,
        "exactitud_top2": top2_acc,
        "brier_multiclase": brier,
        "ece": ece(y_r, probs),
        "ms_por_queja": float(ms_por_queja),
        "segundos_por_10k_quejas": float(ms_por_queja * 10.0),
        "costo_retrabajo_1k_usd": float(costo_retrabajo_1k),
    }


def metricas_por_categoria(
    resultados_probs: dict[str, np.ndarray],
    y_real: list[str] | np.ndarray,
) -> pd.DataFrame:
    y_r = np.asarray(y_real, dtype=str)
    filas = []
    for nombre, probs in resultados_probs.items():
        y_p = np.asarray([config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)], dtype=str)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_r, y_p, labels=config.CATEGORIAS, zero_division=0
        )
        for cat, p, r, f, s in zip(config.CATEGORIAS, prec, rec, f1, sup):
            filas.append({
                "enfoque": nombre,
                "categoria": cat,
                "etiqueta": config.ETIQUETAS_CATEGORIA[cat],
                "precision": float(p),
                "exhaustividad": float(r),
                "f1": float(f),
                "soporte": int(s),
            })
    return pd.DataFrame(filas)


def tabla_confusion(y_real: list[str] | np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    y_r = np.asarray(y_real, dtype=str)
    y_p = np.asarray([config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)], dtype=str)
    cm = confusion_matrix(y_r, y_p, labels=config.CATEGORIAS)
    return pd.DataFrame(cm, index=config.CATEGORIAS, columns=config.CATEGORIAS)


def curva_abstencion_triaje(
    y_real: list[str] | np.ndarray,
    probs: np.ndarray,
    nombre_modelo: str = "bge-m3 + LR",
) -> pd.DataFrame:
    """
    Evalúa el enrutamiento selectivo: si la confianza máxima max_k P(y=k|x) < umbral,
    el sistema se abstiene y envía la queja a triaje humano (costo 3,50 USD, sin error de mesa)
    en vez de arriesgar un mal enrutamiento (costo 14,00 USD).
    """
    y_r = np.asarray(y_real, dtype=str)
    y_p = np.asarray([config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)], dtype=str)
    conf = np.max(probs, axis=1)

    umbrales = np.linspace(0.20, 0.90, 29)
    filas = []
    for u in umbrales:
        auto = conf >= u
        cobertura = float(np.mean(auto))
        if np.sum(auto) < 5:
            continue
        acc_auto = float(accuracy_score(y_r[auto], y_p[auto]))
        f1_auto = float(f1_score(y_r[auto], y_p[auto], average="macro", zero_division=0))

        # Por cada 1.000 quejas recibidas:
        n_auto_1k = cobertura * 1000.0
        n_humano_1k = (1.0 - cobertura) * 1000.0
        n_errores_1k = n_auto_1k * (1.0 - acc_auto)
        costo_total_1k = (
            n_errores_1k * config.COSTO_MAL_ENRUTAMIENTO_USD
            + n_humano_1k * config.COSTO_TRIAJE_HUMANO_USD
        )
        filas.append({
            "modelo": nombre_modelo,
            "umbral_confianza": round(float(u), 3),
            "cobertura_automatizada_pct": cobertura * 100.0,
            "enviadas_a_triaje_pct": (1.0 - cobertura) * 100.0,
            "exactitud_en_automatizadas": acc_auto,
            "f1_macro_en_automatizadas": f1_auto,
            "errores_por_1k_quejas": n_errores_1k,
            "costo_operativo_1k_usd": costo_total_1k,
        })
    return pd.DataFrame(filas)


def aplicar_umbral_abstencion(y_real: list[str] | np.ndarray, probs: np.ndarray, umbral: float) -> dict[str, float]:
    """Resultado operativo de un umbral fijo (elegido antes, en validación) sobre otro conjunto."""
    y_r = np.asarray(y_real, dtype=str)
    y_p = np.asarray([config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)], dtype=str)
    auto = probs.max(axis=1) >= umbral
    cobertura = float(np.mean(auto))
    acc_auto = float(np.mean(y_r[auto] == y_p[auto])) if auto.any() else float("nan")
    acc_total = float(np.mean(y_r == y_p))
    n_err = cobertura * 1000.0 * (1.0 - acc_auto) if auto.any() else 0.0
    costo = n_err * config.COSTO_MAL_ENRUTAMIENTO_USD + (1.0 - cobertura) * 1000.0 * config.COSTO_TRIAJE_HUMANO_USD
    costo_sin = (1.0 - acc_total) * 1000.0 * config.COSTO_MAL_ENRUTAMIENTO_USD
    return {
        "umbral": float(umbral),
        "cobertura_automatizada_pct": cobertura * 100.0,
        "exactitud_sin_abstencion": acc_total,
        "exactitud_en_automatizadas": acc_auto,
        "costo_operativo_1k_usd": float(costo),
        "costo_sin_abstencion_1k_usd": float(costo_sin),
        "ahorro_pct": float((1.0 - costo / costo_sin) * 100.0) if costo_sin > 0 else 0.0,
    }


def umbral_optimo_costo(y_real: list[str] | np.ndarray, probs: np.ndarray) -> float:
    curva = curva_abstencion_triaje(y_real, probs)
    return float(curva.loc[curva["costo_operativo_1k_usd"].idxmin(), "umbral_confianza"])
