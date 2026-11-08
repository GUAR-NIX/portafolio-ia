"""Pruebas de métricas multiclase, Brier, bootstrap, matriz de confusión y curva de abstención."""
from __future__ import annotations

import numpy as np
import pytest

from quejas import config, evaluacion


def test_brier_multiclase_perfecto_y_uniforme():
    y = ["reporte_credito", "hipotecas"]
    probs_perf = np.zeros((2, 6), dtype=float)
    probs_perf[0, 0] = 1.0
    probs_perf[1, 2] = 1.0
    assert evaluacion.brier_multiclase(y, probs_perf) == pytest.approx(0.0)

    probs_unif = np.full((2, 6), 1.0 / 6.0)
    # Para distribución uniforme en K=6 clases: (5/6)^2 + 5*(1/6)^2 = 30/36 = 5/6 = 0.83333
    assert evaluacion.brier_multiclase(y, probs_unif) == pytest.approx(5.0 / 6.0)


def test_evaluar_enfoque_y_abstencion(corpus_sintetico):
    _, y, _ = corpus_sintetico
    n = len(y)
    probs = np.full((n, 6), 0.04, dtype=float)
    for i, cat in enumerate(y):
        probs[i, config.CATEGORIAS.index(str(cat))] = 0.80

    res = evaluacion.evaluar_enfoque("Prueba", y, probs, ms_por_queja=1.2)
    assert res["exactitud"] == pytest.approx(1.0)
    assert res["f1_macro"] == pytest.approx(1.0)
    assert res["costo_retrabajo_1k_usd"] == pytest.approx(0.0)

    cm = evaluacion.tabla_confusion(y, probs)
    assert cm.shape == (6, 6)
    assert int(np.trace(cm.to_numpy())) == n

    curva = evaluacion.curva_abstencion_triaje(y, probs)
    assert len(curva) > 5
    assert (curva["exactitud_en_automatizadas"] == 1.0).all()
