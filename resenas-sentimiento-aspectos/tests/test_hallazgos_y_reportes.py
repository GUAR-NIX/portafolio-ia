import json

import numpy as np
import pandas as pd
import pytest

from resenas import config
from resenas.etiquetas import como_dict, leer_jsonl
from resenas.hallazgos import impacto_en_estrellas, indicadores
from resenas.metricas import evaluar


def test_indicadores_por_resena():
    r = pd.DataFrame({"id": ["a", "b", "c"]})
    m = pd.DataFrame({"id": ["a", "a", "b"], "aspecto": ["calidad", "precio", "calidad"],
                      "polaridad": ["negativo", "positivo", "positivo"], "cita": ["x", "y", "z"]})
    ind = indicadores(r, m)
    assert ind.loc["a", "neg_calidad"] == 1 and ind.loc["a", "pos_precio"] == 1
    assert ind.loc["b", "neg_calidad"] == 0 and ind.loc["b", "pos_calidad"] == 1
    assert ind.loc["c"].sum() == 0


def test_regresion_recupera_coeficientes_conocidos():
    rng = np.random.default_rng(0)
    n = 4000
    neg = pd.DataFrame({a: rng.integers(0, 2, n) for a in config.ASPECTOS})
    pos = pd.DataFrame({a: rng.integers(0, 2, n) for a in config.ASPECTOS})
    estrellas = 4.2 - 1.5 * neg["calidad"] + 0.3 * pos["precio"] + rng.normal(0, 0.3, n)
    t = impacto_en_estrellas(pd.Series(estrellas), neg, pos)
    assert t.loc["queja_calidad", "coef"] == pytest.approx(-1.5, abs=0.05)
    assert t.loc["elogio_precio", "coef"] == pytest.approx(0.3, abs=0.05)
    assert t.loc["queja_calidad", "ic_bajo"] < -1.5 < t.loc["queja_calidad", "ic_alto"]
    # con pesos uniformes da lo mismo
    tw = impacto_en_estrellas(pd.Series(estrellas), neg, pos, np.full(n, 2.0))
    assert tw.loc["queja_calidad", "coef"] == pytest.approx(t.loc["queja_calidad", "coef"])


# --------------------------------------------------------------------------- reportes versionados
RUTA_EV = config.DIR_REPORTES / "evaluacion.json"


@pytest.mark.skipif(not RUTA_EV.exists(), reason="falta correr python -m resenas.evaluar")
def test_evaluacion_completa_y_coherente():
    ev = json.loads(RUTA_EV.read_text(encoding="utf-8"))
    assert ev["n_test"] == 320
    assert ev["espanol"]["n"] == 80
    for nombre, r in ev["enfoques"].items():
        for m in ("f1_aspectos", "f1_aspecto_polaridad", "exactitud_polaridad", "exactitud_global"):
            assert 0 <= r[m] <= 1
            lo, hi = r["ic95"][m]
            assert lo <= r[m] <= hi
    c = ev["enfoques"]
    assert c["clasico"]["costo"]["segundos_por_resena"] < c["embeddings"]["costo"]["segundos_por_resena"] \
        < c["llm"]["costo"]["segundos_por_resena"]


@pytest.mark.skipif(not RUTA_EV.exists(), reason="falta correr python -m resenas.evaluar")
def test_metricas_del_llm_se_recalculan_desde_lo_grabado():
    """Las cifras del README para el LLM salen de reports/predicciones_llm.jsonl y de las etiquetas a mano."""
    ev = json.loads(RUTA_EV.read_text(encoding="utf-8"))
    verdad = {d["id"]: como_dict(d) for d in leer_jsonl(config.ETIQUETAS_TEST)}
    pred = {d["id"]: {"global": d["sentimiento_global"],
                      "aspectos": {m["aspecto"]: m["polaridad"] for m in d["aspectos"]}}
            for d in leer_jsonl(config.DIR_REPORTES / "predicciones_llm.jsonl") if d["id"] in verdad}
    r = evaluar(verdad, pred, bootstrap=0)
    assert r["f1_aspectos"] == pytest.approx(ev["enfoques"]["llm"]["f1_aspectos"])
    assert r["exactitud_polaridad"] == pytest.approx(ev["enfoques"]["llm"]["exactitud_polaridad"])
