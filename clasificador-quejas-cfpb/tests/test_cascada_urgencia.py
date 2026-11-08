"""Pruebas de la cascada TF-IDF → LLM y de la detección de urgencia (segunda tarea)."""
from __future__ import annotations

import numpy as np
import pytest

from quejas import config, datos, predecir
from quejas.modelos import ModeloUrgenciaTFIDF, cascada, umbral_por_presupuesto, urgencia_por_reglas


def _fila(cat: str, p: float) -> np.ndarray:
    v = np.full(6, (1.0 - p) / 5)
    v[config.CATEGORIAS.index(cat)] = p
    return v


def test_cascada_solo_envia_al_llm_las_de_baja_confianza():
    barato = np.stack([_fila("hipotecas", 0.90), _fila("cobranza_deudas", 0.35), _fila("tarjetas_credito", 0.55)])
    llm = np.stack([_fila("reporte_credito", 0.99)] * 3)
    combinadas, al_llm = cascada(barato, llm, umbral=0.50)
    assert al_llm.tolist() == [False, True, False]
    assert combinadas[0].argmax() == config.CATEGORIAS.index("hipotecas")
    assert combinadas[1].argmax() == config.CATEGORIAS.index("reporte_credito")
    # Umbral 0: nada va al LLM; umbral > 1: todo va al LLM
    assert not cascada(barato, llm, 0.0)[1].any()
    assert cascada(barato, llm, 1.01)[1].all()


def test_umbral_por_presupuesto_respeta_la_fraccion():
    conf = np.linspace(0.2, 1.0, 1000)
    u = umbral_por_presupuesto(conf, 0.25)
    assert np.mean(conf < u) == pytest.approx(0.25, abs=0.01)


def test_reglas_de_urgencia():
    marcadas = urgencia_por_reglas([
        "They started foreclosure on my home after the loan modification was denied.",
        "Someone opened a credit card in my name; this is identity theft.",
        "The collector threatened to have me arrested if I do not pay today.",
        "My car was repossessed without notice.",
        "I was charged an annual fee on my credit card that I did not expect.",
    ])
    assert marcadas.tolist() == [True, True, True, True, False]


def test_modelo_urgencia_tfidf_aprende_y_elige_umbral():
    urg = ["fraud identity theft account opened in my name police report"] * 20 + \
          ["foreclosure sale of my home scheduled lender refused modification"] * 20
    normal = ["annual fee charged on my credit card statement"] * 40 + \
             ["overdraft fee on checking account after deposit"] * 40
    textos = urg + normal
    y = np.array([1] * len(urg) + [0] * len(normal))
    m = ModeloUrgenciaTFIDF().ajustar(textos, y)
    u = m.elegir_umbral(textos, y)
    assert 0.05 <= u <= 0.95
    assert (m.predecir(textos) == y.astype(bool)).mean() > 0.95


def test_enrutar_con_cascada_queja_clara_no_va_al_llm():
    res = predecir.enrutar_con_cascada(
        "My mortgage servicer misapplied my escrow payment for property taxes and homeowners insurance "
        "on my home loan and my monthly mortgage payment went up."
    )
    assert res["fue_al_llm"] is False
    assert res["categoria_final"] == "hipotecas"
    assert res["decidido_por"] == "TF-IDF + LR"
    assert len(res["pasos"]) >= 3


def test_enrutar_con_cascada_usa_llm_grabado_o_triaje(monkeypatch):
    art = dict(predecir.cargar_artefacto())
    art["umbral_cascada"] = 1.01  # fuerza a que todo vaya al LLM
    monkeypatch.setattr(predecir, "cargar_artefacto", lambda *a, **k: art)

    # Queja traducida al español: su respuesta few-shot del LLM está grabada
    q = datos.cargar_quejas_espanol().iloc[0]
    res = predecir.enrutar_con_cascada(q["texto_queja"])
    assert res["fue_al_llm"] is True
    assert res["decidido_por"].startswith("LLM few-shot")
    assert res["categoria_final"] in config.CATEGORIAS

    # Texto sin respuesta grabada y sin Ollama: va a triaje humano
    res2 = predecir.enrutar_con_cascada("Texto inventado para la prueba que no tiene respuesta grabada del LLM.")
    assert res2["decidido_por"] == "Triaje humano"
    assert res2["destino_operativo"] == "Triaje humano"


def test_clasificar_queja_incluye_urgencia():
    res = predecir.clasificar_queja(
        "Someone stole my identity and opened a loan in my name. The debt collector is now threatening to sue me "
        "and garnish my wages for a debt that is not mine.", enfoque="tfidf")
    assert set(["prob_urgente", "urgente", "umbral_urgencia"]) <= set(res)
    assert res["urgente"] is True
