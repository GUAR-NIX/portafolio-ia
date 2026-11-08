"""Prueba de humo de la app Streamlit (sin navegador ni Ollama)."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app_streamlit.py"


def test_app_muestra_ruta_decision_y_urgencia():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    at.selectbox[0].select("Hipoteca con riesgo de ejecución").run()
    at.button[0].click().run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Destino"] == "Hipotecas y préstamos de vivienda"
    assert "Alta" in metricas["Prioridad"]
    assert any("TF-IDF + LR propone" in m.value for m in at.markdown)
