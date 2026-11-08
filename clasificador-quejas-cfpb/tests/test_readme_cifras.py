"""Las cifras del README, la ficha y la guía deben salir de reports/: ninguna se escribe a mano."""
from __future__ import annotations

import json
import re

import pandas as pd

from quejas import config, informe

R = json.loads((config.REPORTES / "resultados.json").read_text(encoding="utf-8"))
README = (config.RAIZ / "README.md").read_text(encoding="utf-8")


def test_documentos_coinciden_con_lo_que_genera_resultados_json():
    generados = informe.generar()
    for nombre, destino in informe.DESTINOS.items():
        assert destino.read_text(encoding="utf-8") == generados[nombre], (
            f"{nombre} no coincide con reports/resultados.json: ejecuta python -m quejas.informe"
        )


def test_cifras_clave_del_readme_estan_en_reports():
    comp = pd.read_csv(config.REPORTES / "comparacion_enfoques.csv").set_index("enfoque")
    for enfoque in ("1. TF-IDF + LR (25.722 ej.)", "4. LLM zero-shot (qwen2.5:7b)", "5. LLM few-shot (qwen2.5:7b, 6 ej.)"):
        fila = comp.loc[enfoque]
        assert informe.n(fila["f1_macro"]) in README
        assert f"{informe.n(fila['f1_ic95_bajo'])}–{informe.n(fila['f1_ic95_alto'])}" in README
    # Un único conjunto de prueba para todos los enfoques, de 300 a 500 quejas, balanceado
    assert comp["n_quejas"].nunique() == 1
    n = int(comp["n_quejas"].iloc[0])
    assert 300 <= n <= 500 and n % 6 == 0
    assert f"{n} quejas" in README
    # Tamaños de entrenamiento coherentes entre metadata, CSV y README
    tp = R["metadata"]["tamanos_particion"]
    assert tp["train_tfidf"] == 25722 and "25.722" in README
    assert tp["train_embeddings"] == 1800 and "1.800" in README
    assert tp["test_espanol"] == 60


def test_umbrales_elegidos_en_validacion_y_reportados_en_prueba():
    abst = pd.read_csv(config.REPORTES / "abstencion_umbral_validacion.csv")
    assert abst["elegido_en"].str.startswith("val").all()
    assert abst["evaluado_en"].str.startswith("test_comun").all()
    assert informe.n(R["cascada"]["umbral"]) in README
    assert 0.0 < R["cascada"]["pct_al_llm_test"] < 100.0


def test_conclusiones_no_afirman_diferencias_no_distinguibles():
    difs = pd.read_csv(config.REPORTES / "diferencias_pareadas.csv")
    fila = difs[(difs["enfoque_a"].str.startswith("1. TF-IDF")) & (difs["enfoque_b"].str.startswith("4. LLM"))].iloc[0]
    if not fila["distinguible"]:
        assert "supera claramente" not in README


def test_documentos_describen_el_proyecto_como_esta():
    """Sin menciones a versiones previas, con diagrama y con al menos 8 preguntas de entrevista con respuesta."""
    guia = (config.RAIZ / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")
    ficha = (config.RAIZ / "MODEL_CARD.md").read_text(encoding="utf-8")
    for texto in (README, guia, ficha):
        assert "versión anterior" not in texto.lower() and "version_anterior" not in texto
    assert "```mermaid" in README
    preguntas = re.findall(r"^\*\*\d+\. .+\?\*\*$", guia, flags=re.MULTILINE)
    assert len(preguntas) >= 8


def test_plantillas_fuera_de_docs():
    assert (config.RAIZ / "scripts" / "plantillas" / "README.md").exists()
    assert informe.PLANTILLAS == config.RAIZ / "scripts" / "plantillas"
    assert not (config.RAIZ / "docs" / "plantillas").exists()
