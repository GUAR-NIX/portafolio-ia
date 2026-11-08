"""Fixtures compartidos para las pruebas automáticas de IA_05_clasificador_quejas."""
from __future__ import annotations

import numpy as np
import pytest

from quejas import config


@pytest.fixture
def corpus_sintetico() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Textos sintéticos y vectores ortogonales por categoría para probar los modelos sin Ollama."""
    plantillas = {
        "reporte_credito": "Equifax and Experian credit report has inaccurate late payment and unauthorized inquiry lowering credit score",
        "cobranza_deudas": "Debt collection agency keeps calling my phone daily threatening lawsuit over medical debt already paid",
        "hipotecas": "Home mortgage loan servicer miscalculated escrow account property taxes and denied refinancing modification",
        "tarjetas_credito": "Credit card statement has unauthorized charge annual fee and high APR interest rate billing dispute",
        "cuentas_bancarias": "Checking bank account charged multiple overdraft fees after ATM deposit and froze wire transfer funds",
        "prestamos_consumo": "Student loan servicer and auto vehicle loan lender misapplied monthly installment payment and title",
    }
    rng = np.random.default_rng(42)
    textos = []
    etiquetas = []
    vectores = []

    for i, cat in enumerate(config.CATEGORIAS):
        base_vec = np.zeros(config.DIM_EMBEDDING, dtype=np.float32)
        base_vec[i * 20 : (i + 1) * 20] = 1.0
        for j in range(25):
            textos.append(f"{plantillas[cat]} case number {j}")
            etiquetas.append(cat)
            ruido = rng.normal(0.0, 0.08, size=config.DIM_EMBEDDING).astype(np.float32)
            v = base_vec + ruido
            v /= np.linalg.norm(v)
            vectores.append(v)

    return textos, np.array(etiquetas), np.stack(vectores)
