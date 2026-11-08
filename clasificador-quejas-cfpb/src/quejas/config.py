"""Configuración central de IA_05_clasificador_quejas: rutas, taxonomía CFPB, modelos Ollama y costos."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw" / "quejas_cfpb.parquet"
RESUMEN_FUENTE = RAIZ / "data" / "raw" / "resumen_fuente_cfpb.json"
CACHE_EMBEDDINGS = RAIZ / "data" / "embeddings_bge_m3.npz"
GRABACION_LLM = Path(os.environ.get("QUEJAS_GRABACION_LLM", RAIZ / "data" / "grabacion_llm.jsonl"))
EJEMPLOS_FEW_SHOT = RAIZ / "data" / "ejemplos_few_shot.json"
# Traducciones al español de quejas REALES del CFPB (ver columna `origen` y el README)
QUEJAS_ES = RAIZ / "data" / "quejas_es_traducidas.jsonl"
MODELOS = RAIZ / "models"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"

# ---------------------------------------------------------------------------
# Taxonomía de las 6 mesas de producto consolidadas del CFPB
# ---------------------------------------------------------------------------
CATEGORIAS = [
    "reporte_credito",
    "cobranza_deudas",
    "hipotecas",
    "tarjetas_credito",
    "cuentas_bancarias",
    "prestamos_consumo",
]

ETIQUETAS_CATEGORIA = {
    "reporte_credito": "Reporte y burós de crédito",
    "cobranza_deudas": "Cobranza de deudas",
    "hipotecas": "Hipotecas y préstamos de vivienda",
    "tarjetas_credito": "Tarjetas de crédito y prepago",
    "cuentas_bancarias": "Cuentas bancarias y transferencias",
    "prestamos_consumo": "Préstamos estudiantiles, auto y consumo",
}

# Definiciones semánticas usadas como prototipos para Zero-Shot con embeddings bge-m3
PROTOTIPOS_TEXTO = {
    "reporte_credito": (
        "Credit reporting bureau complaint about Equifax, Experian, or TransUnion: "
        "inaccurate information on credit report, unauthorized hard inquiries, identity theft "
        "accounts not removed after dispute, credit score drop, or credit file freeze."
    ),
    "cobranza_deudas": (
        "Debt collection agency complaint: harassing phone calls, attempting to collect a debt "
        "not owed or already paid, medical debt collection, failure to validate debt, or threats "
        "of legal action by third-party debt collectors."
    ),
    "hipotecas": (
        "Home mortgage loan complaint: escrow account calculation error, loan modification or "
        "refinancing denial, foreclosure proceedings, mortgage servicer misapplying monthly payments, "
        "or closing costs on home loan."
    ),
    "tarjetas_credito": (
        "Credit card or prepaid card complaint: unauthorized card charges, billing statement dispute, "
        "unexpected late payment fee or annual fee, high APR interest rate calculation, rewards points, "
        "or closing a credit card account."
    ),
    "cuentas_bancarias": (
        "Checking or savings bank account and wire transfer complaint: overdraft fees, account frozen "
        "or closed without notice, ATM deposit or withdrawal error, check hold, direct deposit, or "
        "Zelle and wire money transfer fraud."
    ),
    "prestamos_consumo": (
        "Student loan, vehicle auto loan, lease, or personal installment loan complaint: "
        "auto repossession, student loan repayment plan or forgiveness servicer error, "
        "payday loan high interest, or vehicle loan payoff title."
    ),
}

# ---------------------------------------------------------------------------
# Configuración de Ollama y reproducción determinista
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = os.environ.get("QUEJAS_MODELO_LLM", "qwen2.5:7b")
MODELO_EMB = os.environ.get("QUEJAS_MODELO_EMB", "bge-m3")
PROVEEDOR = os.environ.get("QUEJAS_PROVEEDOR", "grabado")  # "ollama" o "grabado"
TEMPERATURA = 0.0
DIM_EMBEDDING = 1024

# ---------------------------------------------------------------------------
# Parámetros operativos de enrutamiento y triaje humano
#   UMBRAL_ABSTENCION : si max_k P(y = k | x) < 0,55, la queja es ambigua (cruza áreas)
#                       y se envía a triaje humano en lugar de mal-enrutarla.
#   COSTO_MAL_ENRUTAMIENTO_USD : retrabajo entre áreas especialistas (~14 USD por ticket).
#   COSTO_TRIAJE_HUMANO_USD    : lectura rápida por mesa de entrada (~3,50 USD por ticket).
# ---------------------------------------------------------------------------
UMBRAL_ABSTENCION = 0.55
COSTO_MAL_ENRUTAMIENTO_USD = 14.0
COSTO_TRIAJE_HUMANO_USD = 3.50
SEMILLA = 42

COLORES = [
    "#2a78d6",  # Azul
    "#eb6834",  # Naranja
    "#1baf7a",  # Verde
    "#eda100",  # Amarillo
    "#e87ba4",  # Rosa
    "#6250d6",  # Violeta
    "#008300",  # Verde oscuro
    "#e34948",  # Rojo
]
