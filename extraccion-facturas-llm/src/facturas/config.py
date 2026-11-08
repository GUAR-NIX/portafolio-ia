"""Rutas y parámetros (todos se pueden cambiar con variables de entorno)."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_FACTURAS = RAIZ / "data" / "raw" / "facturas"       # 200 PDF generados (se regeneran con una semilla)
DIR_VERDAD = RAIZ / "data" / "raw" / "verdad"            # JSON con la respuesta correcta de cada factura
DIR_MUESTRAS = RAIZ / "data" / "muestras"                # 10 facturas versionadas para la demo y las pruebas
CATALOGO = RAIZ / "data" / "catalogo.csv"                # índice: id, plantilla, condición, ruido
GRABACION = RAIZ / "data" / "grabacion_llm.jsonl"
REPORTES = RAIZ / "reports"
FIGURAS = REPORTES / "figures"

PROVEEDOR = os.environ.get("FACTURAS_PROVEEDOR", "ollama")      # ollama | grabado
MODELO = os.environ.get("FACTURAS_MODELO", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TEMPERATURA = float(os.environ.get("FACTURAS_TEMPERATURA", "0"))
TIMEOUT_LLM = float(os.environ.get("FACTURAS_TIMEOUT", "600"))
MAX_REINTENTOS = int(os.environ.get("FACTURAS_REINTENTOS", "1"))   # cada llamada cuesta ~1 min en CPU

SEMILLA = 2026
N_FACTURAS = 200
TOLERANCIA = 0.02          # USD: diferencia aceptada por redondeo en las validaciones de montos
