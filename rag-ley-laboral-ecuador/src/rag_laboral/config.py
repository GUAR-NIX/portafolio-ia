"""Rutas y parámetros del proyecto (todo se puede cambiar con variables de entorno)."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
RAW = DATA / "raw"
REPORTS = RAIZ / "reports"
FIGURAS = REPORTS / "figures"

ARTICULOS = DATA / "articulos.jsonl"           # corpus troceado por artículo (versionado)
INDICE = DATA / "indice"                       # embeddings float16 de artículos y consultas (versionados: pruebas sin Ollama)
EMB_CACHE = RAW / "cache_embeddings.jsonl"     # caché completa de embeddings por huella (reanudable, no se versiona)
GRABACION = DATA / "grabacion_llm.jsonl"       # respuestas grabadas del LLM (pruebas y demo sin Ollama)
PREGUNTAS = DATA / "evaluacion" / "preguntas.csv"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PROVEEDOR = os.environ.get("RAG_PROVEEDOR", "ollama")      # ollama | grabado
MODELO = os.environ.get("RAG_MODELO", "qwen2.5:7b")
MODELO_EMB = os.environ.get("RAG_MODELO_EMB", "bge-m3")
TEMPERATURA = 0.0
TIMEOUT_LLM = 600

# Recuperación
K_CANDIDATOS = 20   # lo que entra al re-ranking
K_CONTEXTO = 5      # artículos que lee el LLM
K_RRF = 60          # constante de Reciprocal Rank Fusion
