"""Configuración central: rutas, países, indicadores del Banco Mundial y parámetros de Ollama."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
RAW = DATA / "raw"
REPORTS = RAIZ / "reports"
FIGURAS = REPORTS / "figures"

INSTANTANEA = DATA / "instantanea_wdi.csv"
METADATOS = DATA / "metadatos_wdi.json"
GRABACION_LLM = DATA / "grabacion_llm.jsonl"
CONJUNTOS = DATA / "conjuntos_evaluacion.json"
REFERENCIA_HALLAZGOS = DATA / "hallazgos_referencia.json"
RUBRICA = DATA / "rubrica_legibilidad.csv"

API_BM = "https://api.worldbank.org/v2"
ANIO_INICIO = 2000
ANIO_FIN = 2025

# 18 países de América Latina (continental + República Dominicana). Cuba se excluye por falta de datos.
PAISES: dict[str, str] = {
    "ARG": "Argentina", "BOL": "Bolivia", "BRA": "Brasil", "CHL": "Chile", "COL": "Colombia",
    "CRI": "Costa Rica", "DOM": "República Dominicana", "ECU": "Ecuador", "GTM": "Guatemala",
    "HND": "Honduras", "MEX": "México", "NIC": "Nicaragua", "PAN": "Panamá", "PER": "Perú",
    "PRY": "Paraguay", "SLV": "El Salvador", "URY": "Uruguay", "VEN": "Venezuela",
}
REGION = "LAT"  # agregado propio: mediana de los 18 países con dato
NOMBRE_REGION = "América Latina (mediana)"


@dataclass(frozen=True)
class Indicador:
    codigo: str
    clave: str            # nombre corto usado en prompts, referencias y tablas
    nombre: str           # nombre en español para el informe
    unidad: str           # "%", "USD", "personas"
    escala_texto: str     # cómo se suele escribir: "%", "USD", "mil_millones_usd", "millones"
    mayor_es_mejor: bool | None
    palabras: tuple[str, ...]  # palabras que delatan el indicador en un texto libre


INDICADORES: tuple[Indicador, ...] = (
    Indicador("NY.GDP.MKTP.CD", "pib_usd", "PIB (USD corrientes)", "USD", "mil_millones_usd", True,
              ("pib nominal", "tamaño de la economía", "producto interno bruto", "pib total", "pib de")),
    Indicador("NY.GDP.PCAP.CD", "pib_pc", "PIB per cápita (USD corrientes)", "USD", "USD", True,
              ("per cápita", "per capita", "por habitante", "por persona")),
    Indicador("NY.GDP.MKTP.KD.ZG", "crecimiento", "Crecimiento del PIB real (% anual)", "%", "%", True,
              ("crecimiento", "creció", "crece", "contracción", "se contrajo", "expansión", "recesión", "pib real")),
    Indicador("FP.CPI.TOTL.ZG", "inflacion", "Inflación, precios al consumidor (% anual)", "%", "%", False,
              ("inflación", "precios", "ipc", "deflación")),
    Indicador("SL.UEM.TOTL.ZS", "desempleo", "Desempleo (% de la fuerza laboral, estimación OIT)", "%", "%", False,
              ("desempleo", "desocupación", "paro", "mercado laboral")),
    Indicador("NE.EXP.GNFS.ZS", "exportaciones", "Exportaciones de bienes y servicios (% del PIB)", "%", "%", None,
              ("exportaciones", "exporta", "ventas externas")),
    Indicador("NE.IMP.GNFS.ZS", "importaciones", "Importaciones de bienes y servicios (% del PIB)", "%", "%", None,
              ("importaciones", "importa", "compras externas")),
    Indicador("BN.CAB.XOKA.GD.ZS", "cuenta_corriente", "Saldo de cuenta corriente (% del PIB)", "%", "%", True,
              ("cuenta corriente", "balanza de pagos", "déficit externo", "superávit externo")),
    Indicador("BX.KLT.DINV.WD.GD.ZS", "ied", "Inversión extranjera directa, entradas netas (% del PIB)", "%", "%", True,
              ("inversión extranjera", "ied", "inversión directa")),
    Indicador("BX.TRF.PWKR.DT.GD.ZS", "remesas", "Remesas recibidas (% del PIB)", "%", "%", None,
              ("remesas",)),
    Indicador("DT.DOD.DECT.GN.ZS", "deuda_externa", "Deuda externa total (% del INB)", "%", "%", False,
              ("deuda externa", "deuda", "endeudamiento")),
    Indicador("SI.POV.DDAY", "pobreza", "Pobreza extrema, USD 3,00 al día PPA 2021 (% de la población)", "%", "%", False,
              ("pobreza", "pobres")),
    Indicador("SI.POV.GINI", "gini", "Índice de Gini (0–100)", "índice", "índice", False,
              ("gini", "desigualdad")),
    Indicador("SP.POP.TOTL", "poblacion", "Población total", "personas", "millones", None,
              ("población", "habitantes")),
)
IND_POR_CLAVE = {i.clave: i for i in INDICADORES}
IND_POR_CODIGO = {i.codigo: i for i in INDICADORES}

# Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO = os.environ.get("KPIS_MODELO", "qwen2.5:7b")
# "grabado": solo lee data/grabacion_llm.jsonl (pytest, app, API). "ollama": llama y graba lo que falte.
PROVEEDOR = os.environ.get("KPIS_PROVEEDOR", "grabado")
OPCIONES_LLM = {"temperature": 0.0, "seed": 7, "num_ctx": 4096, "num_predict": 2048, "num_thread": 6}

# Conjuntos de evaluación: validación y prueba no comparten país ni año. Venezuela queda fuera
# (le faltan inflación, pobreza y Gini recientes). Los prompts se ajustan solo con validación.
VALIDACION = [("ARG", 2023), ("CHL", 2023), ("PER", 2023), ("GTM", 2023), ("PRY", 2023)]
PRUEBA = [("ECU", 2024), ("BOL", 2024), ("BRA", 2024), ("COL", 2024), ("CRI", 2024), ("DOM", 2024),
          ("HND", 2024), ("MEX", 2024), ("NIC", 2024), ("PAN", 2024), ("SLV", 2024), ("URY", 2024)]
MAX_REINTENTOS = 2
