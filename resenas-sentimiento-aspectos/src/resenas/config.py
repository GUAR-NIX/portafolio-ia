"""Rutas, parámetros y catálogo de aspectos del proyecto."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_ETIQUETAS = DIR_DATOS / "etiquetas"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"

# --- Fuente: Amazon Reviews 2023 (McAuley Lab, UCSD) en Hugging Face ---
CATEGORIA = "Cell_Phones_and_Accessories"
URL_HF = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
URL_RESENAS = f"{URL_HF}/review_categories/{CATEGORIA}.jsonl"
URL_META = f"{URL_HF}/meta_categories/meta_{CATEGORIA}.jsonl"

PRODUCTOS_RAW = DIR_RAW / "productos.parquet"      # metadatos de productos con >= MIN_CALIFICACIONES
RESENAS_RAW = DIR_RAW / "resenas.parquet"          # subconjunto muestreado (no se versiona)
CHECKPOINT_DESCARGA = DIR_RAW / "_descarga_resenas.pkl"

MIN_CALIFICACIONES = 300      # productos con al menos estas calificaciones en Amazon
# Tipos de producto (categoría de Amazon → nombre corto). "Accessories" y "Cases" se abren en su subcategoría.
TIPOS_PRODUCTO = {
    "Cases, Holsters & Sleeves > Basic Cases": "Fundas",
    "Accessories > Maintenance, Upkeep & Repairs": "Protectores de pantalla",
    "Accessories > Chargers & Power Adapters": "Cargadores y power banks",
    "Accessories > Cables & Adapters": "Cables y adaptadores",
    "Accessories > Smartwatch Accessories": "Accesorios de smartwatch",
    "Accessories > Automobile Accessories": "Accesorios para auto",
    "Accessories > Single Ear Bluetooth Headsets": "Auriculares Bluetooth",
    "Cell Phones": "Teléfonos",
}
PRODUCTOS_POR_TIPO = 40       # los 40 productos más calificados de cada tipo
RESENAS_POR_PRODUCTO = 200    # muestreo de reservorio por producto (uniforme en el tiempo)
SEMILLA = 2023

# --- Conjuntos etiquetados a mano ---
# El texto de las reseñas NO se versiona (licencia de investigación): data/etiquetas/indice_muestra.jsonl guarda ids y
# huellas, y  python -m resenas.muestra  reconstruye el texto en data/raw/ desde la descarga.
INDICE_MUESTRA = DIR_ETIQUETAS / "indice_muestra.jsonl"
MUESTRA_ETIQUETADO = DIR_RAW / "muestra.jsonl"       # texto de las reseñas de dev + test (reconstruido, no se versiona)
ETIQUETAS_DEV = DIR_ETIQUETAS / "dev.jsonl"
ETIQUETAS_TEST = DIR_ETIQUETAS / "test.jsonl"
ETIQUETAS_TEST_PASADA2 = DIR_ETIQUETAS / "test_pasada2.jsonl"
MUESTRA_ES = DIR_RAW / "muestra_es.jsonl"            # texto de las reseñas en español (reconstruido, no se versiona)
ETIQUETAS_ES = DIR_ETIQUETAS / "es.jsonl"

# --- Aspectos (ver docs/GUIA_ANOTACION.md) ---
ASPECTOS = {
    "bateria": "Duración o capacidad de la batería (del teléfono, reloj, audífonos o power bank).",
    "carga": "Cargar el dispositivo: velocidad, cable, conector, puerto, carga inalámbrica, si carga o no.",
    "pantalla": "Pantalla o protector de pantalla: claridad, brillo, sensibilidad táctil, burbujas, rayones en la pantalla.",
    "sonido": "Audio, volumen, micrófono, calidad de llamadas, cancelación de ruido.",
    "calidad": "Calidad de construcción, materiales, durabilidad, defectos, se rompió o dejó de funcionar.",
    "ajuste": "Compatibilidad o encaje con el dispositivo: tamaño, botones, cortes de cámara, se suelta.",
    "diseno": "Apariencia, color, estilo, grosor, tacto o agarre.",
    "proteccion": "Cuánto protege el teléfono o la pantalla de caídas, golpes o rayones.",
    "precio": "Precio, relación calidad-precio, caro o barato.",
    "envio": "Envío, entrega, empaque, llegó dañado, incompleto o usado.",
    "atencion": "Atención del vendedor o fabricante, garantía, reembolso, devolución, reemplazo.",
    "facilidad_uso": "Instalación, configuración, emparejamiento, instrucciones, facilidad de uso.",
}
POLARIDADES = ("positivo", "negativo", "neutro")

# --- Ollama ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PROVEEDOR = os.environ.get("RESENAS_PROVEEDOR", "ollama")   # ollama | grabado
MODELO_LLM = os.environ.get("RESENAS_MODELO", "qwen2.5:7b")
MODELO_EMB = os.environ.get("RESENAS_MODELO_EMB", "bge-m3")
TEMPERATURA = 0.0
TIMEOUT_LLM = 600
GRABACION = DIR_DATOS / "grabacion_llm.jsonl"
CACHE_EMB = DIR_RAW / "embeddings"   # .npy por lote (no se versiona; se reanuda)
