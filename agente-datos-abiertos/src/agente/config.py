"""Configuración central de IA_06_agente_datos_abiertos: rutas, catálogos de APIs, ciudades, indicadores y Ollama."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS_DIR = RAIZ / "data"
DATOS_CRUDOS_DIR = DATOS_DIR / "raw"

CRUDOS_BM_JSON = DATOS_CRUDOS_DIR / "banco_mundial_crudo.json"
CRUDOS_CLIMA_JSON = DATOS_CRUDOS_DIR / "open_meteo_diario_crudo.json"
CRUDOS_PRONOSTICO_JSON = DATOS_CRUDOS_DIR / "open_meteo_pronostico_crudo.json"

INSTANTANEA_BM = DATOS_DIR / "instantanea_indicadores.json"
INSTANTANEA_CLIMA = DATOS_DIR / "instantanea_clima.json"
GRABACION_LLM = DATOS_DIR / "grabacion_llm.jsonl"

MODELOS_DIR = RAIZ / "models"
REPORTES_DIR = RAIZ / "reports"
FIGURAS_DIR = REPORTES_DIR / "figures"

# ---------------------------------------------------------------------------
# Endpoints públicos oficiales (sin credenciales)
# ---------------------------------------------------------------------------
URL_BANCO_MUNDIAL = "https://api.worldbank.org/v2"
URL_OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
URL_OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

ANIO_INICIO_BM = 1990
ANIO_FIN_BM = 2023
FECHA_INICIO_CLIMA = "2020-01-01"
FECHA_FIN_CLIMA = "2024-12-31"

# ---------------------------------------------------------------------------
# Catálogo de 12 países (América Latina + referencia EE. UU.)
# ---------------------------------------------------------------------------
PAISES: dict[str, dict[str, str]] = {
    "ECU": {"nombre": "Ecuador", "capital": "Quito", "region": "Andina"},
    "COL": {"nombre": "Colombia", "capital": "Bogotá", "region": "Andina"},
    "PER": {"nombre": "Perú", "capital": "Lima", "region": "Andina"},
    "CHL": {"nombre": "Chile", "capital": "Santiago", "region": "Cono Sur"},
    "ARG": {"nombre": "Argentina", "capital": "Buenos Aires", "region": "Cono Sur"},
    "BRA": {"nombre": "Brasil", "capital": "São Paulo", "region": "Atlántico Sur"},
    "MEX": {"nombre": "México", "capital": "Ciudad de México", "region": "Norte y Centroamérica"},
    "URY": {"nombre": "Uruguay", "capital": "Montevideo", "region": "Cono Sur"},
    "CRI": {"nombre": "Costa Rica", "capital": "San José", "region": "Norte y Centroamérica"},
    "BOL": {"nombre": "Bolivia", "capital": "La Paz", "region": "Andina"},
    "PRY": {"nombre": "Paraguay", "capital": "Asunción", "region": "Cono Sur"},
    "USA": {"nombre": "Estados Unidos", "capital": "Washington D. C.", "region": "Referencia Global"},
}

ALIAS_PAISES: dict[str, str] = {
    "ecuador": "ECU",
    "ecu": "ECU",
    "colombia": "COL",
    "col": "COL",
    "peru": "PER",
    "perú": "PER",
    "per": "PER",
    "chile": "CHL",
    "chl": "CHL",
    "argentina": "ARG",
    "arg": "ARG",
    "brasil": "BRA",
    "brazil": "BRA",
    "bra": "BRA",
    "mexico": "MEX",
    "méxico": "MEX",
    "mex": "MEX",
    "uruguay": "URY",
    "ury": "URY",
    "costa rica": "CRI",
    "costa_rica": "CRI",
    "cri": "CRI",
    "bolivia": "BOL",
    "bol": "BOL",
    "paraguay": "PRY",
    "pry": "PRY",
    "estados unidos": "USA",
    "eeuu": "USA",
    "ee. uu.": "USA",
    "united states": "USA",
    "usa": "USA",
    "us": "USA",
}

# ---------------------------------------------------------------------------
# Catálogo de 8 indicadores oficiales del Banco Mundial (WDI)
# ---------------------------------------------------------------------------
INDICADORES_BM: dict[str, dict[str, str]] = {
    "pib_per_capita_usd": {
        "codigo_bm": "NY.GDP.PCAP.CD",
        "nombre": "PIB per cápita (USD actuales)",
        "unidad": "USD",
        "descripcion": "Producto Interno Bruto dividido por la población a mitad de año en dólares corrientes.",
    },
    "crecimiento_pib_pct": {
        "codigo_bm": "NY.GDP.MKTP.KD.ZG",
        "nombre": "Crecimiento del PIB real anual (%)",
        "unidad": "%",
        "descripcion": "Tasa de variación porcentual anual del PIB a precios constantes en moneda local.",
    },
    "inflacion_anual_pct": {
        "codigo_bm": "FP.CPI.TOTL.ZG",
        "nombre": "Inflación anual al consumidor (%)",
        "unidad": "%",
        "descripcion": "Variación porcentual anual del índice de precios al consumidor (IPC).",
    },
    "desempleo_pct": {
        "codigo_bm": "SL.UEM.TOTL.ZS",
        "nombre": "Desempleo total (% de la fuerza laboral)",
        "unidad": "%",
        "descripcion": "Porcentaje de la población activa sin empleo que busca trabajo (estimación modelada OIT).",
    },
    "energias_renovables_pct": {
        "codigo_bm": "EG.FEC.RNEW.ZS",
        "nombre": "Consumo de energía renovable (% del total)",
        "unidad": "%",
        "descripcion": "Participación de fuentes renovables en el consumo final total de energía.",
    },
    "esperanza_vida_anios": {
        "codigo_bm": "SP.DYN.LE00.IN",
        "nombre": "Esperanza de vida al nacer (años)",
        "unidad": "años",
        "descripcion": "Número promedio de años que viviría un recién nacido según las tasas de mortalidad vigentes.",
    },
    "usuarios_internet_pct": {
        "codigo_bm": "IT.NET.USER.ZS",
        "nombre": "Usuarios de Internet (% de la población)",
        "unidad": "%",
        "descripcion": "Personas que utilizaron internet desde cualquier dispositivo en los últimos 3 meses.",
    },
    "indice_gini": {
        "codigo_bm": "SI.POV.GINI",
        "nombre": "Índice de Gini (0-100)",
        "unidad": "puntos",
        "descripcion": "Medida de desigualdad en la distribución del ingreso o consumo (0 igualdad perfecta, 100 desigualdad máxima).",
    },
}

ALIAS_INDICADORES: dict[str, str] = {
    "pib_per_capita_usd": "pib_per_capita_usd",
    "ny.gdp.pcap.cd": "pib_per_capita_usd",
    "pib_per_capita": "pib_per_capita_usd",
    "gdp_per_capita": "pib_per_capita_usd",
    "pib": "pib_per_capita_usd",
    "crecimiento_pib_pct": "crecimiento_pib_pct",
    "ny.gdp.mktp.kd.zg": "crecimiento_pib_pct",
    "crecimiento_pib": "crecimiento_pib_pct",
    "gdp_growth": "crecimiento_pib_pct",
    "inflacion_anual_pct": "inflacion_anual_pct",
    "fp.cpi.totl.zg": "inflacion_anual_pct",
    "inflacion": "inflacion_anual_pct",
    "inflación": "inflacion_anual_pct",
    "ipc": "inflacion_anual_pct",
    "desempleo_pct": "desempleo_pct",
    "sl.uem.totl.zs": "desempleo_pct",
    "desempleo": "desempleo_pct",
    "tasa_desempleo": "desempleo_pct",
    "energias_renovables_pct": "energias_renovables_pct",
    "eg.fec.rnew.zs": "energias_renovables_pct",
    "energias_renovables": "energias_renovables_pct",
    "energia_renovable": "energias_renovables_pct",
    "renovables": "energias_renovables_pct",
    "esperanza_vida_anios": "esperanza_vida_anios",
    "sp.dyn.le00.in": "esperanza_vida_anios",
    "esperanza_vida": "esperanza_vida_anios",
    "esperanza_de_vida": "esperanza_vida_anios",
    "usuarios_internet_pct": "usuarios_internet_pct",
    "it.net.user.zs": "usuarios_internet_pct",
    "usuarios_internet": "usuarios_internet_pct",
    "acceso_internet": "usuarios_internet_pct",
    "internet": "usuarios_internet_pct",
    "indice_gini": "indice_gini",
    "si.pov.gini": "indice_gini",
    "gini": "indice_gini",
    "desigualdad": "indice_gini",
}

# ---------------------------------------------------------------------------
# Catálogo de 12 ciudades principales para Open-Meteo Archive / Forecast
# ---------------------------------------------------------------------------
CIUDADES_CLIMA: dict[str, dict[str, object]] = {
    "Quito": {
        "codigo_pais": "ECU",
        "latitud": -0.1807,
        "longitud": -78.4678,
        "altitud_m": 2850,
        "zona_horaria": "America/Guayaquil",
    },
    "Guayaquil": {
        "codigo_pais": "ECU",
        "latitud": -2.1894,
        "longitud": -79.8891,
        "altitud_m": 4,
        "zona_horaria": "America/Guayaquil",
    },
    "Bogotá": {
        "codigo_pais": "COL",
        "latitud": 4.7110,
        "longitud": -74.0721,
        "altitud_m": 2640,
        "zona_horaria": "America/Bogota",
    },
    "Lima": {
        "codigo_pais": "PER",
        "latitud": -12.0464,
        "longitud": -77.0428,
        "altitud_m": 154,
        "zona_horaria": "America/Lima",
    },
    "Santiago": {
        "codigo_pais": "CHL",
        "latitud": -33.4489,
        "longitud": -70.6693,
        "altitud_m": 570,
        "zona_horaria": "America/Santiago",
    },
    "Buenos Aires": {
        "codigo_pais": "ARG",
        "latitud": -34.6037,
        "longitud": -58.3816,
        "altitud_m": 25,
        "zona_horaria": "America/Argentina/Buenos_Aires",
    },
    "São Paulo": {
        "codigo_pais": "BRA",
        "latitud": -23.5505,
        "longitud": -46.6333,
        "altitud_m": 760,
        "zona_horaria": "America/Sao_Paulo",
    },
    "Ciudad de México": {
        "codigo_pais": "MEX",
        "latitud": 19.4326,
        "longitud": -99.1332,
        "altitud_m": 2240,
        "zona_horaria": "America/Mexico_City",
    },
    "Montevideo": {
        "codigo_pais": "URY",
        "latitud": -34.9011,
        "longitud": -56.1645,
        "altitud_m": 43,
        "zona_horaria": "America/Montevideo",
    },
    "San José": {
        "codigo_pais": "CRI",
        "latitud": 9.9281,
        "longitud": -84.0907,
        "altitud_m": 1172,
        "zona_horaria": "America/Costa_Rica",
    },
    "La Paz": {
        "codigo_pais": "BOL",
        "latitud": -16.4897,
        "longitud": -68.1193,
        "altitud_m": 3640,
        "zona_horaria": "America/La_Paz",
    },
    "Asunción": {
        "codigo_pais": "PRY",
        "latitud": -25.2637,
        "longitud": -57.5759,
        "altitud_m": 43,
        "zona_horaria": "America/Asuncion",
    },
}

ALIAS_CIUDADES: dict[str, str] = {
    "quito": "Quito",
    "guayaquil": "Guayaquil",
    "bogota": "Bogotá",
    "bogotá": "Bogotá",
    "lima": "Lima",
    "santiago": "Santiago",
    "santiago de chile": "Santiago",
    "buenos aires": "Buenos Aires",
    "buenos_aires": "Buenos Aires",
    "sao paulo": "São Paulo",
    "são paulo": "São Paulo",
    "sao_paulo": "São Paulo",
    "ciudad de mexico": "Ciudad de México",
    "ciudad de méxico": "Ciudad de México",
    "cdmx": "Ciudad de México",
    "mexico df": "Ciudad de México",
    "montevideo": "Montevideo",
    "san jose": "San José",
    "san josé": "San José",
    "san_jose": "San José",
    "la paz": "La Paz",
    "la_paz": "La Paz",
    "asuncion": "Asunción",
    "asunción": "Asunción",
}

VARIABLES_CLIMA: dict[str, dict[str, str]] = {
    "temp_media_c": {
        "campo_open_meteo": "temperature_2m_mean",
        "nombre": "Temperatura media diaria (°C)",
        "unidad": "°C",
        "agregacion": "promedio",
    },
    "temp_max_c": {
        "campo_open_meteo": "temperature_2m_max",
        "nombre": "Temperatura máxima extrema (°C)",
        "unidad": "°C",
        "agregacion": "maximo",
    },
    "temp_min_c": {
        "campo_open_meteo": "temperature_2m_min",
        "nombre": "Temperatura mínima extrema (°C)",
        "unidad": "°C",
        "agregacion": "minimo",
    },
    "precipitacion_mm": {
        "campo_open_meteo": "precipitation_sum",
        "nombre": "Precipitación acumulada (mm)",
        "unidad": "mm",
        "agregacion": "suma",
    },
    "radiacion_solar_mj_m2": {
        "campo_open_meteo": "shortwave_radiation_sum",
        "nombre": "Radiación solar media diaria (MJ/m²)",
        "unidad": "MJ/m²",
        "agregacion": "promedio",
    },
    "viento_max_kmh": {
        "campo_open_meteo": "wind_speed_10m_max",
        "nombre": "Velocidad máxima del viento (km/h)",
        "unidad": "km/h",
        "agregacion": "maximo",
    },
}

ALIAS_VARIABLES_CLIMA: dict[str, str] = {
    "temp_media_c": "temp_media_c",
    "temperature_2m_mean": "temp_media_c",
    "temperatura_media": "temp_media_c",
    "temperatura": "temp_media_c",
    "temp_media": "temp_media_c",
    "temp_max_c": "temp_max_c",
    "temperature_2m_max": "temp_max_c",
    "temperatura_maxima": "temp_max_c",
    "temp_max": "temp_max_c",
    "temp_min_c": "temp_min_c",
    "temperature_2m_min": "temp_min_c",
    "temperatura_minima": "temp_min_c",
    "temp_min": "temp_min_c",
    "precipitacion_mm": "precipitacion_mm",
    "precipitation_sum": "precipitacion_mm",
    "precipitacion": "precipitacion_mm",
    "precipitación": "precipitacion_mm",
    "lluvia": "precipitacion_mm",
    "radiacion_solar_mj_m2": "radiacion_solar_mj_m2",
    "shortwave_radiation_sum": "radiacion_solar_mj_m2",
    "radiacion_solar": "radiacion_solar_mj_m2",
    "radiación_solar": "radiacion_solar_mj_m2",
    "radiacion": "radiacion_solar_mj_m2",
    "viento_max_kmh": "viento_max_kmh",
    "wind_speed_10m_max": "viento_max_kmh",
    "viento_max": "viento_max_kmh",
    "viento": "viento_max_kmh",
}

# ---------------------------------------------------------------------------
# Vocabulario de capacidades de las herramientas (verificación de anclaje).
# Describe QUÉ mide cada indicador o variable; no contiene palabras de las preguntas
# trampa. Se congeló antes de redactar el conjunto de prueba (data/conjunto_prueba.json).
# ---------------------------------------------------------------------------
TERMINOS_INDICADORES: dict[str, list[str]] = {
    "pib_per_capita_usd": ["pib per capita", "producto interno bruto per", "ingreso per capita",
                           "ingreso por habitante", "pib por habitante", "gdp per capita", "ny.gdp.pcap"],
    "crecimiento_pib_pct": ["crecimiento del pib", "crecimiento economico", "crecimiento de la economia",
                            "pib real", "crecio la economia", "ny.gdp.mktp"],
    "inflacion_anual_pct": ["inflacion", "precios al consumidor", "ipc", "indice de precios", "fp.cpi"],
    "desempleo_pct": ["desempleo", "paro", "desocupacion", "sl.uem"],
    "energias_renovables_pct": ["renovable", "energia limpia", "eg.fec.rnew"],
    "esperanza_vida_anios": ["esperanza de vida", "longevidad", "sp.dyn.le00"],
    "usuarios_internet_pct": ["internet", "conectividad", "it.net.user"],
    "indice_gini": ["gini", "desigualdad", "si.pov.gini"],
}
TERMINOS_VARIABLES_CLIMA: dict[str, list[str]] = {
    "temp_media_c": ["temperatura media", "temperatura promedio", "temperatura"],
    "temp_max_c": ["maxima", "calor", "temperatura"],
    "temp_min_c": ["minima", "frio", "temperatura"],
    "precipitacion_mm": ["precipitacion", "lluvia", "llovio", "mm"],
    "radiacion_solar_mj_m2": ["radiacion", "solar", "irradiacion"],
    "viento_max_kmh": ["viento", "rafaga"],
}
TERMINOS_RESUMEN_CLIMA: list[str] = ["clima", "climatico", "climatica", "meteorolog"] + sorted(
    {t for ts in TERMINOS_VARIABLES_CLIMA.values() for t in ts}
)

# ---------------------------------------------------------------------------
# Modo de datos: "instantanea" (reproducible, evaluación y pytest) o "vivo" (APIs HTTP)
# ---------------------------------------------------------------------------
MODO_DATOS = os.environ.get("AGENTE_MODO_DATOS", "instantanea")
CACHE_API_DIR = DATOS_DIR / "cache_api"
PAUSA_ENTRE_PETICIONES_S = 1.0
REINTENTOS_HTTP = 4

# Cobertura por modo. En "instantanea" es la del archivo versionado (decisión de diseño
# para evaluar de forma reproducible). En "vivo" es la de las APIs: Open-Meteo Archive
# llega hasta 1940 (ERA5) y el Banco Mundial publica WDI desde 1960.
COBERTURA = {
    "instantanea": {"bm": (1990, 2024), "clima": ("2020-01-01", "2024-12-31")},
    "vivo": {"bm": (1960, 2024), "clima": ("1940-01-01", None)},  # None = hoy menos 7 días
}

# ---------------------------------------------------------------------------
# Configuración de Ollama y ejecución determinista
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = os.environ.get("AGENTE_MODELO_LLM", "qwen2.5:7b")
PROVEEDOR = os.environ.get("AGENTE_PROVEEDOR", "grabado")  # "ollama" o "grabado"
TEMPERATURA = 0.0
MAX_TOKENS_SALIDA = 220
TOLERANCIA_ERROR_RELATIVO = 0.01  # 1,0 % para fidelidad numérica (Numerical Groundedness)
SEMILLA = 42

# Agente con herramientas: número máximo de turnos del LLM y versión de prompt elegida
# en el conjunto de VALIDACIÓN (ver reports/mejora_prompt_validacion.csv).
MAX_PASOS_AGENTE = 3
VERSION_PROMPT_AGENTE = os.environ.get("AGENTE_VERSION_PROMPT", "v2")
CONJUNTO_VALIDACION = DATOS_DIR / "conjunto_validacion.json"
CONJUNTO_PRUEBA = DATOS_DIR / "conjunto_prueba.json"

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
