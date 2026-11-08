"""Descarga, normalización y carga de datos reales de Banco Mundial (WDI) y Open-Meteo."""
from __future__ import annotations

import json
import time
import unicodedata
from datetime import datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd

from . import config

_CACHE_BM: dict[str, Any] | None = None
_CACHE_CLIMA: dict[str, Any] | None = None


def quitar_acentos(texto: str) -> str:
    """Elimina tildes y diacríticos manteniendo caracteres ASCII base."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_pais(codigo_o_nombre: str) -> str:
    """Convierte código ISO3 o nombre de país al código canónico ISO3 soportado."""
    if not codigo_o_nombre or not isinstance(codigo_o_nombre, str):
        raise ValueError(
            f"País vacío o inválido. Códigos ISO3 permitidos: {sorted(config.PAISES.keys())}"
        )
    limpio = codigo_o_nombre.strip()
    mayus = limpio.upper()
    if mayus in config.PAISES:
        return mayus
    clave = limpio.lower()
    if clave in config.ALIAS_PAISES:
        return config.ALIAS_PAISES[clave]
    clave_sin = quitar_acentos(clave)
    for k, v in config.ALIAS_PAISES.items():
        if quitar_acentos(k) == clave_sin:
            return v
    raise ValueError(
        f"País '{codigo_o_nombre}' no reconocido en el catálogo. "
        f"Usa uno de los códigos ISO3 válidos: {sorted(config.PAISES.keys())}."
    )


def normalizar_indicador(indicador: str) -> str:
    """Convierte clave o código del Banco Mundial al nombre canónico del indicador."""
    if not indicador or not isinstance(indicador, str):
        raise ValueError(
            f"Indicador vacío. Indicadores permitidos: {sorted(config.INDICADORES_BM.keys())}"
        )
    limpio = indicador.strip()
    if limpio in config.INDICADORES_BM:
        return limpio
    clave = limpio.lower()
    if clave in config.ALIAS_INDICADORES:
        return config.ALIAS_INDICADORES[clave]
    clave_sin = quitar_acentos(clave).replace(" ", "_").replace("-", "_")
    if clave_sin in config.ALIAS_INDICADORES:
        return config.ALIAS_INDICADORES[clave_sin]
    for cod_can, meta in config.INDICADORES_BM.items():
        if meta["codigo_bm"].lower() == clave:
            return cod_can
    raise ValueError(
        f"Indicador '{indicador}' no soportado. "
        f"Indicadores válidos: {sorted(config.INDICADORES_BM.keys())} "
        f"(o sus códigos WDI: {[m['codigo_bm'] for m in config.INDICADORES_BM.values()]})."
    )


def normalizar_ciudad(ciudad: str) -> str:
    """Convierte el nombre de una ciudad a su forma canónica en el catálogo de Open-Meteo."""
    if not ciudad or not isinstance(ciudad, str):
        raise ValueError(
            f"Ciudad vacía. Ciudades disponibles: {sorted(config.CIUDADES_CLIMA.keys())}"
        )
    limpio = ciudad.strip()
    if limpio in config.CIUDADES_CLIMA:
        return limpio
    clave = limpio.lower()
    if clave in config.ALIAS_CIUDADES:
        return config.ALIAS_CIUDADES[clave]
    clave_sin = quitar_acentos(clave).replace("_", " ")
    for k, v in config.ALIAS_CIUDADES.items():
        if quitar_acentos(k).replace("_", " ") == clave_sin:
            return v
    raise ValueError(
        f"Ciudad '{ciudad}' no disponible en el catálogo climático. "
        f"Ciudades soportadas: {sorted(config.CIUDADES_CLIMA.keys())}."
    )


def normalizar_variable_clima(variable: str) -> str:
    """Convierte nombre o alias de variable meteorológica a su clave canónica."""
    if not variable or not isinstance(variable, str):
        raise ValueError(
            f"Variable climática vacía. Variables válidas: {sorted(config.VARIABLES_CLIMA.keys())}"
        )
    limpio = variable.strip()
    if limpio in config.VARIABLES_CLIMA:
        return limpio
    clave = limpio.lower()
    if clave in config.ALIAS_VARIABLES_CLIMA:
        return config.ALIAS_VARIABLES_CLIMA[clave]
    clave_sin = quitar_acentos(clave).replace(" ", "_").replace("-", "_")
    if clave_sin in config.ALIAS_VARIABLES_CLIMA:
        return config.ALIAS_VARIABLES_CLIMA[clave_sin]
    raise ValueError(
        f"Variable climática '{variable}' no reconocida. "
        f"Variables válidas: {sorted(config.VARIABLES_CLIMA.keys())}."
    )


def descargar_banco_mundial(forzar: bool = False) -> dict[str, Any]:
    """Consulta la API oficial del Banco Mundial (v2) para los 12 países y 8 indicadores."""
    if config.INSTANTANEA_BM.exists() and not forzar:
        return cargar_instantanea_bm()

    config.DATOS_CRUDOS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATOS_DIR.mkdir(parents=True, exist_ok=True)

    codigos_paises = ";".join(config.PAISES.keys())
    crudo_por_indicador: dict[str, list[dict[str, Any]]] = {}
    series: dict[str, dict[str, dict[str, float]]] = {p: {} for p in config.PAISES}

    with httpx.Client(timeout=45.0, follow_redirects=True) as cliente:
        for clave_ind, meta in config.INDICADORES_BM.items():
            codigo_bm = meta["codigo_bm"]
            url = (
                f"{config.URL_BANCO_MUNDIAL}/country/{codigos_paises}"
                f"/indicator/{codigo_bm}"
            )
            params = {
                "date": f"{config.ANIO_INICIO_BM}:2024",
                "format": "json",
                "per_page": 2000,
            }
            ultimo_error: Exception | None = None
            registros: list[dict[str, Any]] = []
            for intento in range(3):
                try:
                    resp = cliente.get(url, params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                    if isinstance(payload, list) and len(payload) >= 2 and payload[1]:
                        registros = payload[1]
                    break
                except Exception as exc:
                    ultimo_error = exc
                    time.sleep(1.0 * (intento + 1))
            if not registros and ultimo_error is not None:
                raise RuntimeError(
                    f"Error al descargar {codigo_bm} desde Banco Mundial: {ultimo_error}"
                )

            crudo_por_indicador[codigo_bm] = registros
            for p in config.PAISES:
                series[p][clave_ind] = {}

            for fila in registros:
                iso3 = fila.get("countryiso3code")
                anio = str(fila.get("date", ""))
                valor = fila.get("value")
                if iso3 in series and valor is not None and anio.isdigit():
                    series[iso3][clave_ind][anio] = round(float(valor), 4)

            for p in config.PAISES:
                series[p][clave_ind] = dict(
                    sorted(series[p][clave_ind].items(), key=lambda kv: int(kv[0]))
                )

    config.CRUDOS_BM_JSON.write_text(
        json.dumps(crudo_por_indicador, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resumen_cobertura: list[dict[str, Any]] = []
    for p, meta_p in config.PAISES.items():
        for ind, meta_i in config.INDICADORES_BM.items():
            valores_anio = series[p].get(ind, {})
            anios_int = [int(a) for a in valores_anio.keys()]
            ultimo_anio = max(anios_int) if anios_int else None
            resumen_cobertura.append({
                "codigo_pais": p,
                "pais": meta_p["nombre"],
                "indicador": ind,
                "codigo_bm": meta_i["codigo_bm"],
                "n_observaciones": len(anios_int),
                "anio_min": min(anios_int) if anios_int else None,
                "anio_max": ultimo_anio,
                "ultimo_valor": valores_anio.get(str(ultimo_anio)) if ultimo_anio else None,
            })

    instantanea = {
        "fuente": "World Bank Open Data API v2 (https://api.worldbank.org/v2/)",
        "fecha_descarga_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rango_anios_consultado": [config.ANIO_INICIO_BM, 2024],
        "paises": config.PAISES,
        "indicadores": config.INDICADORES_BM,
        "series": series,
        "resumen_cobertura": resumen_cobertura,
    }
    config.INSTANTANEA_BM.write_text(
        json.dumps(instantanea, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    global _CACHE_BM
    _CACHE_BM = instantanea
    return instantanea


def descargar_open_meteo(forzar: bool = False) -> dict[str, Any]:
    """Descarga series climáticas diarias (2020-2024) y pronóstico actual de Open-Meteo para 12 ciudades."""
    if config.INSTANTANEA_CLIMA.exists() and not forzar:
        return cargar_instantanea_clima()

    config.DATOS_CRUDOS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATOS_DIR.mkdir(parents=True, exist_ok=True)

    campos_diarios = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "wind_speed_10m_max",
    ]

    crudo_diario: dict[str, Any] = {}
    crudo_pronostico: dict[str, Any] = {}
    anual_por_ciudad: dict[str, dict[str, dict[str, float]]] = {}
    mensual_por_ciudad: dict[str, dict[str, dict[str, float]]] = {}
    pronostico_resumen: dict[str, dict[str, Any]] = {}
    info_ciudades: dict[str, dict[str, Any]] = {}

    nombres_ciudades = list(config.CIUDADES_CLIMA.keys())
    latitudes_str = ",".join(str(config.CIUDADES_CLIMA[c]["latitud"]) for c in nombres_ciudades)
    longitudes_str = ",".join(str(config.CIUDADES_CLIMA[c]["longitud"]) for c in nombres_ciudades)

    with httpx.Client(timeout=90.0, follow_redirects=True) as cliente:
        # Consulta por lotes (1 sola petición multi-coordenada a Open-Meteo Archive)
        params_hist_lote = {
            "latitude": latitudes_str,
            "longitude": longitudes_str,
            "start_date": config.FECHA_INICIO_CLIMA,
            "end_date": config.FECHA_FIN_CLIMA,
            "daily": ",".join(campos_diarios),
            "timezone": "UTC",
        }
        lista_hist: list[dict[str, Any]] = []
        for intento in range(5):
            try:
                r = cliente.get(config.URL_OPEN_METEO_ARCHIVE, params=params_hist_lote)
                if r.status_code == 429:
                    time.sleep(5.0 * (intento + 1))
                    continue
                r.raise_for_status()
                payload_lote = r.json()
                lista_hist = payload_lote if isinstance(payload_lote, list) else [payload_lote]
                break
            except Exception:
                if intento == 4:
                    raise
                time.sleep(4.0 * (intento + 1))

        # Pronóstico actual por lotes (1 sola petición multi-coordenada a Open-Meteo Forecast)
        params_fc_lote = {
            "latitude": latitudes_str,
            "longitude": longitudes_str,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "UTC",
            "forecast_days": 7,
        }
        lista_fc: list[dict[str, Any]] = []
        try:
            r_fc = cliente.get(config.URL_OPEN_METEO_FORECAST, params=params_fc_lote)
            if r_fc.status_code == 200:
                payload_fc_lote = r_fc.json()
                lista_fc = payload_fc_lote if isinstance(payload_fc_lote, list) else [payload_fc_lote]
        except Exception:
            lista_fc = []

        for idx, ciudad in enumerate(nombres_ciudades):
            meta = config.CIUDADES_CLIMA[ciudad]
            payload_hist = lista_hist[idx]
            crudo_diario[ciudad] = payload_hist
            diario = payload_hist.get("daily", {})
            df = pd.DataFrame({
                "fecha": pd.to_datetime(diario["time"]),
                "temp_media_c": pd.to_numeric(diario["temperature_2m_mean"], errors="coerce"),
                "temp_max_c": pd.to_numeric(diario["temperature_2m_max"], errors="coerce"),
                "temp_min_c": pd.to_numeric(diario["temperature_2m_min"], errors="coerce"),
                "precipitacion_mm": np.nan_to_num(
                    pd.to_numeric(diario["precipitation_sum"], errors="coerce"), nan=0.0
                ),
                "radiacion_solar_mj_m2": pd.to_numeric(diario["shortwave_radiation_sum"], errors="coerce"),
                "viento_max_kmh": pd.to_numeric(diario["wind_speed_10m_max"], errors="coerce"),
            })
            df["anio"] = df["fecha"].dt.year.astype(str)
            df["anio_mes"] = df["fecha"].dt.strftime("%Y-%m")

            anual_ciudad: dict[str, dict[str, float]] = {}
            for anio, grp in df.groupby("anio", sort=True):
                anual_ciudad[str(anio)] = {
                    "temp_media_c": round(float(grp["temp_media_c"].mean()), 2),
                    "temp_max_c": round(float(grp["temp_max_c"].max()), 2),
                    "temp_min_c": round(float(grp["temp_min_c"].min()), 2),
                    "precipitacion_mm": round(float(grp["precipitacion_mm"].sum()), 2),
                    "radiacion_solar_mj_m2": round(float(grp["radiacion_solar_mj_m2"].mean()), 2),
                    "viento_max_kmh": round(float(grp["viento_max_kmh"].max()), 2),
                    "dias_lluvia": int((grp["precipitacion_mm"] >= 1.0).sum()),
                    "dias_registrados": int(len(grp)),
                }
            anual_por_ciudad[ciudad] = anual_ciudad

            mensual_ciudad: dict[str, dict[str, float]] = {}
            for anio_mes, grp in df.groupby("anio_mes", sort=True):
                mensual_ciudad[str(anio_mes)] = {
                    "temp_media_c": round(float(grp["temp_media_c"].mean()), 2),
                    "temp_max_c": round(float(grp["temp_max_c"].max()), 2),
                    "temp_min_c": round(float(grp["temp_min_c"].min()), 2),
                    "precipitacion_mm": round(float(grp["precipitacion_mm"].sum()), 2),
                    "radiacion_solar_mj_m2": round(float(grp["radiacion_solar_mj_m2"].mean()), 2),
                    "viento_max_kmh": round(float(grp["viento_max_kmh"].max()), 2),
                    "dias_lluvia": int((grp["precipitacion_mm"] >= 1.0).sum()),
                    "dias_registrados": int(len(grp)),
                }
            mensual_por_ciudad[ciudad] = mensual_ciudad

            info_ciudades[ciudad] = {
                **meta,
                "elevacion_open_meteo_m": payload_hist.get("elevation", meta["altitud_m"]),
                "total_dias_historico": int(len(df)),
            }

            if idx < len(lista_fc):
                payload_fc = lista_fc[idx]
                crudo_pronostico[ciudad] = payload_fc
                actual = payload_fc.get("current", {})
                diario_fc = payload_fc.get("daily", {})
                pronostico_resumen[ciudad] = {
                    "fecha_hora_lectura": actual.get("time"),
                    "temp_actual_c": actual.get("temperature_2m"),
                    "humedad_relativa_pct": actual.get("relative_humidity_2m"),
                    "viento_actual_kmh": actual.get("wind_speed_10m"),
                    "temp_max_7d_c": round(float(np.max(diario_fc.get("temperature_2m_max", [0.0]))), 2),
                    "temp_min_7d_c": round(float(np.min(diario_fc.get("temperature_2m_min", [0.0]))), 2),
                    "precipitacion_7d_mm": round(float(np.sum(diario_fc.get("precipitation_sum", [0.0]))), 2),
                }
            else:
                pronostico_resumen[ciudad] = {}

    config.CRUDOS_CLIMA_JSON.write_text(
        json.dumps(crudo_diario, ensure_ascii=False),
        encoding="utf-8",
    )
    config.CRUDOS_PRONOSTICO_JSON.write_text(
        json.dumps(crudo_pronostico, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    instantanea_clima = {
        "fuente_historico": (
            "Open-Meteo Historical Weather API (ERA5 / ERA5-Land, "
            "https://archive-api.open-meteo.com/v1/archive)"
        ),
        "fuente_pronostico": "Open-Meteo Forecast API (https://api.open-meteo.com/v1/forecast)",
        "fecha_descarga_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rango_fechas_historico": [config.FECHA_INICIO_CLIMA, config.FECHA_FIN_CLIMA],
        "variables": config.VARIABLES_CLIMA,
        "ciudades": info_ciudades,
        "anual": anual_por_ciudad,
        "mensual": mensual_por_ciudad,
        "pronostico_actual": pronostico_resumen,
    }
    config.INSTANTANEA_CLIMA.write_text(
        json.dumps(instantanea_clima, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    global _CACHE_CLIMA
    _CACHE_CLIMA = instantanea_clima
    return instantanea_clima


def cargar_instantanea_bm() -> dict[str, Any]:
    """Carga en memoria la instantánea versionada del Banco Mundial."""
    global _CACHE_BM
    if _CACHE_BM is None:
        if not config.INSTANTANEA_BM.exists():
            return descargar_banco_mundial(forzar=True)
        _CACHE_BM = json.loads(config.INSTANTANEA_BM.read_text(encoding="utf-8"))
    return _CACHE_BM


def cargar_instantanea_clima() -> dict[str, Any]:
    """Carga en memoria la instantánea versionada de Open-Meteo."""
    global _CACHE_CLIMA
    if _CACHE_CLIMA is None:
        if not config.INSTANTANEA_CLIMA.exists():
            return descargar_open_meteo(forzar=True)
        _CACHE_CLIMA = json.loads(config.INSTANTANEA_CLIMA.read_text(encoding="utf-8"))
    return _CACHE_CLIMA


# ===========================================================================
# Fuente de datos de las herramientas según el modo ("instantanea" o "vivo")
# ===========================================================================
def establecer_modo(modo: str) -> None:
    """Cambia el modo de datos en caliente (usado por la API, la app y las pruebas en vivo)."""
    if modo not in config.COBERTURA:
        raise ValueError(f"Modo de datos '{modo}' no válido. Usa uno de {list(config.COBERTURA)}.")
    config.MODO_DATOS = modo


def cobertura_bm() -> tuple[int, int]:
    return config.COBERTURA[config.MODO_DATOS]["bm"]


def cobertura_clima() -> tuple[str, str]:
    from . import clientes_api

    f_ini, f_fin = config.COBERTURA[config.MODO_DATOS]["clima"]
    return f_ini, f_fin or clientes_api.fecha_fin_clima_vivo()


def serie_indicador(codigo_pais: str, indicador: str, anio_inicio: int, anio_fin: int) -> dict[str, float]:
    """Serie anual {año: valor} de un indicador para un país en el rango pedido."""
    if config.MODO_DATOS == "vivo":
        from . import clientes_api

        return clientes_api.serie_banco_mundial(
            codigo_pais, config.INDICADORES_BM[indicador]["codigo_bm"], anio_inicio, anio_fin
        )
    serie = cargar_instantanea_bm()["series"].get(codigo_pais, {}).get(indicador, {})
    return {a: v for a, v in serie.items() if anio_inicio <= int(a) <= anio_fin and v is not None}


def clima_mensual(ciudad: str, fecha_inicio: str, fecha_fin: str) -> dict[str, dict[str, float]]:
    """Agregados mensuales {AAAA-MM: métricas} de una ciudad entre dos fechas."""
    if config.MODO_DATOS == "vivo":
        from . import clientes_api

        meta = config.CIUDADES_CLIMA[ciudad]
        df = clientes_api.diario_open_meteo(meta["latitud"], meta["longitud"], fecha_inicio, fecha_fin)
        return clientes_api.agregar_periodos(df, "anio_mes")
    mensual = cargar_instantanea_clima()["mensual"][ciudad]
    return {ym: m for ym, m in mensual.items() if fecha_inicio[:7] <= ym <= fecha_fin[:7]}


def clima_anual(ciudad: str, anio: int) -> dict[str, float] | None:
    """Métricas anuales de una ciudad (None si no hay registros)."""
    if config.MODO_DATOS == "vivo":
        from . import clientes_api

        meta = config.CIUDADES_CLIMA[ciudad]
        f_fin = min(f"{anio}-12-31", clientes_api.fecha_fin_clima_vivo())
        df = clientes_api.diario_open_meteo(meta["latitud"], meta["longitud"], f"{anio}-01-01", f_fin)
        return clientes_api.agregar_periodos(df, "anio").get(str(anio))
    return cargar_instantanea_clima()["anual"][ciudad].get(str(anio))


def construir_dataframe_indicadores() -> pd.DataFrame:
    """Convierte las series del Banco Mundial en un DataFrame tabular limpio."""
    bm = cargar_instantanea_bm()
    filas: list[dict[str, Any]] = []
    for iso3, ind_dict in bm["series"].items():
        nombre_pais = config.PAISES[iso3]["nombre"]
        region = config.PAISES[iso3]["region"]
        for ind, serie_anual in ind_dict.items():
            unidad = config.INDICADORES_BM[ind]["unidad"]
            for anio_str, val in serie_anual.items():
                filas.append({
                    "codigo_pais": iso3,
                    "pais": nombre_pais,
                    "region": region,
                    "indicador": ind,
                    "unidad": unidad,
                    "anio": int(anio_str),
                    "valor": float(val),
                })
    return pd.DataFrame(filas)


def construir_dataframe_clima_anual() -> pd.DataFrame:
    """Convierte los resúmenes climáticos anuales de las 12 ciudades en un DataFrame."""
    clima = cargar_instantanea_clima()
    filas: list[dict[str, Any]] = []
    for ciudad, anios_dict in clima["anual"].items():
        meta_c = clima["ciudades"][ciudad]
        for anio_str, metricas in anios_dict.items():
            filas.append({
                "ciudad": ciudad,
                "codigo_pais": meta_c["codigo_pais"],
                "altitud_m": meta_c["altitud_m"],
                "anio": int(anio_str),
                **metricas,
            })
    return pd.DataFrame(filas)
