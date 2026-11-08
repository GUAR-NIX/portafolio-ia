"""Clientes HTTP reales de Banco Mundial (WDI v2) y Open-Meteo con caché en disco, pausas y reintentos.

Se usan en el modo "vivo" (`AGENTE_MODO_DATOS=vivo`) y en `descargar_datos.py`. En el modo
"instantanea" las herramientas leen `data/instantanea_*.json` y no tocan la red.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from . import config

CAMPOS_DIARIOS_OPEN_METEO = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
]


class ErrorAPI(RuntimeError):
    """Fallo definitivo de una API pública tras agotar los reintentos."""


class ClienteHTTPCacheado:
    """GET con caché en disco (clave SHA-1 de URL + parámetros), pausa mínima y reintentos exponenciales."""

    def __init__(
        self,
        dir_cache: Path | None = None,
        pausa_s: float = config.PAUSA_ENTRE_PETICIONES_S,
        reintentos: int = config.REINTENTOS_HTTP,
        timeout_s: float = 60.0,
        transporte: httpx.BaseTransport | None = None,
        dormir=time.sleep,
    ) -> None:
        self.dir_cache = dir_cache or config.CACHE_API_DIR
        self.pausa_s = pausa_s
        self.reintentos = reintentos
        self.timeout_s = timeout_s
        self.transporte = transporte
        self.dormir = dormir
        self._ultima_peticion = 0.0
        self.estadisticas = {"aciertos_cache": 0, "peticiones_red": 0, "reintentos": 0}

    def _ruta_cache(self, url: str, params: dict[str, Any]) -> Path:
        clave = json.dumps({"url": url, "params": params}, sort_keys=True, ensure_ascii=False)
        return self.dir_cache / f"{hashlib.sha1(clave.encode('utf-8')).hexdigest()}.json"

    def get_json(self, url: str, params: dict[str, Any], usar_cache: bool = True) -> Any:
        ruta = self._ruta_cache(url, params)
        if usar_cache and ruta.exists():
            self.estadisticas["aciertos_cache"] += 1
            return json.loads(ruta.read_text(encoding="utf-8"))["respuesta"]

        ultimo_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_s, follow_redirects=True, transport=self.transporte) as cliente:
            for intento in range(self.reintentos):
                espera = self.pausa_s - (time.monotonic() - self._ultima_peticion)
                if espera > 0:
                    self.dormir(espera)
                self._ultima_peticion = time.monotonic()
                try:
                    self.estadisticas["peticiones_red"] += 1
                    resp = cliente.get(url, params=params)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        retry_after = resp.headers.get("Retry-After", "")
                        pausa = float(retry_after) if retry_after.isdigit() else 2.0 * 2**intento
                        ultimo_error = ErrorAPI(f"HTTP {resp.status_code} en {url}")
                        self.estadisticas["reintentos"] += 1
                        self.dormir(pausa)
                        continue
                    resp.raise_for_status()
                    datos = resp.json()
                    break
                except (httpx.TransportError, ValueError) as exc:
                    ultimo_error = exc
                    self.estadisticas["reintentos"] += 1
                    self.dormir(2.0 * 2**intento)
            else:
                raise ErrorAPI(f"Sin respuesta válida de {url} tras {self.reintentos} intentos: {ultimo_error}")

        self.dir_cache.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps({"url": url, "params": params, "respuesta": datos}, ensure_ascii=False),
            encoding="utf-8",
        )
        return datos


_CLIENTE: ClienteHTTPCacheado | None = None


def cliente_global() -> ClienteHTTPCacheado:
    global _CLIENTE
    if _CLIENTE is None:
        _CLIENTE = ClienteHTTPCacheado()
    return _CLIENTE


def fecha_fin_clima_vivo() -> str:
    """Open-Meteo Archive publica ERA5 con unos días de rezago: se limita a hoy menos 7 días."""
    return (date.today() - timedelta(days=7)).isoformat()


# ===========================================================================
# Banco Mundial (WDI v2)
# ===========================================================================
def serie_banco_mundial(
    codigo_iso3: str,
    codigo_bm: str,
    anio_inicio: int,
    anio_fin: int,
    cliente: ClienteHTTPCacheado | None = None,
) -> dict[str, float]:
    """Serie anual {año: valor} de un indicador WDI para un país (ordenada por año, sin nulos)."""
    cli = cliente or cliente_global()
    url = f"{config.URL_BANCO_MUNDIAL}/country/{codigo_iso3}/indicator/{codigo_bm}"
    payload = cli.get_json(url, {"date": f"{anio_inicio}:{anio_fin}", "format": "json", "per_page": 500})
    if not isinstance(payload, list) or len(payload) < 2:
        mensaje = payload[0].get("message") if isinstance(payload, list) and payload else payload
        raise ErrorAPI(f"Respuesta inesperada del Banco Mundial para {codigo_iso3}/{codigo_bm}: {mensaje}")
    serie = {
        str(fila["date"]): round(float(fila["value"]), 4)
        for fila in (payload[1] or [])
        if fila.get("value") is not None and str(fila.get("date", "")).isdigit()
    }
    return dict(sorted(serie.items(), key=lambda kv: int(kv[0])))


# ===========================================================================
# Open-Meteo Archive (ERA5)
# ===========================================================================
def diario_open_meteo(
    latitud: float,
    longitud: float,
    fecha_inicio: str,
    fecha_fin: str,
    cliente: ClienteHTTPCacheado | None = None,
) -> pd.DataFrame:
    """Serie diaria de las 6 variables del catálogo para unas coordenadas."""
    cli = cliente or cliente_global()
    payload = cli.get_json(
        config.URL_OPEN_METEO_ARCHIVE,
        {
            "latitude": latitud,
            "longitude": longitud,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "daily": ",".join(CAMPOS_DIARIOS_OPEN_METEO),
            "timezone": "UTC",
        },
    )
    if "daily" not in payload:
        raise ErrorAPI(f"Open-Meteo no devolvió datos diarios: {payload.get('reason', payload)}")
    return dataframe_diario(payload["daily"])


def dataframe_diario(diario: dict[str, list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame({
        "fecha": pd.to_datetime(diario["time"]),
        "temp_media_c": pd.to_numeric(diario["temperature_2m_mean"], errors="coerce"),
        "temp_max_c": pd.to_numeric(diario["temperature_2m_max"], errors="coerce"),
        "temp_min_c": pd.to_numeric(diario["temperature_2m_min"], errors="coerce"),
        "precipitacion_mm": np.nan_to_num(pd.to_numeric(diario["precipitation_sum"], errors="coerce"), nan=0.0),
        "radiacion_solar_mj_m2": pd.to_numeric(diario["shortwave_radiation_sum"], errors="coerce"),
        "viento_max_kmh": pd.to_numeric(diario["wind_speed_10m_max"], errors="coerce"),
    })
    df["anio"] = df["fecha"].dt.year.astype(str)
    df["anio_mes"] = df["fecha"].dt.strftime("%Y-%m")
    return df


def agregar_periodos(df: pd.DataFrame, columna: str) -> dict[str, dict[str, float]]:
    """Agrega días a meses ("anio_mes") o años ("anio") con las mismas reglas que la instantánea."""
    salida: dict[str, dict[str, float]] = {}
    for clave, grp in df.groupby(columna, sort=True):
        salida[str(clave)] = {
            "temp_media_c": round(float(grp["temp_media_c"].mean()), 2),
            "temp_max_c": round(float(grp["temp_max_c"].max()), 2),
            "temp_min_c": round(float(grp["temp_min_c"].min()), 2),
            "precipitacion_mm": round(float(grp["precipitacion_mm"].sum()), 2),
            "radiacion_solar_mj_m2": round(float(grp["radiacion_solar_mj_m2"].mean()), 2),
            "viento_max_kmh": round(float(grp["viento_max_kmh"].max()), 2),
            "dias_lluvia": int((grp["precipitacion_mm"] >= 1.0).sum()),
            "dias_registrados": int(len(grp)),
        }
    return salida
