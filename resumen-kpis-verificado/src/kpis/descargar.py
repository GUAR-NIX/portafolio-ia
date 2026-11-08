"""Descarga los indicadores del Banco Mundial (WDI, API v2 sin clave) con caché en disco y
deja una instantánea versionada en data/instantanea_wdi.csv.

    python -m kpis.descargar            # usa la caché de data/raw/ si existe
    python -m kpis.descargar --forzar   # vuelve a pedir todo a la API

Datos: World Development Indicators, Banco Mundial, licencia CC BY 4.0.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import httpx
import pandas as pd

from . import config


def _pedir(url: str, params: dict, intentos: int = 4) -> list:
    for i in range(intentos):
        try:
            r = httpx.get(url, params=params, timeout=60)
            r.raise_for_status()
            datos = r.json()
            if not isinstance(datos, list) or len(datos) < 2:
                raise ValueError(f"Respuesta inesperada: {str(datos)[:200]}")
            return datos
        except (httpx.HTTPError, ValueError) as e:
            espera = 2 ** i
            print(f"  fallo ({e}); reintento en {espera} s", flush=True)
            time.sleep(espera)
    raise RuntimeError(f"No se pudo descargar {url}")


def descargar_indicador(ind: config.Indicador, forzar: bool = False) -> list[dict]:
    config.RAW.mkdir(parents=True, exist_ok=True)
    cache = config.RAW / f"{ind.codigo}.json"
    if cache.exists() and not forzar:
        return json.loads(cache.read_text(encoding="utf-8"))
    paises = ";".join(config.PAISES)
    url = f"{config.API_BM}/country/{paises}/indicator/{ind.codigo}"
    datos = _pedir(url, {"format": "json", "date": f"{config.ANIO_INICIO}:{config.ANIO_FIN}",
                         "per_page": 5000, "source": 2})
    cache.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    time.sleep(1)
    return datos


def main(forzar: bool = False) -> pd.DataFrame:
    filas, meta = [], {"fuente": "World Development Indicators (Banco Mundial), API v2",
                       "licencia": "CC BY 4.0", "url": config.API_BM,
                       "descargado": dt.date.today().isoformat(), "indicadores": {}}
    for ind in config.INDICADORES:
        datos = descargar_indicador(ind, forzar)
        cab, registros = datos[0], datos[1] or []
        meta["indicadores"][ind.clave] = {"codigo": ind.codigo, "nombre_bm": registros[0]["indicator"]["value"]
                                          if registros else None, "ultima_actualizacion": cab.get("lastupdated")}
        for r in registros:
            if r["value"] is None:
                continue
            filas.append({"iso3": r["countryiso3code"], "indicador": ind.clave, "anio": int(r["date"]),
                          "valor": float(r["value"])})
        print(f"{ind.codigo:24s} {sum(1 for r in registros if r['value'] is not None):4d} valores", flush=True)
    df = pd.DataFrame(filas).sort_values(["indicador", "iso3", "anio"]).reset_index(drop=True)
    df.to_csv(config.INSTANTANEA, index=False, float_format="%.6g")
    config.METADATOS.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Instantánea: {len(df)} filas -> {config.INSTANTANEA.relative_to(config.RAIZ)}")
    return df


if __name__ == "__main__":
    main(forzar="--forzar" in sys.argv)
