"""Regenera las instantáneas versionadas (modo "instantanea") desde las APIs del Banco Mundial y Open-Meteo.

El modo "vivo" del agente no necesita este script: consulta las APIs al vuelo con caché en data/cache_api/.
"""
from __future__ import annotations

import argparse

from agente import config, datos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga datos reales del Banco Mundial (1990-2024) y Open-Meteo (2020-2024)."
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Fuerza la consulta en vivo a las APIs aunque ya existan las instantáneas locales.",
    )
    args = parser.parse_args()

    print("1/2 · Consultando API del Banco Mundial (12 países × 8 indicadores, 1990-2024)...")
    bm = datos.descargar_banco_mundial(forzar=args.forzar)
    total_obs_bm = sum(r["n_observaciones"] for r in bm["resumen_cobertura"])
    print(
        f"    -> Instantánea guardada en {config.INSTANTANEA_BM.relative_to(config.RAIZ)} "
        f"({total_obs_bm} observaciones anuales reales)."
    )

    print("2/2 · Consultando API Open-Meteo Archive & Forecast (12 ciudades, 2020-2024)...")
    clima = datos.descargar_open_meteo(forzar=args.forzar)
    total_dias = sum(c["total_dias_historico"] for c in clima["ciudades"].values())
    print(
        f"    -> Instantánea guardada en {config.INSTANTANEA_CLIMA.relative_to(config.RAIZ)} "
        f"({len(clima['ciudades'])} ciudades, {total_dias} registros diarios procesados)."
    )


if __name__ == "__main__":
    main()
