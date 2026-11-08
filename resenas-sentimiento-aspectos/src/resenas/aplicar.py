"""
Aplica el enfoque elegido (el clásico: ver README, "por qué este") a las ~55 mil reseñas del subconjunto.

    python -m resenas.aplicar

Salida (no versionada): data/raw/menciones.parquet, una fila por (reseña, aspecto) con su polaridad, y
data/raw/resenas_analizadas.parquet con el sentimiento global predicho. Tarda ~2 min en CPU.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from . import config
from .enfoque_clasico import RUTA_MODELO, ModeloClasico
from .preparar import cargar_resenas

MENCIONES = config.DIR_RAW / "menciones.parquet"
RESENAS_ANALIZADAS = config.DIR_RAW / "resenas_analizadas.parquet"


def aplicar(lote: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = cargar_resenas()
    modelo = ModeloClasico.cargar(RUTA_MODELO)
    filas, globales = [], []
    t0 = time.perf_counter()
    for i in range(0, len(r), lote):
        parte = r.iloc[i:i + lote]
        for id_, res in zip(parte["id"], modelo.analizar_lote(list(zip(parte["titulo_resena"], parte["texto"])))):
            globales.append(res["sentimiento_global"])
            for m in res["aspectos"]:
                filas.append({"id": id_, "aspecto": m["aspecto"], "polaridad": m["polaridad"], "cita": m["cita"]})
        print(f"  {min(i + lote, len(r)):,}/{len(r):,} · {time.perf_counter() - t0:.0f} s", flush=True)
    r["sentimiento_predicho"] = globales
    menciones = pd.DataFrame(filas)
    r.to_parquet(RESENAS_ANALIZADAS, index=False)
    menciones.to_parquet(MENCIONES, index=False)
    seg = (time.perf_counter() - t0) / len(r)
    print(f"{len(r):,} reseñas · {len(menciones):,} menciones · {seg * 1000:.2f} ms por reseña")
    return r, menciones


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    aplicar()
