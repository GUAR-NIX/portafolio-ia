"""
Índice de la muestra etiquetada y reconstrucción de su texto.

Amazon Reviews 2023 se publica solo para investigación, así que el repositorio **no** versiona el texto de las
reseñas: guarda `data/etiquetas/indice_muestra.jsonl` con el id, el conjunto (dev / test / es), las estrellas, el tipo
de producto, la marca y una huella SHA-1 del título + texto. `python -m resenas.muestra` vuelve a armar el texto
desde `data/raw/resenas.parquet` (lo que baja `descargar_datos.py`) y comprueba cada huella: si el texto de la fuente
cambió, avisa en lugar de evaluar con reseñas distintas a las que se etiquetaron.
"""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd

from . import config

CAMPOS_INDICE = ("id", "conjunto", "estrellas", "tipo_producto", "marca", "huella")
COLUMNAS_TEXTO = ("id", "conjunto", "estrellas", "tipo_producto", "marca", "titulo_resena", "texto")


def huella(titulo: str, texto: str) -> str:
    return hashlib.sha1(f"{titulo}\n{texto}".encode("utf-8")).hexdigest()


def _leer(ruta) -> list[dict]:
    return [json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip()]


def _escribir(ruta, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        for d in filas:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def leer_indice() -> list[dict]:
    return _leer(config.INDICE_MUESTRA)


def ids_muestra(conjuntos: tuple[str, ...] | None = None) -> set[str]:
    """ids de la muestra etiquetada (desde el índice versionado; si aún no existe, desde los textos sorteados)."""
    if config.INDICE_MUESTRA.exists():
        filas = leer_indice()
    else:
        filas = [d for ruta in (config.MUESTRA_ETIQUETADO, config.MUESTRA_ES) if ruta.exists() for d in _leer(ruta)]
    return {d["id"] for d in filas if conjuntos is None or d["conjunto"] in conjuntos}


def escribir_indice() -> int:
    """Arma el índice desde los textos sorteados (solo al crear la muestra por primera vez)."""
    filas = [d for ruta in (config.MUESTRA_ETIQUETADO, config.MUESTRA_ES) for d in _leer(ruta)]
    indice = [{**{k: d[k] for k in CAMPOS_INDICE if k != "huella"}, "huella": huella(d["titulo_resena"], d["texto"])}
              for d in filas]
    _escribir(config.INDICE_MUESTRA, indice)
    return len(indice)


def reconstruir(verbose: bool = True) -> dict[str, dict]:
    """Escribe data/raw/muestra.jsonl y muestra_es.jsonl con el texto de la fuente, verificado con las huellas."""
    if not config.RESENAS_RAW.exists():
        raise FileNotFoundError(
            "Falta data/raw/resenas.parquet: el texto de las reseñas no se "
            "versiona (licencia de Amazon Reviews 2023). Ejecuta  python descargar_datos.py  y luego  "
            "python -m resenas.muestra")
    fuente = pd.read_parquet(config.RESENAS_RAW, columns=["id", "titulo_resena", "texto"]).set_index("id")
    indice = leer_indice()
    salida, distintas, ausentes = [], [], []
    for d in indice:
        if d["id"] not in fuente.index:
            ausentes.append(d["id"])
            continue
        titulo, texto = fuente.at[d["id"], "titulo_resena"], fuente.at[d["id"], "texto"]
        if huella(titulo, texto) != d["huella"]:
            distintas.append(d["id"])
            continue
        salida.append({k: d[k] for k in COLUMNAS_TEXTO if k in d} | {"titulo_resena": titulo, "texto": texto})
    if ausentes or distintas:
        raise ValueError(
            f"La descarga no coincide con la muestra etiquetada: {len(ausentes)} ids ausentes y {len(distintas)} "
            "con texto distinto (la fuente cambió o el muestreo con reservorio dio otro resultado). Las etiquetas "
            f"no se pueden aplicar a otro texto. Primeros: {(ausentes + distintas)[:5]}")
    _escribir(config.MUESTRA_ETIQUETADO, [d for d in salida if d["conjunto"] != "es"])
    _escribir(config.MUESTRA_ES, [d for d in salida if d["conjunto"] == "es"])
    if verbose:
        print(f"Texto de {len(salida)} reseñas reconstruido y verificado en data/raw/muestra*.jsonl")
    return {d["id"]: d for d in salida}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    reconstruir()
