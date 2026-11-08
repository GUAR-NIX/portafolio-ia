"""
Une reseñas con productos y sortea la muestra que se etiqueta a mano (dev para ajustar, test para medir).

La muestra se estratifica por estrellas para tener suficientes quejas (en Amazon ~60 % son de 5★) y se limita a
reseñas de 5 a 150 palabras para que etiquetar 620 sea factible; las métricas por estrella permiten reponderar.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .muestra import escribir_indice, ids_muestra

PROPORCION_ESTRELLAS = {1: 0.20, 2: 0.15, 3: 0.15, 4: 0.20, 5: 0.30}
N_DEV, N_TEST = 300, 320
MIN_PALABRAS, MAX_PALABRAS = 5, 150


def cargar_resenas() -> pd.DataFrame:
    """Reseñas con marca, tipo y título del producto."""
    r = pd.read_parquet(config.RESENAS_RAW)
    p = pd.read_parquet(config.DIR_RAW / "productos_elegidos.parquet")
    p = p.rename(columns={"titulo": "producto"})[["parent_asin", "producto", "marca", "tipo_producto",
                                                  "calificacion_media", "n_calificaciones", "precio"]]
    r = r.merge(p, on="parent_asin", how="left")
    r["palabras"] = (r["titulo_resena"].fillna("") + " " + r["texto"].fillna("")).str.split().str.len()
    return r


def sortear_muestra(r: pd.DataFrame) -> pd.DataFrame:
    cand = r[r["palabras"].between(MIN_PALABRAS, MAX_PALABRAS)].drop_duplicates("texto")
    total = N_DEV + N_TEST
    partes = [cand[cand["estrellas"] == e].sample(round(total * p), random_state=config.SEMILLA)
              for e, p in PROPORCION_ESTRELLAS.items()]
    m = pd.concat(partes).sample(frac=1, random_state=config.SEMILLA).reset_index(drop=True)
    m["conjunto"] = ["dev"] * N_DEV + ["test"] * (len(m) - N_DEV)
    return m


def main() -> None:
    r = cargar_resenas()
    print(f"{len(r):,} reseñas · {r['parent_asin'].nunique()} productos · {r['marca'].nunique()} marcas")
    print(f"{r['palabras'].between(MIN_PALABRAS, MAX_PALABRAS).mean():.1%} tienen entre "
          f"{MIN_PALABRAS} y {MAX_PALABRAS} palabras")
    if config.INDICE_MUESTRA.exists():
        print("La muestra ya está definida (data/etiquetas/indice_muestra.jsonl): no se vuelve a sortear, las "
              "etiquetas dependen de ella. El texto se reconstruye con  python -m resenas.muestra")
        return
    m = sortear_muestra(r)
    config.DIR_RAW.mkdir(parents=True, exist_ok=True)
    cols = ["id", "conjunto", "estrellas", "tipo_producto", "marca", "titulo_resena", "texto"]
    with open(config.MUESTRA_ETIQUETADO, "w", encoding="utf-8") as fh:
        for d in m[cols].to_dict("records"):
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Muestra: {len(m)} reseñas → {config.MUESTRA_ETIQUETADO}")
    print(m.groupby(["conjunto", "estrellas"]).size().unstack())


N_DEBIL_POR_ESTRELLA = 600


def muestra_debil(r: pd.DataFrame) -> pd.DataFrame:
    """Reseñas NO etiquetadas para la supervisión débil del enfoque de embeddings (etiqueta = estrellas):
    600 por estrella, de 5 a 60 palabras (embeber cuesta ~0,2 s por texto en CPU). Se sortea excluyendo dev/test y
    luego se quitan las que caen en el conjunto en español (sorteado después), para no cambiar el sorteo."""
    excluir = ids_muestra(("dev", "test"))
    c = r[~r["id"].isin(excluir) & r["palabras"].between(5, 60)].drop_duplicates("texto")
    d = pd.concat([c[c["estrellas"] == e].sample(N_DEBIL_POR_ESTRELLA, random_state=config.SEMILLA)
                   for e in range(1, 6)]).reset_index(drop=True)
    es = ids_muestra(("es",))
    if es:
        d = d[~d["id"].isin(es)].reset_index(drop=True)
    return d


N_ES = 80


def sortear_espanol(r: pd.DataFrame, excluir: set[str]) -> pd.DataFrame:
    """80 reseñas en español (las que escriben clientes hispanohablantes en Amazon EE. UU.), fuera de dev/test."""
    from .texto import es_espanol, texto_completo
    es = r[[es_espanol(texto_completo(t, x)) for t, x in zip(r["titulo_resena"], r["texto"])]]
    es = es[~es["id"].isin(excluir) & es["palabras"].between(MIN_PALABRAS, MAX_PALABRAS)].drop_duplicates("texto")
    m = es.sample(N_ES, random_state=config.SEMILLA).reset_index(drop=True)
    m["conjunto"] = "es"
    return m


def guardar_espanol() -> None:
    if config.INDICE_MUESTRA.exists() or config.MUESTRA_ES.exists():
        print("La muestra en español ya está definida.")
        return
    excluir = ids_muestra(("dev", "test"))
    m = sortear_espanol(cargar_resenas(), excluir)
    cols = ["id", "conjunto", "estrellas", "tipo_producto", "marca", "titulo_resena", "texto"]
    with open(config.MUESTRA_ES, "w", encoding="utf-8") as fh:
        for d in m[cols].to_dict("records"):
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Muestra en español: {len(m)} reseñas → {config.MUESTRA_ES}")
    print(m["estrellas"].value_counts().sort_index().to_dict(), m["tipo_producto"].value_counts().to_dict())


if __name__ == "__main__":
    main()
    guardar_espanol()
    if not config.INDICE_MUESTRA.exists():
        print(f"Índice de la muestra: {escribir_indice()} reseñas → {config.INDICE_MUESTRA}")
