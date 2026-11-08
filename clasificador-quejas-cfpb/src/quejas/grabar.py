"""Corridas lentas contra Ollama, reanudables: lo ya grabado no se vuelve a pedir.

    python -m quejas.grabar embeddings                 # bge-m3 de los subconjuntos usados
    python -m quejas.grabar llm --hasta 300            # qwen2.5:7b zero-shot y few-shot sobre test_comun
    python -m quejas.grabar llm_es                     # qwen2.5:7b sobre las quejas traducidas al español

El orden del conjunto común está intercalado por mesa, así que cualquier prefijo múltiplo de 6
queda balanceado: si la corrida se corta, la evaluación usa el mayor prefijo completo.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, datos
from .llm import CacheEmbeddings, GrabadoraLLM, clasificar_texto_llm, huella


MESAS_CON_EJEMPLO_URGENTE = ("cobranza_deudas", "hipotecas")


def elegir_ejemplos_few_shot(df: pd.DataFrame, rehacer: bool = False) -> list[dict]:
    """
    Un ejemplo por mesa tomado de las quejas ANTIGUAS de entrenamiento: entre las de 40–90 palabras
    con pocas máscaras XXXX, la más cercana al centroide TF-IDF de su mesa (la más «típica»).
    En cobranza e hipotecas el ejemplo se toma entre las urgentes, para que el prompt muestre
    también la segunda tarea (2 de 6 ejemplos urgentes).
    Se graba en data/ejemplos_few_shot.json para que el prompt sea fijo y reproducible.
    """
    if config.EJEMPLOS_FEW_SHOT.exists() and not rehacer:
        return json.loads(config.EJEMPLOS_FEW_SHOT.read_text(encoding="utf-8"))
    pool = df[(df["orden_train_emb"] >= 0)].reset_index(drop=True)
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
    X = vec.fit_transform(pool["texto_limpio_tfidf"])
    ejemplos = []
    for cat in config.CATEGORIAS:
        m = (pool["categoria"] == cat).to_numpy()
        centroide = np.asarray(X[m].mean(axis=0)).ravel()
        cand = m & pool["n_palabras"].between(40, 90).to_numpy() & (pool["n_mascaras_xxxx"] <= 2).to_numpy()
        if cat in MESAS_CON_EJEMPLO_URGENTE:
            cand &= pool["urgente"].to_numpy()
        idx = np.where(cand)[0]
        sim = X[idx] @ centroide
        i = int(idx[int(np.argmax(sim))])
        fila = pool.iloc[i]
        ejemplos.append({
            "id_queja": str(fila["id_queja"]),
            "categoria": cat,
            "urgente": bool(fila["urgente"]),
            "texto_corto": datos.preparar_texto_para_modelo(fila["texto_queja"], max_palabras=90),
        })
    config.EJEMPLOS_FEW_SHOT.write_text(json.dumps(ejemplos, indent=2, ensure_ascii=False), encoding="utf-8")
    return ejemplos


def textos_para_embeddings(df: pd.DataFrame) -> list[str]:
    partes = [list(config.PROTOTIPOS_TEXTO.values())]
    for col in ("orden_comun", "orden_val_emb", "orden_train_emb"):
        partes.append(df[df[col] >= 0].sort_values(col)["texto_corto"].tolist())
    if config.QUEJAS_ES.exists():
        partes.append(datos.cargar_quejas_espanol()["texto_corto"].tolist())
    return [t for p in partes for t in p]


def correr_llm(textos: list[str], modo: str, ejemplos: list[dict], grabadora: GrabadoraLLM, etiqueta: str) -> None:
    t0 = time.perf_counter()
    nuevos = 0
    for i, t in enumerate(textos):
        if grabadora.obtener(t, modo) is not None:
            continue
        clasificar_texto_llm(t, grabadora, modo=modo, ejemplos=ejemplos, permitir_ollama=True)
        nuevos += 1
        if nuevos % 10 == 0:
            print(
                f"      [qwen2.5:7b · {modo} · {etiqueta}] {i + 1}/{len(textos)} "
                f"({(time.perf_counter() - t0) / nuevos:.1f} s/queja nueva)",
                flush=True,
            )
    print(f"      [qwen2.5:7b · {modo} · {etiqueta}] completo: {len(textos)} quejas ({nuevos} nuevas)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tarea", choices=["embeddings", "llm", "llm_es"])
    ap.add_argument("--hasta", type=int, default=300, help="tamaño del conjunto común para el LLM")
    ap.add_argument("--bloque", type=int, default=120, help="quejas por bloque alternando zero/few-shot")
    args = ap.parse_args()

    df = datos.cargar_corpus_cfpb()
    if args.tarea == "embeddings":
        cache = CacheEmbeddings(config.CACHE_EMBEDDINGS)
        textos = textos_para_embeddings(df)
        print(f"Embeddings bge-m3: {len(textos)} textos ({sum(huella(t) not in cache.vectores for t in textos)} nuevos)")
        cache.asegurar_textos(textos, lote=16)
        return

    ejemplos = elegir_ejemplos_few_shot(df)
    grabadora = GrabadoraLLM(config.GRABACION_LLM)
    if args.tarea == "llm":
        comun = df[df["orden_comun"] >= 0].sort_values("orden_comun")["texto_corto"].tolist()[: args.hasta]
        # Bloques alternos: así un corte a mitad deja zero-shot y few-shot sobre el mismo prefijo
        for ini in range(0, len(comun), args.bloque):
            parte = comun[: ini + args.bloque]
            correr_llm(parte, "cero", ejemplos, grabadora, f"comun[:{len(parte)}]")
            correr_llm(parte, "pocos", ejemplos, grabadora, f"comun[:{len(parte)}]")
    else:
        es = datos.cargar_quejas_espanol()["texto_corto"].tolist()
        correr_llm(es, "cero", ejemplos, grabadora, "espanol")
        correr_llm(es, "pocos", ejemplos, grabadora, "espanol")


if __name__ == "__main__":
    main()
