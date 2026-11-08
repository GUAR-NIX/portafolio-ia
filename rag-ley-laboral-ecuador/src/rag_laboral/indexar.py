"""
Embeddings bge-m3 de los 627 artículos → data/indice/emb_articulos.npy (float16, 1,3 MB, versionado).

    python -m rag_laboral.indexar

Reanudable: cada vector se guarda en data/raw/cache_embeddings.jsonl apenas Ollama lo devuelve.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config
from .llm import embeber
from .recuperar import cargar_articulos, texto_para_indexar


def indexar(lote: int = 8) -> dict:
    articulos = cargar_articulos()
    textos = [texto_para_indexar(a) for a in articulos]
    t0 = time.perf_counter()
    vectores = []
    for i in range(0, len(textos), lote):
        vectores += embeber(textos[i:i + lote], lote=lote)
        print(f"\r{min(i + lote, len(textos))}/{len(textos)} artículos", end="", flush=True)
    m = np.array(vectores, dtype=np.float32)
    config.INDICE.mkdir(parents=True, exist_ok=True)
    np.save(config.INDICE / "emb_articulos.npy", m.astype(np.float16))
    info = {"articulos": len(articulos), "dimension": int(m.shape[1]), "modelo": config.MODELO_EMB,
            "segundos": round(time.perf_counter() - t0, 1)}
    (config.INDICE / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    return info


if __name__ == "__main__":
    print(indexar())
