"""
Recuperación en cuatro variantes que se evalúan por separado:

    bm25        palabras exactas (rank-bm25 sobre tokens normalizados)
    vectorial   significado (bge-m3 por Ollama, coseno en numpy)
    hibrido     Reciprocal Rank Fusion de las dos listas
    rerank      híbrido → 20 candidatos → bge-reranker-v2-m3 (cross-encoder) → top 4–6

Con `sinonimos=True` la consulta se expande antes de buscar («décimo tercero» → «decimotercera remuneración»).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .llm import Almacen, embeber, huella
from .texto import expandir, tokens, ventana_relevante


def cargar_articulos(ruta: Path = config.ARTICULOS) -> list[dict]:
    return [json.loads(ln) for ln in ruta.read_text(encoding="utf-8").splitlines() if ln.strip()]


def texto_para_indexar(a: dict) -> str:
    """Lo que ven BM25 y el embedding: la ubicación del artículo ayuda («Capítulo IV: De las vacaciones»)."""
    partes = [a["fuente"], f"Art. {a['numero']}", a.get("epigrafe", ""), a.get("seccion", ""), a.get("capitulo", ""),
              a.get("titulo", ""), a["texto"]]
    return ". ".join(p for p in partes if p)


def texto_para_rerank(a: dict, consulta: str, max_chars: int = 1500) -> str:
    """El cross-encoder lee ~512 tokens: ubicación + epígrafe + la parte del artículo más parecida a la consulta."""
    cab = f"{a['fuente']}, Art. {a['numero']}. {a.get('epigrafe', '')}. {a.get('seccion') or a.get('capitulo', '')}"
    return f"{cab}\n{ventana_relevante(a['texto'], consulta, max_chars)}"


class ConsultasGuardadas:
    """Embeddings de consultas en float16 (versionados): la evaluación y las pruebas corren sin Ollama."""

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self.vectores: dict[str, np.ndarray] = {}
        if ruta.exists():
            z = np.load(ruta)
            self.vectores = {k: z["vectores"][i] for i, k in enumerate(z["claves"])}

    def guardar(self, nuevas: dict[str, np.ndarray]) -> None:
        self.vectores.update(nuevas)
        claves = sorted(self.vectores)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.ruta, claves=np.array(claves),
                            vectores=np.stack([self.vectores[k] for k in claves]).astype(np.float16))


@dataclass
class Resultado:
    id: str
    puntaje: float
    rango: int


class Buscador:
    def __init__(self, articulos: list[dict] | None = None, indice: Path = config.INDICE):
        self.articulos = articulos or cargar_articulos()
        self.por_id = {a["id"]: a for a in self.articulos}
        self.ids = [a["id"] for a in self.articulos]
        self.indice = indice
        self.consultas = ConsultasGuardadas(indice / "emb_consultas.npz")
        self.cache_rerank = Almacen(indice / "rerank_cache.jsonl")
        self._reranker = None

    # ---------- BM25 ----------
    @cached_property
    def bm25(self) -> BM25Okapi:
        return BM25Okapi([tokens(texto_para_indexar(a)) for a in self.articulos], k1=1.2, b=0.75)

    def bm25_top(self, consulta: str, k: int) -> list[Resultado]:
        puntajes = self.bm25.get_scores(tokens(consulta))
        orden = np.argsort(-puntajes)[:k]
        return [Resultado(self.ids[i], float(puntajes[i]), r + 1) for r, i in enumerate(orden) if puntajes[i] > 0]

    # ---------- Vectorial ----------
    @cached_property
    def matriz(self) -> np.ndarray:
        ruta = self.indice / "emb_articulos.npy"
        m = np.load(ruta).astype(np.float32)
        if m.shape[0] != len(self.articulos):
            raise RuntimeError("emb_articulos.npy no coincide con articulos.jsonl: corre  python -m rag_laboral.indexar")
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def vector_consulta(self, consulta: str) -> np.ndarray:
        clave = huella(config.MODELO_EMB, consulta)
        v = self.consultas.vectores.get(clave)
        if v is None:
            if config.PROVEEDOR == "grabado":
                raise RuntimeError("Modo grabado: esta consulta no tiene embedding guardado (abre Ollama).")
            v = np.array(embeber([consulta])[0], dtype=np.float32)
            if os.environ.get("RAG_GRABAR", "0") == "1":
                self.consultas.guardar({clave: v})
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    def vectorial_top(self, consulta: str, k: int) -> list[Resultado]:
        s = self.matriz @ self.vector_consulta(consulta)
        orden = np.argsort(-s)[:k]
        return [Resultado(self.ids[i], float(s[i]), r + 1) for r, i in enumerate(orden)]

    # ---------- Fusión ----------
    @staticmethod
    def rrf(listas: list[list[Resultado]], k: int, k_rrf: int = config.K_RRF) -> list[Resultado]:
        acumulado: dict[str, float] = {}
        for lista in listas:
            for r in lista:
                acumulado[r.id] = acumulado.get(r.id, 0.0) + 1.0 / (k_rrf + r.rango)
        orden = sorted(acumulado.items(), key=lambda x: -x[1])[:k]
        return [Resultado(i, s, n + 1) for n, (i, s) in enumerate(orden)]

    def hibrido_top(self, consulta: str, k: int) -> list[Resultado]:
        profundidad = max(k, 50)
        return self.rrf([self.bm25_top(consulta, profundidad), self.vectorial_top(consulta, profundidad)], k)

    # ---------- Re-ranking ----------
    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # import pesado: solo si hace falta
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
        return self._reranker

    def puntajes_rerank(self, consulta: str, ids: list[str]) -> list[float]:
        claves = [huella("bge-reranker-v2-m3", consulta, i) for i in ids]
        faltan = [n for n, c in enumerate(claves) if self.cache_rerank.get(c) is None]
        if faltan:
            if config.PROVEEDOR == "grabado":
                raise RuntimeError("Modo grabado: faltan puntajes del re-ranker para esta consulta.")
            pares = [(consulta, texto_para_rerank(self.por_id[ids[n]], consulta)) for n in faltan]
            for n, s in zip(faltan, self.reranker.predict(pares, batch_size=4)):
                self.cache_rerank.guardar(claves[n], round(float(s), 5))
        return [self.cache_rerank.get(c) for c in claves]

    def rerank_top(self, consulta: str, k: int, candidatos: int = config.K_CANDIDATOS) -> list[Resultado]:
        base = self.hibrido_top(consulta, candidatos)
        s = self.puntajes_rerank(consulta, [r.id for r in base])
        orden = sorted(zip(base, s), key=lambda x: -x[1])[:k]
        return [Resultado(r.id, float(p), n + 1) for n, (r, p) in enumerate(orden)]

    # ---------- Punto de entrada ----------
    def buscar(self, consulta: str, metodo: str = "rerank", k: int = config.K_CONTEXTO,
               sinonimos: bool | str = True) -> list[Resultado]:
        """sinonimos: False (consulta tal cual), "v1" (lista inicial) o True / "v2" (lista corregida)."""
        q = consulta if not sinonimos else expandir(consulta, "v1" if sinonimos == "v1" else "v2")
        funciones = {"bm25": self.bm25_top, "vectorial": self.vectorial_top, "hibrido": self.hibrido_top,
                     "rerank": self.rerank_top}
        return funciones[metodo](q, k)
