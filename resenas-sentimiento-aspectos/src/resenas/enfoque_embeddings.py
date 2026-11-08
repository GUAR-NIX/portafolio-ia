"""
Enfoque 2 · embeddings `bge-m3` (Ollama) + clasificadores lineales.

* Cada reseña se divide en cláusulas (`texto.clausulas`) y cada cláusula se convierte en un vector de 1024 dims.
* **Aspectos**: una regresión logística por aspecto sobre los vectores de cláusula. Las etiquetas de entrenamiento
  salen del conjunto dev hecho a mano: una cláusula es positiva para un aspecto si contiene (o está contenida en)
  la cita que anoté. Umbral por validación cruzada en dev.
* **Polaridad**: regresión logística de 3 clases sobre las cláusulas con aspecto del dev (polaridad anotada) más
  reseñas cortas no etiquetadas con su etiqueta de estrellas (supervisión débil, con menos peso).
* **Sentimiento global**: regresión logística sobre el vector de la reseña completa, entrenada con estrellas.

Los vectores se guardan en una caché por huella del texto: la evaluación se reproduce sin Ollama.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config
from .llm import embeber
from .texto import clausulas, normalizar, texto_completo

DIM = 1024


def huella(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:16]


class CacheEmbeddings:
    """Vectores por huella de texto en un .npz (float16). `embeber_faltantes` solo llama a Ollama por lo nuevo."""

    def __init__(self, ruta: Path, funcion=embeber):
        self.ruta = Path(ruta)
        self.funcion = funcion
        self.vectores: dict[str, np.ndarray] = {}
        self.segundos_por_texto: float | None = None
        if self.ruta.exists():
            z = np.load(self.ruta)
            for h, v in zip(z["huellas"], z["vectores"]):
                self.vectores[str(h)] = v.astype(np.float32)

    def guardar(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        hs = list(self.vectores)
        tmp = self.ruta.with_name(self.ruta.stem + "_tmp.npz")
        np.savez_compressed(tmp, huellas=np.array(hs),
                            vectores=np.stack([self.vectores[h] for h in hs]).astype(np.float16))
        tmp.replace(self.ruta)

    def embeber_faltantes(self, textos: list[str], lote: int = 64, guardar_cada: int = 512) -> int:
        faltan = list(dict.fromkeys(t for t in textos if huella(t) not in self.vectores))
        t0, hechos = time.perf_counter(), 0
        for i in range(0, len(faltan), lote):
            parte = faltan[i:i + lote]
            for t, v in zip(parte, self.funcion(parte)):
                self.vectores[huella(t)] = v
            hechos += len(parte)
            if hechos % guardar_cada < lote:
                self.guardar()
                print(f"  embeddings: {hechos}/{len(faltan)} · {(time.perf_counter() - t0) / hechos:.2f} s/texto",
                      flush=True)
        if faltan:
            self.segundos_por_texto = (time.perf_counter() - t0) / len(faltan)
            self.guardar()
        return len(faltan)

    def matriz(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack([self.vectores[huella(t)] for t in textos])


def clausula_tiene_cita(clausula: str, cita: str) -> bool:
    """La cláusula corresponde a la cita si una contiene a la otra o comparten >= 60 % de las palabras de la cita."""
    c, q = normalizar(clausula).strip(" .!?,"), normalizar(cita).strip(" .!?,")
    if not c or not q:
        return False
    if q in c or c in q:
        return True
    pc, pq = set(c.split()), set(q.split())
    return len(pc & pq) / max(1, len(pq)) >= 0.6


def etiquetas_clausulas(resenas: list[dict], etiquetas: dict[str, dict]) -> tuple[list[str], np.ndarray, list]:
    """Para entrenar: cláusulas de las reseñas etiquetadas, matriz Y (cláusulas × aspectos) y la polaridad de cada
    (cláusula, aspecto) positiva."""
    aspectos = list(config.ASPECTOS)
    textos, filas, pols = [], [], []
    for r in resenas:
        ets = etiquetas[r["id"]]["aspectos"]
        for c in clausulas(texto_completo(r["titulo_resena"], r["texto"])):
            y = np.zeros(len(aspectos), dtype=int)
            for m in ets:
                if clausula_tiene_cita(c, m["cita"]):
                    k = aspectos.index(m["aspecto"])
                    y[k] = 1
                    pols.append((len(textos), m["polaridad"]))
            textos.append(c)
            filas.append(y)
    return textos, np.array(filas), pols


@dataclass
class ModeloEmbeddings:
    C_aspecto: float = 2.0
    umbral: float = 0.5
    peso_debil: float = 0.3
    aspectos: list[str] = field(default_factory=lambda: list(config.ASPECTOS))
    clf_aspecto: dict = field(default_factory=dict)
    clf_polaridad: LogisticRegression | None = None
    clf_global: LogisticRegression | None = None

    def entrenar(self, X_cl: np.ndarray, Y: np.ndarray, pols: list, X_debil: np.ndarray, y_debil: list[str],
                 X_glob: np.ndarray, y_glob: list[str]) -> "ModeloEmbeddings":
        for k, a in enumerate(self.aspectos):
            if Y[:, k].min() == Y[:, k].max():      # aspecto sin ejemplos: nunca se predice
                self.clf_aspecto[a] = None
                continue
            clf = LogisticRegression(C=self.C_aspecto, max_iter=3000, class_weight="balanced")
            clf.fit(X_cl, Y[:, k])
            self.clf_aspecto[a] = clf
        Xp = np.vstack([X_cl[[i for i, _ in pols]], X_debil])
        yp = [p for _, p in pols] + list(y_debil)
        w = np.r_[np.ones(len(pols)), np.full(len(y_debil), self.peso_debil)]
        self.clf_polaridad = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xp, yp, sample_weight=w)
        self.clf_global = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(X_glob, y_glob)
        return self

    def prob_aspectos(self, X_cl: np.ndarray) -> np.ndarray:
        return np.column_stack([self.clf_aspecto[a].predict_proba(X_cl)[:, 1] if self.clf_aspecto[a] is not None
                                else np.zeros(len(X_cl)) for a in self.aspectos])

    def analizar(self, cls: list[str], X_cl: np.ndarray, x_glob: np.ndarray) -> dict:
        glob = str(self.clf_global.predict(x_glob.reshape(1, -1))[0])
        if not len(cls):
            return {"sentimiento_global": glob, "aspectos": []}
        P = self.prob_aspectos(X_cl)
        Ppol = self.clf_polaridad.predict_proba(X_cl)
        clases = list(self.clf_polaridad.classes_)
        menciones = []
        for k, a in enumerate(self.aspectos):
            sel = np.where(P[:, k] >= self.umbral)[0]
            if not len(sel):
                continue
            pp = (Ppol[sel] * P[sel, k:k + 1]).sum(axis=0)   # polaridad ponderada por la confianza del aspecto
            mejor = sel[np.argmax(P[sel, k])]
            menciones.append({"aspecto": a, "polaridad": clases[int(np.argmax(pp))], "cita": cls[mejor]})
        return {"sentimiento_global": glob, "aspectos": menciones}


def preparar_textos(resenas: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Cláusulas por reseña y texto completo de cada reseña."""
    completos = [texto_completo(r["titulo_resena"], r["texto"]) for r in resenas]
    return [clausulas(c) for c in completos], completos


def analizar_lote(modelo: ModeloEmbeddings, cache: CacheEmbeddings, resenas: list[dict]) -> list[dict]:
    cls, completos = preparar_textos(resenas)
    return [modelo.analizar(c, cache.matriz(c), cache.matriz([t])[0]) for c, t in zip(cls, completos)]


RUTA_CACHE = config.DIR_DATOS / "embeddings_etiquetadas.npz"   # versionada (~10 MB): evaluación sin Ollama


def resumen_json(modelo: ModeloEmbeddings) -> str:
    return json.dumps({"C_aspecto": modelo.C_aspecto, "umbral": modelo.umbral, "peso_debil": modelo.peso_debil})


def main() -> None:
    """Calcula (o completa) la caché versionada: cláusulas y textos de las 700 reseñas etiquetadas (dev, test, español) y las 3000
    reseñas de supervisión débil. ~25 min en CPU; reanudable."""
    import sys

    from .etiquetas import cargar_muestra
    from .preparar import cargar_resenas, muestra_debil
    sys.stdout.reconfigure(encoding="utf-8")
    muestra = list(cargar_muestra().values())
    debil = muestra_debil(cargar_resenas())
    cls, completos = preparar_textos(muestra)
    textos = [c for cs in cls for c in cs] + completos
    textos += [texto_completo(t, x) for t, x in zip(debil["titulo_resena"], debil["texto"])]
    cache = CacheEmbeddings(RUTA_CACHE)
    print(f"{len(set(textos))} textos únicos, {len(cache.vectores)} ya en caché", flush=True)
    n = cache.embeber_faltantes(textos)
    print(f"Listo: {n} textos nuevos · {cache.segundos_por_texto} s/texto")


if __name__ == "__main__":
    main()
