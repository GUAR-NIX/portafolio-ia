"""Enfoques de clasificación de quejas CFPB y piezas de enrutamiento:
  1. Clásico léxico      : TF-IDF (1-2 gramas, sin máscaras XXXX) + Regresión Logística
  2. Zero-Shot bge-m3    : similitud coseno contra prototipos semánticos de las 6 mesas
  3. Embeddings + LR     : bge-m3 (1.024 dims) + Regresión Logística (+ curva few-shot)
  4. LLM                 : qwen2.5:7b zero-shot y few-shot con salida JSON Schema
  5. Cascada             : el clasificador barato decide; solo las quejas de baja confianza van al LLM
  + Calibración por temperatura y detección de urgencia (reglas, TF-IDF + LR y LLM).
"""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from . import config
from .llm import CacheEmbeddings, GrabadoraLLM, clasificar_texto_llm


def _tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
    )


class ModeloTFIDF:
    """TF-IDF sublineal de 1–2 gramas + Regresión Logística multinomial (clases balanceadas)."""

    def __init__(self, C: float = 4.0, max_features: int = 60000) -> None:
        self.pipeline = Pipeline([
            ("tfidf", _tfidf(max_features)),
            ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=config.SEMILLA)),
        ])
        self.clases_: list[str] = list(config.CATEGORIAS)
        self.ms_por_queja_: float = 0.25
        self.temperatura_: float = 1.0

    def ajustar(self, textos_limpios: list[str], y: list[str] | np.ndarray) -> "ModeloTFIDF":
        self.pipeline.fit(textos_limpios, y)
        self.clases_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def logits(self, textos_limpios: list[str]) -> np.ndarray:
        t0 = time.perf_counter()
        z = self.pipeline.decision_function(textos_limpios)
        self.ms_por_queja_ = max((time.perf_counter() - t0) / max(len(textos_limpios), 1) * 1000.0, 0.01)
        idx = [self.clases_.index(c) for c in config.CATEGORIAS]
        return z[:, idx]

    def predecir_proba(self, textos_limpios: list[str], calibrado: bool = True) -> np.ndarray:
        t = self.temperatura_ if calibrado else 1.0
        return softmax(self.logits(textos_limpios) / t, axis=1)

    def predecir(self, textos_limpios: list[str]) -> list[str]:
        probs = self.predecir_proba(textos_limpios)
        return [config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)]


class ModeloZeroShotEmbeddings:
    """Clasificador Zero-Shot por similitud coseno en el espacio bge-m3 contra descripciones de mesa."""

    def __init__(self, temperatura_escala: float = 16.0) -> None:
        self.temperatura_escala = temperatura_escala
        self.matriz_prototipos_: np.ndarray | None = None

    def ajustar_prototipos(self, cache: CacheEmbeddings, permitir_ollama: bool = True) -> "ModeloZeroShotEmbeddings":
        textos_proto = [config.PROTOTIPOS_TEXTO[c] for c in config.CATEGORIAS]
        self.matriz_prototipos_ = cache.matriz(textos_proto, permitir_ollama=permitir_ollama)
        return self

    def predecir_proba_desde_vectores(self, X_vec: np.ndarray) -> np.ndarray:
        if self.matriz_prototipos_ is None:
            raise RuntimeError("Llama primero a ajustar_prototipos().")
        similitud = X_vec @ self.matriz_prototipos_.T  # (N, 6)
        return softmax(similitud * self.temperatura_escala, axis=1)

    def predecir_desde_vectores(self, X_vec: np.ndarray) -> list[str]:
        probs = self.predecir_proba_desde_vectores(X_vec)
        return [config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)]


class ModeloEmbeddingsSupervisado:
    """Regresión Logística L2 sobre embeddings densos bge-m3 (1.024 dimensiones)."""

    def __init__(self, C: float = 4.0) -> None:
        self.C = C
        self.clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=config.SEMILLA)
        self.clases_: list[str] = list(config.CATEGORIAS)
        self.temperatura_: float = 1.0

    def ajustar_vectores(self, X_vec: np.ndarray, y: list[str] | np.ndarray) -> "ModeloEmbeddingsSupervisado":
        self.clf.fit(X_vec, y)
        self.clases_ = list(self.clf.classes_)
        return self

    def logits(self, X_vec: np.ndarray) -> np.ndarray:
        z = self.clf.decision_function(X_vec)
        idx = [self.clases_.index(c) for c in config.CATEGORIAS]
        return z[:, idx]

    def predecir_proba_desde_vectores(self, X_vec: np.ndarray, calibrado: bool = True) -> np.ndarray:
        t = self.temperatura_ if calibrado else 1.0
        return softmax(self.logits(X_vec) / t, axis=1)

    def predecir_desde_vectores(self, X_vec: np.ndarray) -> list[str]:
        probs = self.predecir_proba_desde_vectores(X_vec)
        return [config.CATEGORIAS[i] for i in np.argmax(probs, axis=1)]


def ajustar_temperatura(logits: np.ndarray, y: list[str] | np.ndarray) -> float:
    """Escalado de temperatura (Guo et al., 2017): un solo parámetro T que minimiza la log-pérdida en validación.
    No cambia la clase predicha, solo qué tan seguras son las probabilidades."""
    idx = np.array([config.CATEGORIAS.index(str(c)) for c in y])

    def nll(t: float) -> float:
        p = softmax(logits / t, axis=1)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def curva_aprendizaje_few_shot(
    X_train_vec: np.ndarray,
    X_train_tfidf_textos: list[str],
    y_train: np.ndarray,
    X_eval_vec: np.ndarray,
    X_eval_tfidf_textos: list[str],
    y_eval: np.ndarray,
    ejemplos_por_clase: tuple[int, ...] = (2, 5, 10, 30, 100, 300),
    n_repeticiones: int = 5,
) -> list[dict[str, Any]]:
    """
    Compara cuántos ejemplos etiquetados por mesa necesitan bge-m3 + LR y TF-IDF + LR
    (promediando sobre `n_repeticiones` sorteos; con k igual al total disponible hay un solo sorteo posible).
    """
    y_train_arr = np.asarray(y_train)
    y_eval_arr = np.asarray(y_eval)
    filas = []

    for k in ejemplos_por_clase:
        f1_emb_lista = []
        f1_tfidf_lista = []
        for rep in range(n_repeticiones):
            rng = np.random.default_rng(config.SEMILLA + rep * 100 + k)
            indices = []
            for cat in config.CATEGORIAS:
                idx_cat = np.where(y_train_arr == cat)[0]
                tomar = min(len(idx_cat), k)
                indices.extend(rng.choice(idx_cat, size=tomar, replace=False).tolist())
            idx_arr = np.array(indices, dtype=int)

            m_emb = ModeloEmbeddingsSupervisado(C=4.0).ajustar_vectores(X_train_vec[idx_arr], y_train_arr[idx_arr])
            f1_emb_lista.append(float(f1_score(y_eval_arr, m_emb.predecir_desde_vectores(X_eval_vec), average="macro")))

            sub_txt = [X_train_tfidf_textos[i] for i in idx_arr]
            m_tf = ModeloTFIDF(C=4.0, max_features=25000).ajustar(sub_txt, y_train_arr[idx_arr])
            f1_tfidf_lista.append(float(f1_score(y_eval_arr, m_tf.predecir(X_eval_tfidf_textos), average="macro")))

        filas.append({
            "ejemplos_por_clase": int(k),
            "total_ejemplos": int(k * len(config.CATEGORIAS)),
            "f1_macro_bge_m3_media": float(np.mean(f1_emb_lista)),
            "f1_macro_bge_m3_std": float(np.std(f1_emb_lista)),
            "f1_macro_tfidf_media": float(np.mean(f1_tfidf_lista)),
            "f1_macro_tfidf_std": float(np.std(f1_tfidf_lista)),
        })
    return filas


class ModeloLLM:
    """qwen2.5:7b (zero-shot o few-shot). Las probabilidades salen de los logprobs de la primera ficha
    de la categoría; si faltan, se usa la respuesta como distribución casi determinista."""

    def __init__(self, grabadora: GrabadoraLLM, modo: str = "cero", ejemplos: list[dict] | None = None) -> None:
        self.grabadora = grabadora
        self.modo = modo
        self.ejemplos = ejemplos
        self.segundos_medio_: float = 0.0

    def predecir_lote(self, textos_cortos: list[str], permitir_ollama: bool = False) -> dict[str, Any]:
        n, k = len(textos_cortos), len(config.CATEGORIAS)
        probs = np.zeros((n, k), dtype=float)
        preds, urg, p_urg, tiempos, validas = [], [], [], [], []
        for i, t in enumerate(textos_cortos):
            reg = clasificar_texto_llm(t, self.grabadora, modo=self.modo, ejemplos=self.ejemplos, permitir_ollama=permitir_ollama)
            cat = reg["categoria"]
            if reg.get("probs_categoria"):
                fila = np.array([reg["probs_categoria"].get(c, 0.0) for c in config.CATEGORIAS], dtype=float)
            else:
                fila = np.full(k, 0.01 / (k - 1))
                fila[config.CATEGORIAS.index(cat)] = 0.99
            # La categoría decodificada (con gramática) manda: si no coincide con el argmax de la ficha, se respeta.
            if np.argmax(fila) != config.CATEGORIAS.index(cat):
                fila[config.CATEGORIAS.index(cat)] = fila.max() + 1e-6
            probs[i] = fila / fila.sum()
            preds.append(cat)
            urg.append(bool(reg.get("urgente", False)))
            pu = reg.get("p_urgente")
            p_urg.append(float(pu) if pu is not None else float(reg.get("urgente", False)))
            tiempos.append(float(reg.get("segundos", 0.0)))
            validas.append(bool(reg.get("respuesta_valida", True)))
        self.segundos_medio_ = float(np.mean(tiempos)) if tiempos else 0.0
        return {
            "preds": preds,
            "probs": probs,
            "urgente": np.array(urg, dtype=bool),
            "p_urgente": np.array(p_urg, dtype=float),
            "segundos": np.array(tiempos, dtype=float),
            "validas": np.array(validas, dtype=bool),
        }


def cascada(probs_barato: np.ndarray, probs_llm: np.ndarray, umbral: float) -> tuple[np.ndarray, np.ndarray]:
    """El clasificador barato decide cuando su confianza máxima es >= umbral; si no, decide el LLM.
    Devuelve (probabilidades combinadas, máscara de quejas enviadas al LLM)."""
    al_llm = probs_barato.max(axis=1) < umbral
    combinadas = np.where(al_llm[:, None], probs_llm, probs_barato)
    return combinadas, al_llm


def umbral_por_presupuesto(confianzas_val: np.ndarray, fraccion_al_llm: float) -> float:
    """Umbral tal que, en validación, la fracción indicada de quejas (las de menor confianza) va al LLM."""
    return float(np.quantile(confianzas_val, fraccion_al_llm))


# ---------------------------------------------------------------------------
# Detección de urgencia (segunda tarea)
# ---------------------------------------------------------------------------
RE_URGENCIA = re.compile(
    r"\b(fraud\w*|identity theft|stolen identity|identity (?:was )?stolen|scam\w*|"
    r"foreclos\w*|sheriff sale|lose (?:my|our) home|loan modification|"
    r"repossess\w*|repo(?:ed)?\b|garnish\w*|arrest\w*|jail|seiz\w*|levy|"
    r"sued|lawsuit|summons)\b",
    re.IGNORECASE,
)


def urgencia_por_reglas(textos: list[str]) -> np.ndarray:
    """Línea base: palabras clave de fraude, ejecución hipotecaria, reposesión y amenazas legales."""
    return np.array([bool(RE_URGENCIA.search(t or "")) for t in textos], dtype=bool)


class ModeloUrgenciaTFIDF:
    """TF-IDF + Regresión Logística binaria (urgente sí/no) con umbral elegido en validación."""

    def __init__(self, C: float = 4.0) -> None:
        self.pipeline = Pipeline([
            ("tfidf", _tfidf(60000)),
            ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=config.SEMILLA)),
        ])
        self.umbral_: float = 0.5

    def ajustar(self, textos_limpios: list[str], y: np.ndarray) -> "ModeloUrgenciaTFIDF":
        self.pipeline.fit(textos_limpios, np.asarray(y, dtype=int))
        return self

    def proba(self, textos_limpios: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(textos_limpios)[:, 1]

    def elegir_umbral(self, textos_val: list[str], y_val: np.ndarray) -> float:
        p = self.proba(textos_val)
        y = np.asarray(y_val, dtype=int)
        mejores = [(f1_score(y, (p >= u).astype(int), zero_division=0), u) for u in np.linspace(0.05, 0.95, 91)]
        self.umbral_ = float(max(mejores)[1])
        return self.umbral_

    def predecir(self, textos_limpios: list[str]) -> np.ndarray:
        return self.proba(textos_limpios) >= self.umbral_
