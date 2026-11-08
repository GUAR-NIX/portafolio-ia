"""Carga del corpus CFPB y quejas en español, limpieza de máscaras de anonimización (XXXX) y auditoría."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import config

RE_XXXX = re.compile(r"\bX{2,}(?:/X{2,})*\b", re.IGNORECASE)
RE_MONTO = re.compile(r"\{\$[0-9,\.]+\}")
RE_ESPACIOS = re.compile(r"\s+")


def contar_mascaras_anonimizacion(texto: str) -> tuple[int, int]:
    """Devuelve (n_mascaras_xxxx, n_montos_censurados) presentes en la narrativa CFPB."""
    if not texto:
        return 0, 0
    return len(RE_XXXX.findall(texto)), len(RE_MONTO.findall(texto))


def limpiar_texto_para_tfidf(texto: str) -> str:
    """
    Elimina las máscaras de anonimización XXXX del CFPB y normaliza montos {$...} a un token
    semántico, evitando que el clasificador léxico aprenda artefactos de censura.
    """
    if not texto:
        return ""
    t = RE_XXXX.sub(" ", texto)
    t = RE_MONTO.sub(" monto_usd ", t)
    t = RE_ESPACIOS.sub(" ", t).strip()
    return t


def preparar_texto_para_modelo(texto: str, max_palabras: int = 110) -> str:
    """
    Limpia secuencias repetitivas de XXXX y conserva las primeras `max_palabras` palabras
    (donde el cliente expone el producto financiero y el motivo central del reclamo).
    """
    if not texto:
        return ""
    t = RE_XXXX.sub("[REDACTED]", texto)
    t = re.sub(r"(\[REDACTED\]\s*){2,}", "[REDACTED] ", t)
    palabras = RE_ESPACIOS.sub(" ", t).strip().split()
    return " ".join(palabras[:max_palabras])


def cargar_corpus_cfpb(ruta: Path = config.DATOS_CRUDOS) -> pd.DataFrame:
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero: python descargar_datos.py"
        )
    df = pd.read_parquet(ruta)
    conteos = [contar_mascaras_anonimizacion(t) for t in df["texto_queja"]]
    df["n_mascaras_xxxx"] = [c[0] for c in conteos]
    df["n_montos_usd"] = [c[1] for c in conteos]
    df["tiene_xxxx"] = (df["n_mascaras_xxxx"] > 0).astype(int)
    df["Tiene_monto"] = (df["n_montos_usd"] > 0).astype(int)
    df["texto_limpio_tfidf"] = [limpiar_texto_para_tfidf(t) for t in df["texto_queja"]]
    df["texto_corto"] = [preparar_texto_para_modelo(t, max_palabras=110) for t in df["texto_queja"]]
    return df


def cargar_quejas_espanol(ruta: Path = config.QUEJAS_ES) -> pd.DataFrame:
    filas = []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if linea.strip():
            filas.append(json.loads(linea))
    df = pd.DataFrame(filas)
    df["texto_limpio_tfidf"] = [limpiar_texto_para_tfidf(t) for t in df["texto_queja"]]
    df["texto_corto"] = [preparar_texto_para_modelo(t, max_palabras=110) for t in df["texto_queja"]]
    return df


def resumen_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas del corpus CFPB por mesa de producto."""
    res = (
        df.groupby("categoria", as_index=False)
        .agg(
            n_quejas=("id_queja", "count"),
            palabras_media=("n_palabras", "mean"),
            palabras_mediana=("n_palabras", "median"),
            pct_con_xxxx=("tiene_xxxx", lambda s: float(np.mean(s) * 100.0)),
            mascaras_xxxx_media=("n_mascaras_xxxx", "mean"),
            pct_menciona_monto=("Tiene_monto", lambda s: float(np.mean(s) * 100.0)),
            n_subproductos=("subproducto", "nunique"),
            n_motivos=("motivo_original", "nunique"),
        )
    )
    res["etiqueta"] = res["categoria"].map(config.ETIQUETAS_CATEGORIA)
    orden = {c: i for i, c in enumerate(config.CATEGORIAS)}
    res["orden"] = res["categoria"].map(orden)
    return res.sort_values("orden").drop(columns=["orden"]).reset_index(drop=True)
