"""Evaluación de los cuatro enfoques en validación y prueba.

Métricas por informe: cifras extraídas, correctas / incorrectas / sin fuente, tasa de cifras alucinadas
(incorrectas + sin fuente sobre el total), cobertura de los hallazgos de referencia, informes sin
ningún error, tiempo por informe y llamadas al LLM. Los IC 95 % son bootstrap por informe (se
remuestrean informes completos, no cifras sueltas, porque las cifras de un mismo informe no son
independientes).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import comparaciones, config, generadores

N_BOOT = 2000
SEMILLA = 2026
MIN_CIFRAS_CORRECTAS = 4  # un resumen de KPIs que no cita cifras no sirve, aunque no alucine
REGLA_ELECCION = ("Entre las versiones de prompt que en validación promedian al menos 4 cifras correctas por "
                  "informe con (b), gana la de menor tasa de cifras alucinadas; si empatan, la de mayor cobertura. "
                  "La versión ganadora se usa en (a), (b) y (c) en prueba.")

# Palabras para decidir si un informe menciona un indicador (cobertura). Más estrictas que las del
# verificador: «% del PIB» no cuenta como mención del PIB.
_COBERTURA = {i.clave: list(i.palabras) for i in config.INDICADORES}
_COBERTURA["pib_usd"] += ["mayor economía", "segunda economía", "economía más grande", "tamaño de su economía"]
_COBERTURA["crecimiento"] += ["economía creció", "economía se contrajo"]
_RE_PIB = re.compile(r"(?<!del )\bpib\b(?! per)(?! real)", re.I)


def menciona(texto: str, indicador: str) -> bool:
    low = texto.lower()
    if any(p in low for p in _COBERTURA[indicador]):
        return True
    return indicador == "pib_usd" and bool(_RE_PIB.search(low))


def texto_para_cobertura(r: dict) -> str:
    """Titular, hallazgos y riesgos (las recomendaciones no cuentan como hallazgo)."""
    if r.get("informe"):
        inf = r["informe"]
        return " ".join([inf["titular"]["texto"]] + [a["texto"] for a in inf["hallazgos"] + inf["riesgos"]])
    texto = r["texto"]
    corte = re.search(r"recomendaci", texto, re.I)
    return texto[:corte.start()] if corte else texto


@lru_cache(maxsize=1)
def referencia() -> dict[str, list[dict]]:
    return json.loads(config.REFERENCIA_HALLAZGOS.read_text(encoding="utf-8"))["informes"]


def cobertura(r: dict) -> tuple[int, int, list[str]]:
    ref = referencia()[f"{r['pais']}|{r['anio']}"]
    texto = texto_para_cobertura(r)
    faltan = [h["indicador"] for h in ref if not menciona(texto, h["indicador"])]
    return len(ref) - len(faltan), len(ref), faltan


def fila_informe(r: dict, conjunto: str) -> dict:
    v = r["verificacion"] or {"n_cifras": 0, "n_correcta": 0, "n_incorrecta": 0, "n_sin_fuente": 0,
                              "tasa_alucinacion": np.nan, "n_referencias": 0, "n_referencias_ok": 0,
                              "n_marcas_unidad": 0, "n_atribucion_debil": 0, "aprobado": False}
    cub, n_ref, faltan = cobertura(r) if r["verificacion"] else (0, 4, [])
    comps = comparaciones.revisar_resultado(r)
    return {"conjunto": conjunto, "enfoque": r["enfoque"], "version": r["version"] or "-", "pais": r["pais"],
            "anio": r["anio"], "n_cifras": v["n_cifras"], "n_correcta": v["n_correcta"],
            "n_incorrecta": v["n_incorrecta"], "n_sin_fuente": v["n_sin_fuente"],
            "tasa_alucinacion": v["tasa_alucinacion"], "aprobado": bool(v["aprobado"]),
            "n_referencias": v["n_referencias"], "n_referencias_ok": v["n_referencias_ok"],
            "n_marcas_unidad": v["n_marcas_unidad"], "n_atribucion_debil": v["n_atribucion_debil"],
            "n_refs_en_linea": v.get("n_refs_en_linea", 0),
            "n_comparaciones": len(comps), "n_comparaciones_mal": sum(not c["correcta"] for c in comps),
            "cubiertos": cub, "n_hallazgos_ref": n_ref, "faltan": ",".join(faltan),
            "segundos": r["segundos_llm"] + r["segundos_verificacion"], "segundos_llm": r["segundos_llm"],
            "llamadas_llm": r["llamadas_llm"], "json_valido": r["json_valido"],
            "palabras": len(re.findall(r"\w+", r["texto"]))}


def filas_cifras(r: dict, conjunto: str) -> list[dict]:
    if not r["verificacion"]:
        return []
    out = []
    for s in r["verificacion"]["secciones"]:
        for c in s["cifras"]:
            out.append({"conjunto": conjunto, "enfoque": r["enfoque"], "version": r["version"] or "-",
                        "pais": r["pais"], "anio": r["anio"], "seccion": s["seccion"], "cifra": c["texto"],
                        "estado": c["estado"], "unidad": c["unidad"], "hecho": c["hecho_id"],
                        "valor_real": c["real_formateado"], "marcas": ",".join(c["marcas"]),
                        "oracion": s["texto"][:300]})
    return out


def correr(enfoque: str, conjunto: list[tuple[str, int]], version: str | None) -> list[dict]:
    return [generadores.generar(enfoque, p, a, version or "v2") for p, a in conjunto]


def elegir_version_prompt() -> str:
    tasas = {}
    for v in generadores.VERSIONES_PROMPT:
        res = correr("estructurado", config.VALIDACION, v)
        df = pd.DataFrame([fila_informe(r, "validacion") for r in res])
        tasa = (df["n_incorrecta"].sum() + df["n_sin_fuente"].sum()) / max(df["n_cifras"].sum(), 1)
        elegible = df["n_correcta"].mean() >= MIN_CIFRAS_CORRECTAS
        tasas[v] = (not elegible, tasa, -df["cubiertos"].sum())
    return min(tasas, key=tasas.get)


# ------------------------------------------------------------------------------------- agregados
def _tasa(df: pd.DataFrame) -> float:
    n = df["n_cifras"].sum()
    return float((df["n_incorrecta"].sum() + df["n_sin_fuente"].sum()) / n) if n else float("nan")


METRICAS = {
    "tasa_alucinacion": _tasa,
    "tasa_incorrecta": lambda d: float(d["n_incorrecta"].sum() / max(d["n_cifras"].sum(), 1)),
    "tasa_sin_fuente": lambda d: float(d["n_sin_fuente"].sum() / max(d["n_cifras"].sum(), 1)),
    "cobertura": lambda d: float(d["cubiertos"].sum() / d["n_hallazgos_ref"].sum()),
    "informes_sin_error": lambda d: float(d["aprobado"].mean()),
    "correctas_por_informe": lambda d: float(d["n_correcta"].mean()),
    "comparaciones_mal": lambda d: float(d["n_comparaciones_mal"].sum() / d["n_comparaciones"].sum())
    if d["n_comparaciones"].sum() else float("nan"),
    "segundos_por_informe": lambda d: float(d["segundos"].mean()),
}


def bootstrap(df: pd.DataFrame, fn, n: int = N_BOOT, semilla: int = SEMILLA) -> tuple[float, float]:
    rng = np.random.default_rng(semilla)
    k = len(df)
    vals = [fn(df.iloc[rng.integers(0, k, k)]) for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def bootstrap_pareado(a: pd.DataFrame, b: pd.DataFrame, fn, n: int = N_BOOT, semilla: int = SEMILLA):
    """IC de fn(a) − fn(b) remuestreando los mismos informes en ambos enfoques."""
    a = a.sort_values(["pais", "anio"]).reset_index(drop=True)
    b = b.sort_values(["pais", "anio"]).reset_index(drop=True)
    rng = np.random.default_rng(semilla)
    k = len(a)
    difs = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        difs.append(fn(a.iloc[idx]) - fn(b.iloc[idx]))
    return float(fn(a) - fn(b)), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (conj, enf, ver), g in df.groupby(["conjunto", "enfoque", "version"], sort=False):
        fila = {"conjunto": conj, "enfoque": enf, "version": ver, "informes": len(g),
                "cifras": int(g["n_cifras"].sum()), "correctas": int(g["n_correcta"].sum()),
                "incorrectas": int(g["n_incorrecta"].sum()), "sin_fuente": int(g["n_sin_fuente"].sum()),
                "cifras_por_informe": float(g["n_cifras"].mean()), "llamadas_llm": float(g["llamadas_llm"].mean()),
                "palabras": float(g["palabras"].mean()),
                "refs_en_linea": float(g["n_refs_en_linea"].mean()),
                "comparaciones": int(g["n_comparaciones"].sum()),
                "comparaciones_mal_n": int(g["n_comparaciones_mal"].sum()),
                "referencias_ok": float(g["n_referencias_ok"].sum() / g["n_referencias"].sum())
                if g["n_referencias"].sum() else float("nan")}
        for m, fn in METRICAS.items():
            fila[m] = fn(g)
            if m != "segundos_por_informe":
                fila[f"{m}_ic"] = bootstrap(g, fn)
        filas.append(fila)
    return pd.DataFrame(filas)
