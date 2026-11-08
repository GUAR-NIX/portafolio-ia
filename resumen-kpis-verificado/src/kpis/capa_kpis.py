"""Capa de KPIs determinista: la fuente de verdad contra la que se verifica cada cifra.

Todo sale de la instantánea data/instantanea_wdi.csv: tablas por país y año, variaciones interanuales,
mediana regional, brecha con la región, rankings y anomalías. `hechos(pais, anio)` devuelve la lista
plana de cifras que un informe puede citar; el verificador solo acepta números que estén aquí.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

MIN_PAISES_MEDIANA = 10   # la mediana regional solo se publica si hay datos de al menos 10 países
MIN_HISTORIA_ANOMALIA = 8  # años previos de variaciones para estimar lo «normal» de un país
UMBRAL_ANOMALIA = 3.0      # |z robusto| a partir del cual un cambio interanual es anómalo
VENTANA_DATO_ESCASO = 3    # pobreza y Gini: se usa el último dato de hasta 3 años antes
# Piso de la escala del z robusto: evita que series muy suaves (población) marquen como anómalo
# un cambio de décimas. Un cambio debe superar 3 × (0,5 pp | 2 %) para ser anomalía.
PISO_ESCALA = {"pp": 0.5, "%": 2.0}


def es_tasa(clave: str) -> bool:
    """Indicadores en % o índice: su variación se expresa en puntos (pp), no en % relativo."""
    return config.IND_POR_CLAVE[clave].unidad in ("%", "índice")


@lru_cache(maxsize=1)
def cargar() -> pd.DataFrame:
    df = pd.read_csv(config.INSTANTANEA)
    return df[df["iso3"].isin(config.PAISES) & df["indicador"].isin(config.IND_POR_CLAVE)].copy()


@lru_cache(maxsize=1)
def tabla_larga() -> pd.DataFrame:
    """Una fila por (indicador, país, año) con valor, valor previo, variación, mediana, brecha, ranking y z."""
    df = cargar().sort_values(["indicador", "iso3", "anio"]).copy()
    g = df.groupby(["indicador", "iso3"])
    df["anio_previo"] = g["anio"].shift(1)
    df["valor_previo"] = g["valor"].shift(1)
    consecutivo = df["anio_previo"] == df["anio"] - 1
    tasa = df["indicador"].map(es_tasa)
    var_pp = df["valor"] - df["valor_previo"]
    var_pct = (df["valor"] / df["valor_previo"] - 1) * 100
    df["variacion"] = np.where(consecutivo, np.where(tasa, var_pp, var_pct), np.nan)
    df["unidad_variacion"] = np.where(tasa, "pp", "%")

    # Mediana regional y ranking (1 = valor más alto) por indicador y año
    grupo = df.groupby(["indicador", "anio"])["valor"]
    n = grupo.transform("count")
    df["n_paises"] = n
    df["mediana_region"] = np.where(n >= MIN_PAISES_MEDIANA, grupo.transform("median"), np.nan)
    df["ranking"] = grupo.rank(ascending=False, method="min")
    df["brecha_region"] = np.where(tasa, df["valor"] - df["mediana_region"],
                                   (df["valor"] / df["mediana_region"] - 1) * 100)

    # Anomalía: z robusto de la variación frente a las variaciones previas del mismo país (sin mirar el futuro)
    df["z_anomalia"] = np.nan
    for (ind, _), sub in df.groupby(["indicador", "iso3"]):
        piso = PISO_ESCALA["pp" if es_tasa(ind) else "%"]
        v = sub["variacion"].to_numpy()
        for k, idx in enumerate(sub.index):
            hist = v[:k][~np.isnan(v[:k])]
            if np.isnan(v[k]) or len(hist) < MIN_HISTORIA_ANOMALIA:
                continue
            med = np.median(hist)
            mad = 1.4826 * np.median(np.abs(hist - med))
            escala = max(mad, piso)
            df.at[idx, "z_anomalia"] = (v[k] - med) / escala
    df["anomalia"] = df["z_anomalia"].abs() >= UMBRAL_ANOMALIA
    return df.reset_index(drop=True)


def tabla_pais_anio(pais: str) -> pd.DataFrame:
    """Tabla ancha de un país: filas = años, columnas = indicadores."""
    t = tabla_larga()
    return t[t["iso3"] == pais].pivot(index="anio", columns="indicador", values="valor")


def mediana_regional() -> pd.DataFrame:
    t = tabla_larga()
    return t.dropna(subset=["mediana_region"]).groupby(["indicador", "anio"])["mediana_region"].first().unstack(0)


def ranking_regional(indicador: str, anio: int) -> pd.DataFrame:
    t = tabla_larga()
    r = t[(t["indicador"] == indicador) & (t["anio"] == anio)][["iso3", "valor", "ranking", "n_paises"]]
    r = r.assign(pais=r["iso3"].map(config.PAISES)).sort_values("ranking")
    return r.reset_index(drop=True)


def comparacion_con_region(pais: str = "ECU", anio: int | None = None) -> pd.DataFrame:
    """Valor del país, mediana regional, brecha y puesto por indicador (por defecto Ecuador)."""
    t = tabla_larga()
    sub = t[t["iso3"] == pais]
    if anio is not None:
        sub = sub[sub["anio"] == anio]
    cols = ["indicador", "anio", "valor", "mediana_region", "brecha_region", "ranking", "n_paises", "variacion"]
    return sub[cols].reset_index(drop=True)


def anomalias(pais: str | None = None, desde: int = config.ANIO_INICIO) -> pd.DataFrame:
    t = tabla_larga()
    a = t[t["anomalia"] & (t["anio"] >= desde)]
    if pais:
        a = a[a["iso3"] == pais]
    return a[["indicador", "iso3", "anio", "valor_previo", "valor", "variacion", "unidad_variacion",
              "z_anomalia"]].sort_values(["anio", "indicador"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Hechos citables
# ----------------------------------------------------------------------------------------------
METRICAS = ("valor", "variacion", "mediana_region", "brecha_region", "ranking")


@dataclass(frozen=True)
class Hecho:
    id: str
    indicador: str
    pais: str          # ISO3 o "LAT" (mediana regional)
    anio: int
    metrica: str       # valor | variacion | mediana_region | brecha_region | ranking
    valor: float
    unidad: str        # % | pp | USD | personas | índice | puesto
    n: int | None = None          # países con dato (solo ranking)
    anomalia: bool = False

    def dict(self) -> dict:
        return asdict(self)


def _unidad_valor(clave: str) -> str:
    return config.IND_POR_CLAVE[clave].unidad


def _fila(t: pd.DataFrame, ind: str, pais: str, anio: int, escaso: bool) -> pd.Series | None:
    sub = t[(t["indicador"] == ind) & (t["iso3"] == pais) & (t["anio"] <= anio)]
    if sub.empty:
        return None
    ult = sub.iloc[-1]
    if ult["anio"] == anio or (escaso and ult["anio"] >= anio - VENTANA_DATO_ESCASO):
        return ult
    return None


@lru_cache(maxsize=256)
def _hechos_cache(pais: str, anio: int) -> tuple[Hecho, ...]:
    t = tabla_larga()
    salida: list[Hecho] = []
    for ind in config.INDICADORES:
        escaso = ind.clave in ("pobreza", "gini", "deuda_externa")
        for a in (anio, anio - 1):
            f = _fila(t, ind.clave, pais, a, escaso)
            if f is None or (a == anio - 1 and int(f["anio"]) != anio - 1):
                continue
            fa = int(f["anio"])
            base = f"{ind.clave}|{pais}|{fa}"
            uv = _unidad_valor(ind.clave)
            salida.append(Hecho(f"{base}|valor", ind.clave, pais, fa, "valor", float(f["valor"]), uv,
                                anomalia=bool(f["anomalia"])))
            if a != anio:
                continue
            if pd.notna(f["variacion"]):
                salida.append(Hecho(f"{base}|variacion", ind.clave, pais, fa, "variacion", float(f["variacion"]),
                                    str(f["unidad_variacion"]), anomalia=bool(f["anomalia"])))
            if pd.notna(f["mediana_region"]):
                salida.append(Hecho(f"{ind.clave}|LAT|{fa}|mediana_region", ind.clave, "LAT", fa, "mediana_region",
                                    float(f["mediana_region"]), uv))
                salida.append(Hecho(f"{base}|brecha_region", ind.clave, pais, fa, "brecha_region",
                                    float(f["brecha_region"]), "pp" if es_tasa(ind.clave) else "%"))
            salida.append(Hecho(f"{base}|ranking", ind.clave, pais, fa, "ranking", float(f["ranking"]), "puesto",
                                n=int(f["n_paises"])))
    return tuple(salida)


def hechos(pais: str, anio: int) -> list[Hecho]:
    """Cifras citables en el informe de (país, año): valor del año y del anterior, variación,
    mediana regional, brecha y ranking. Pobreza, Gini y deuda usan el último dato de hasta 3 años antes."""
    return list(_hechos_cache(pais, anio))


def hechos_por_id(pais: str, anio: int) -> dict[str, Hecho]:
    return {h.id: h for h in hechos(pais, anio)}


# ----------------------------------------------------------------------------------------------
# Formato en español (lo usan los prompts, la plantilla y la app)
# ----------------------------------------------------------------------------------------------
def fmt_num(x: float, dec: int = 1) -> str:
    s = f"{abs(x):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("-" if x < 0 and float(s.replace(".", "").replace(",", ".")) != 0 else "") + s


def fmt_hecho(h: Hecho, con_signo: bool = False) -> str:
    """Texto canónico de una cifra: '3,5 %', '+1,2 pp', '120,8 mil millones de USD', 'puesto 3 de 17'."""
    if h.unidad == "puesto":
        return f"puesto {int(h.valor)} de {h.n}"
    signo = "+" if con_signo and h.valor > 0 else ""
    if h.unidad == "USD":
        if abs(h.valor) >= 1e9:
            return f"{signo}{fmt_num(h.valor / 1e9, 1)} mil millones de USD"
        return f"{signo}{fmt_num(h.valor, 0)} USD"
    if h.unidad == "personas":
        return f"{signo}{fmt_num(h.valor / 1e6, 1)} millones de personas"
    if h.unidad == "índice":
        return f"{signo}{fmt_num(h.valor, 1)}"
    if h.unidad == "pp":
        return f"{signo}{fmt_num(h.valor, 1)} pp"
    return f"{signo}{fmt_num(h.valor, 1)} %"


def tabla_para_prompt(pais: str, anio: int) -> str:
    """Tabla compacta (Markdown) con todas las cifras citables de un informe."""
    hs = hechos(pais, anio)
    por = {(h.indicador, h.pais, h.anio, h.metrica): h for h in hs}
    filas = ["| indicador (clave) | año | valor | año anterior | variación | mediana región | brecha vs región "
             "| puesto (1 = más alto) | ¿anomalía? |", "|---|---|---|---|---|---|---|---|---|"]
    for ind in config.INDICADORES:
        v = [h for h in hs if h.indicador == ind.clave and h.metrica == "valor" and h.pais == pais]
        if not v:
            continue
        actual = max(v, key=lambda h: h.anio)
        a = actual.anio
        prev = por.get((ind.clave, pais, a - 1, "valor"))
        var = por.get((ind.clave, pais, a, "variacion"))
        med = por.get((ind.clave, "LAT", a, "mediana_region"))
        bre = por.get((ind.clave, pais, a, "brecha_region"))
        rk = por.get((ind.clave, pais, a, "ranking"))
        celdas = [f"{ind.nombre} ({ind.clave})", str(a), fmt_hecho(actual),
                  fmt_hecho(prev) if prev else "s/d", fmt_hecho(var, True) if var else "s/d",
                  fmt_hecho(med) if med else "s/d", fmt_hecho(bre, True) if bre else "s/d",
                  fmt_hecho(rk) if rk else "s/d", "SÍ" if actual.anomalia else "no"]
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


@lru_cache(maxsize=1)
def indice_hechos() -> dict[tuple[str, str, int], tuple[Hecho, ...]]:
    """Todas las cifras de la capa (los 18 países, la mediana regional, todos los años), indexadas por
    (indicador, país, año). El verificador busca aquí, no solo en la tabla del informe."""
    t = tabla_larga()
    idx: dict[tuple[str, str, int], list[Hecho]] = {}
    for f in t.itertuples(index=False):
        ind, pais, a = f.indicador, f.iso3, int(f.anio)
        uv, base = _unidad_valor(ind), f"{f.indicador}|{f.iso3}|{a}"
        lista = idx.setdefault((ind, pais, a), [])
        lista.append(Hecho(f"{base}|valor", ind, pais, a, "valor", float(f.valor), uv, anomalia=bool(f.anomalia)))
        if pd.notna(f.variacion):
            lista.append(Hecho(f"{base}|variacion", ind, pais, a, "variacion", float(f.variacion),
                               str(f.unidad_variacion), anomalia=bool(f.anomalia)))
        if pd.notna(f.mediana_region):
            lista.append(Hecho(f"{base}|brecha_region", ind, pais, a, "brecha_region", float(f.brecha_region),
                               "pp" if es_tasa(ind) else "%"))
            reg = idx.setdefault((ind, "LAT", a), [])
            if not reg:
                reg.append(Hecho(f"{ind}|LAT|{a}|mediana_region", ind, "LAT", a, "mediana_region",
                                 float(f.mediana_region), uv))
        lista.append(Hecho(f"{base}|ranking", ind, pais, a, "ranking", float(f.ranking), "puesto",
                           n=int(f.n_paises)))
    return {k: tuple(v) for k, v in idx.items()}


def buscar_hecho(indicador: str, pais: str, anio: int, metrica: str) -> Hecho | None:
    for h in indice_hechos().get((indicador, pais, anio), ()):
        if h.metrica == metrica:
            return h
    return None
