"""
Hallazgos de negocio sobre las ~55 mil reseñas analizadas con el enfoque elegido.

    python -m resenas.hallazgos

* Qué aspectos generan más quejas y elogios (y su versión corregida con la precisión/exhaustividad medida en test).
* Quejas por tipo de producto, por marca y por producto.
* Evolución anual.
* Relación con las estrellas: cuántas estrellas "cuesta" cada queja (regresión lineal con todas a la vez).

Salida versionada: reports/hallazgos/*.csv y reports/hallazgos.json (la app los usa sin descargar nada).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config
from .aplicar import MENCIONES, RESENAS_ANALIZADAS

DIR_HALLAZGOS = config.DIR_REPORTES / "hallazgos"
MIN_RESENAS_MARCA = 600
MIN_RESENAS_PRODUCTO = 150


def indicadores(r: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Una fila por reseña con columnas neg_<aspecto> y pos_<aspecto> (0/1)."""
    piv = (m.assign(v=1).pivot_table(index="id", columns=["polaridad", "aspecto"], values="v", aggfunc="max")
           .fillna(0))
    salida = r[["id"]].copy()
    for a in config.ASPECTOS:
        for pol, pref in (("negativo", "neg"), ("positivo", "pos")):
            col = (pol, a)
            salida[f"{pref}_{a}"] = salida["id"].map(piv[col]).fillna(0).astype(int) if col in piv else 0
    return salida.set_index("id")


def tablas(r: pd.DataFrame, ind: pd.DataFrame, evaluacion: dict | None = None) -> dict[str, pd.DataFrame]:
    r = r.set_index("id")
    neg = ind[[f"neg_{a}" for a in config.ASPECTOS]].set_axis(list(config.ASPECTOS), axis=1)
    pos = ind[[f"pos_{a}" for a in config.ASPECTOS]].set_axis(list(config.ASPECTOS), axis=1)
    t = {}
    a = pd.DataFrame({"pct_neg": neg.mean() * 100, "pct_pos": pos.mean() * 100,
                      "menciones": (neg.sum() + pos.sum()).astype(int)})
    a["share_quejas"] = a["pct_neg"] / (a["pct_neg"] + a["pct_pos"]) * 100
    if evaluacion:   # corrección de prevalencia: predichas × precisión / exhaustividad (medidas en test)
        pa = evaluacion["enfoques"]["clasico"]["por_aspecto"]
        a["factor_correccion"] = [pa[x]["precision"] / pa[x]["exhaustividad"]
                                  if pa[x]["exhaustividad"] and pa[x]["soporte"] >= 10 else np.nan for x in a.index]
        a["pct_neg_corregido"] = a["pct_neg"] * a["factor_correccion"]
    t["aspectos"] = a.sort_values("pct_neg", ascending=False)

    t["tipo_aspecto"] = (neg.groupby(r["tipo_producto"]).mean() * 100).round(1)
    t["tipo_resumen"] = pd.DataFrame({
        "resenas": r.groupby("tipo_producto").size(),
        "estrellas_medias": r.groupby("tipo_producto")["estrellas"].mean().round(2),
        "pct_con_alguna_queja": (neg.max(axis=1).groupby(r["tipo_producto"]).mean() * 100).round(1),
        "queja_principal": (neg.groupby(r["tipo_producto"]).mean()).idxmax(axis=1),
    }).sort_values("pct_con_alguna_queja", ascending=False)

    t["estrellas_aspecto"] = (neg.groupby(r["estrellas"]).mean() * 100).round(1)
    anio = r["fecha"].dt.year
    validos = anio.value_counts()[lambda s: s >= 1000].index
    t["anio_aspecto"] = (neg[anio.isin(validos)].groupby(anio[anio.isin(validos)]).mean() * 100).round(2).sort_index()

    marcas = r["marca"].value_counts()[lambda s: s >= MIN_RESENAS_MARCA].index
    en_marca = r["marca"].isin(marcas)
    g_neg = neg[en_marca].groupby(r.loc[en_marca, "marca"]).mean() * 100
    g_pos = pos[en_marca].groupby(r.loc[en_marca, "marca"]).mean() * 100
    t["marcas"] = pd.DataFrame({
        "resenas": r[en_marca].groupby("marca").size(),
        "tipos": r[en_marca].groupby("marca")["tipo_producto"].agg(lambda s: ", ".join(sorted(s.unique()))),
        "estrellas_medias": r[en_marca].groupby("marca")["estrellas"].mean().round(2),
        "pct_queja_calidad": g_neg["calidad"].round(1), "pct_elogio_calidad": g_pos["calidad"].round(1),
        "neto_calidad": (g_pos["calidad"] - g_neg["calidad"]).round(1),
        "queja_principal": g_neg.idxmax(axis=1), "pct_queja_principal": g_neg.max(axis=1).round(1),
    }).sort_values("neto_calidad")

    prods = r["parent_asin"].value_counts()[lambda s: s >= MIN_RESENAS_PRODUCTO].index
    en_p = r["parent_asin"].isin(prods)
    gp = neg[en_p].groupby(r.loc[en_p, "parent_asin"]).mean() * 100
    info = r[en_p].groupby("parent_asin").agg(producto=("producto", "first"), marca=("marca", "first"),
                                              tipo=("tipo_producto", "first"), resenas=("estrellas", "size"),
                                              estrellas_medias=("estrellas", "mean"))
    info["producto"] = info["producto"].str.slice(0, 70)
    info["queja_principal"] = gp.idxmax(axis=1)
    info["pct_queja_principal"] = gp.max(axis=1).round(1)
    info["pct_con_alguna_queja"] = (neg[en_p].max(axis=1).groupby(r.loc[en_p, "parent_asin"]).mean() * 100).round(1)
    t["productos"] = info.round(2).sort_values("pct_con_alguna_queja", ascending=False)

    t["impacto_estrellas"] = impacto_en_estrellas(r["estrellas"], neg, pos)
    t["impacto_estrellas_a_mano"] = impacto_con_etiquetas_a_mano(r)

    # Tendencia dentro de cada producto: se resta la tasa media de cada producto antes de promediar por año, para
    # que el cambio en la mezcla de productos a lo largo del tiempo no se confunda con un cambio real.
    centrado = neg - neg.groupby(r["parent_asin"]).transform("mean")
    t["anio_aspecto_intra_producto"] = ((centrado[anio.isin(validos)].groupby(anio[anio.isin(validos)]).mean()
                                         + neg.mean()) * 100).round(2).sort_index()
    return t


def impacto_con_etiquetas_a_mano(r: pd.DataFrame) -> pd.DataFrame:
    """La misma regresión con las 620 reseñas etiquetadas a mano (dev + test). Las etiquetas se pusieron sin ver
    las estrellas, así que no hay circularidad (el clasificador del enfoque clásico sí aprendió de las estrellas).
    Pesos = proporción de cada estrella en la población / en la muestra (que se estratificó por estrellas)."""
    from .etiquetas import leer_jsonl
    filas = leer_jsonl(config.ETIQUETAS_DEV) + leer_jsonl(config.ETIQUETAS_TEST)
    estrellas = r["estrellas"].reindex([d["id"] for d in filas]).values
    neg = pd.DataFrame([{a: int(any(m["aspecto"] == a and m["polaridad"] == "negativo" for m in d["aspectos"]))
                         for a in config.ASPECTOS} for d in filas])
    pos = pd.DataFrame([{a: int(any(m["aspecto"] == a and m["polaridad"] == "positivo" for m in d["aspectos"]))
                         for a in config.ASPECTOS} for d in filas])
    pob = r["estrellas"].value_counts(normalize=True)
    mue = pd.Series(estrellas).value_counts(normalize=True)
    pesos = np.array([pob[e] / mue[e] for e in estrellas])
    return impacto_en_estrellas(pd.Series(estrellas), neg, pos, pesos)


def impacto_en_estrellas(estrellas: pd.Series, neg: pd.DataFrame, pos: pd.DataFrame,
                         pesos: np.ndarray | None = None) -> pd.DataFrame:
    """MCO (ponderado si hay pesos): estrellas ~ 1 + quejas + elogios. El coeficiente de una queja = estrellas de
    diferencia frente a una reseña igual sin esa queja (asociación, no causalidad). Errores estándar robustos (HC1)."""
    X = np.column_stack([np.ones(len(neg)), neg.values, pos.values]).astype(float)
    y = estrellas.values.astype(float)
    w = np.ones(len(y)) if pesos is None else np.asarray(pesos, dtype=float)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ beta
    n, k = X.shape
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    cov = XtX_inv @ (Xw.T * e ** 2) @ Xw @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    nombres = ["intercepto"] + [f"queja_{a}" for a in neg.columns] + [f"elogio_{a}" for a in pos.columns]
    return pd.DataFrame({"coef": beta, "ee": se, "ic_bajo": beta - 1.96 * se, "ic_alto": beta + 1.96 * se},
                        index=nombres).round(3)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    r = pd.read_parquet(RESENAS_ANALIZADAS)
    m = pd.read_parquet(MENCIONES)
    ev_path = config.DIR_REPORTES / "evaluacion.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    t = tablas(r, indicadores(r, m), ev)
    DIR_HALLAZGOS.mkdir(parents=True, exist_ok=True)
    for nombre, df in t.items():
        df.to_csv(DIR_HALLAZGOS / f"{nombre}.csv", encoding="utf-8")
    resumen = {
        "resenas": len(r), "productos": int(r["parent_asin"].nunique()), "marcas": int(r["marca"].nunique()),
        "menciones": len(m), "desde": str(r["fecha"].min().date()), "hasta": str(r["fecha"].max().date()),
        "pct_resenas_con_algun_aspecto": round(r["id"].isin(m["id"]).mean() * 100, 1),
        "pct_resenas_con_alguna_queja": round(r["id"].isin(m.loc[m["polaridad"] == "negativo", "id"]).mean() * 100, 1),
        "estrellas_medias": round(r["estrellas"].mean(), 2),
    }
    (config.DIR_REPORTES / "hallazgos.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    from .graficos import figuras_hallazgos
    figuras_hallazgos(t)
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(t["aspectos"].round(2).to_string())
    print(t["tipo_resumen"].to_string())
    comp = t["impacto_estrellas"][["coef"]].join(t["impacto_estrellas_a_mano"][["coef", "ic_bajo", "ic_alto"]],
                                                 rsuffix="_a_mano")
    print(comp.to_string())
    print(t["anio_aspecto_intra_producto"][["calidad", "carga", "ajuste"]].to_string())


if __name__ == "__main__":
    main()
