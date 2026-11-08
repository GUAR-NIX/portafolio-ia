"""Generación de las 10 figuras analíticas de IA_05_clasificador_quejas en reports/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

AZUL, NARANJA, VERDE, AMARILLO, ROSA, VIOLETA, VERDE_OSC, ROJO = config.COLORES
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c3"
TEXTO, TEXTO2, SUP = "#0b0b0b", "#52514e", "#fcfcfb"

NOMBRES_CORTOS = {
    "reporte_credito": "Reporte crédito",
    "cobranza_deudas": "Cobranza deudas",
    "hipotecas": "Hipotecas",
    "tarjetas_credito": "Tarjetas crédito",
    "cuentas_bancarias": "Cuentas bancarias",
    "prestamos_consumo": "Préstamos consumo",
}


def num(v: float, dec: int = 1) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUP,
        "axes.facecolor": SUP,
        "savefig.facecolor": SUP,
        "savefig.dpi": 150,
        "font.size": 10,
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO2,
        "xtick.color": TEXTO2,
        "ytick.color": TEXTO2,
        "axes.edgecolor": "#d8d7d2",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "axes.axisbelow": True,
        "axes.titlesize": 11.8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 25,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })


def _titulo(ax, t: str, s: str | None = None) -> None:
    ax.set_title(t)
    if s:
        ax.annotate(
            s,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8.6,
            color=TEXTO2,
            va="bottom",
        )


def _guardar(fig, nombre: str):
    fig.tight_layout()
    if nombre:
        config.FIGURAS.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    return fig


def _pct(ax, eje: str = "y", dec: int = 0, escala: float = 1.0) -> None:
    f = mtick.FuncFormatter(lambda v, _: f"{num(v * escala, dec)} %")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def _corto(nombre: str) -> str:
    """Quita el número de orden del nombre del enfoque para etiquetas compactas."""
    return nombre.split(". ", 1)[-1]


def fig_01_datos_y_particiones(resumen_cat: pd.DataFrame, particiones: pd.DataFrame, resumen_fuente: dict,
                               nombre: str = "01_datos_y_particiones_temporales.png"):
    """Izq.: quejas por mesa y partición temporal. Der.: % de narrativas con máscaras XXXX y montos."""
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.2))
    cats = config.CATEGORIAS
    y = np.arange(len(cats))
    izq = np.zeros(len(cats))
    for col, color, etiqueta in (("train", AZUL, "train (hasta mar-2017)"), ("test_misma_epoca", GRIS, "reserva misma época"),
                                 ("val", AMARILLO, "val (abr–sep 2017)"), ("test", NARANJA, "test (oct-2017 en adelante)")):
        v = particiones[col].reindex(cats).to_numpy() / 1e3
        a1.barh(y, v, left=izq, color=color, height=0.6, label=etiqueta)
        izq += v
    a1.invert_yaxis()
    a1.set_yticks(y, [NOMBRES_CORTOS[c] for c in cats], fontsize=9)
    a1.set_xlabel("Miles de quejas")
    a1.grid(axis="y", visible=False)
    a1.legend(fontsize=7.8, loc="lower right")
    _titulo(a1, f"{num(sum(izq) * 1e3, 0)} quejas reales con fecha de recepción",
            f"{resumen_fuente['fecha_min']} a {resumen_fuente['fecha_max']} · partición temporal, no aleatoria")
    ancho = 0.36
    a2.barh(y - ancho / 2, resumen_cat["pct_con_xxxx"], height=ancho, color=NARANJA, label="Con máscaras XXXX")
    a2.barh(y + ancho / 2, resumen_cat["pct_menciona_monto"], height=ancho, color=VERDE, label="Con montos {$...}")
    a2.invert_yaxis()
    a2.set_yticks(y, [NOMBRES_CORTOS[c] for c in cats], fontsize=9)
    _pct(a2, "x")
    a2.set_xlim(0, 110)
    a2.grid(axis="y", visible=False)
    a2.legend(loc="lower right", fontsize=8.5)
    _titulo(a2, "Censura de privacidad del CFPB en las narrativas", "Las máscaras se eliminan antes de TF-IDF")
    return _guardar(fig, nombre)


def fig_02_comparacion_enfoques(tabla: pd.DataFrame, nombre: str = "02_comparacion_enfoques.png"):
    """Macro-F1 con IC 95 % y latencia de cada enfoque sobre el mismo conjunto común."""
    estilo()
    sub = tabla.reset_index(drop=True)
    n = int(sub["n_quejas"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(sub))
    colores = [AZUL, GRIS, AMARILLO, VERDE, ROJO, VIOLETA, ROSA, NARANJA][: len(sub)]
    f1 = sub["f1_macro"].to_numpy() * 100
    lo = f1 - sub["f1_ic95_bajo"].to_numpy() * 100
    hi = sub["f1_ic95_alto"].to_numpy() * 100 - f1
    a1.barh(y, f1, xerr=[lo, hi], color=colores, height=0.6, capsize=3, ecolor=TEXTO2)
    a1.invert_yaxis()
    a1.set_yticks(y, [_corto(e) for e in sub["enfoque"]], fontsize=8.6)
    _pct(a1, "x")
    a1.set_xlim(0, 112)
    a1.grid(axis="y", visible=False)
    for i in range(len(sub)):
        a1.text(f1[i] + hi[i] + 1.5, i, f"{num(f1[i], 1)} %", va="center", fontsize=8.2)
    _titulo(a1, f"Macro-F1 sobre las mismas {n} quejas recientes", "Barras: IC 95 % por bootstrap (1.000 remuestreos)")
    ms = sub["ms_por_queja"].to_numpy()
    a2.barh(y, ms, color=colores, height=0.6)
    a2.set_xscale("log")
    a2.invert_yaxis()
    a2.set_yticks(y, [""] * len(sub))
    a2.set_xlabel("Milisegundos por queja en CPU (escala log)")
    a2.grid(axis="y", visible=False)
    for i, v in enumerate(ms):
        a2.text(v * 1.15, i, f"{num(v / 1000, 1)} s" if v >= 1000 else f"{num(v, 2 if v < 10 else 0)} ms", va="center", fontsize=8.2)
    a2.set_xlim(min(ms) / 3, max(ms) * 20)
    _titulo(a2, "Latencia por queja", "LLM y embeddings medidos con otra carga en la misma CPU")
    return _guardar(fig, nombre)


def fig_03_curva_few_shot(curva: pd.DataFrame, tabla: pd.DataFrame, n_tfidf: str, n_llm0: str, n_llm1: str,
                          nombre: str = "03_curva_few_shot.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    k = curva["ejemplos_por_clase"].to_numpy()
    for col, color, etiqueta in (("bge_m3", VERDE, "bge-m3 + LR"), ("tfidf", AZUL, "TF-IDF + LR")):
        m = curva[f"f1_macro_{col}_media"].to_numpy() * 100
        s = curva[f"f1_macro_{col}_std"].to_numpy() * 100
        ax.plot(k, m, marker="o", color=color, label=f"{etiqueta} (k ejemplos por mesa)")
        ax.fill_between(k, m - s, m + s, color=color, alpha=0.15)
    f1 = tabla.set_index("enfoque")["f1_macro"] * 100
    for n, color, ls in ((n_tfidf, AZUL, ":"), (n_llm0, ROJO, "--"), (n_llm1, VIOLETA, "-.")):
        ax.axhline(f1[n], color=color, ls=ls, lw=1.4, label=f"{_corto(n)}: {num(f1[n], 1)} %")
    ax.set_xscale("log")
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("Ejemplos etiquetados por mesa (6 mesas)")
    ax.set_ylabel("Macro-F1")
    _pct(ax)
    ax.set_xlim(k.min() * 0.8, k.max() * 1.3)
    ax.legend(fontsize=7.8, loc="lower right")
    _titulo(ax, "¿Cuántas etiquetas hacen falta?", "Media ± desv. estándar en 5 sorteos · mismo conjunto común que el LLM")
    return _guardar(fig, nombre)


def fig_04_deriva_temporal(deriva: pd.DataFrame, por_trim: pd.DataFrame, f1_clase: pd.DataFrame,
                           nombre: str = "04_deriva_temporal.png"):
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.2))
    x = np.arange(len(por_trim))
    a1.plot(x, por_trim["f1_macro"] * 100, marker="o", color=NARANJA, label="Quejas posteriores al corte (por trimestre)")
    ref = deriva.iloc[0]
    a1.axhspan(ref["f1_ic95_bajo"] * 100, ref["f1_ic95_alto"] * 100, color=GRIS_CLARO, alpha=0.5)
    a1.axhline(ref["f1_macro"] * 100, color=GRIS, ls="--", lw=1.3, label="Misma época que train (IC 95 %)")
    a1.set_xticks(x, por_trim["trimestre"], fontsize=8.5)
    a1.set_ylabel("Macro-F1 de TF-IDF + LR")
    _pct(a1)
    a1.legend(fontsize=8, loc="lower left")
    _titulo(a1, "Deriva: entrenado con quejas hasta marzo de 2017", "El CFPB cambió su lista de productos y motivos en abril de 2017")
    y = np.arange(len(f1_clase))
    a2.barh(y - 0.18, f1_clase["f1_misma_epoca"] * 100, height=0.36, color=GRIS, label="Misma época")
    a2.barh(y + 0.18, f1_clase["f1_recientes"] * 100, height=0.36, color=NARANJA, label="Recientes")
    a2.invert_yaxis()
    a2.set_yticks(y, [NOMBRES_CORTOS[c] for c in f1_clase["categoria"]], fontsize=9)
    _pct(a2, "x")
    a2.set_xlim(0, 105)
    a2.grid(axis="y", visible=False)
    a2.legend(fontsize=8, loc="lower right")
    _titulo(a2, "F1 por mesa antes y después del corte", "TF-IDF + LR entrenado con 25.722 quejas antiguas")
    return _guardar(fig, nombre)


def fig_05_matrices_confusion(cm_a: pd.DataFrame, cm_b: pd.DataFrame, n_a: str, n_b: str,
                              nombre: str = "05_matrices_confusion.png"):
    estilo()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.0))
    etiquetas = [NOMBRES_CORTOS[c] for c in config.CATEGORIAS]
    for ax, cm, titulo in ((axs[0], cm_a, n_a), (axs[1], cm_b, n_b)):
        m = cm.to_numpy()
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(6), etiquetas, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(6), etiquetas, fontsize=8)
        ax.set_xlabel("Mesa predicha")
        ax.set_ylabel("Mesa real (CFPB)")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8.4,
                        color="white" if norm[i, j] > 0.55 else TEXTO)
        _titulo(ax, _corto(titulo), "Conteos sobre el conjunto común · color = % de la fila")
    return _guardar(fig, nombre)


def fig_06_cascada(curva: pd.DataFrame, tabla: pd.DataFrame, frac_llm: float, n_tfidf: str, n_c0: str, n_c1: str,
                   nombre: str = "06_cascada_tfidf_llm.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for llm, color in (("zero-shot", ROJO), ("few-shot", VIOLETA)):
        c = curva[curva["llm"] == llm]
        ax.plot(c["pct_al_llm"], c["f1_macro"] * 100, marker="o", ms=3.5, color=color, label=f"Cascada → LLM {llm}")
    t = tabla.set_index("enfoque")
    ax.axvline(frac_llm * 100, color=GRIS, ls="--", lw=1.2)
    for n, color in ((n_c0, ROJO), (n_c1, VIOLETA)):
        ax.scatter([frac_llm * 100], [t.loc[n, "f1_macro"] * 100], color=color, s=60, zorder=5, edgecolor="white")
    ymin = ax.get_ylim()[0]
    ax.text(frac_llm * 100 + 1.5, ymin + 0.5, f"umbral fijado en validación:\n{num(frac_llm * 100, 1)} % llega al LLM",
            fontsize=7.8, color=TEXTO2, va="bottom")
    ax.set_xlabel("% de quejas que llegan al LLM (las de menor confianza de TF-IDF)")
    ax.set_ylabel("Macro-F1")
    _pct(ax)
    ax.legend(fontsize=8, loc="lower right")
    _titulo(ax, "Cascada barato → LLM", "0 % = solo TF-IDF · 100 % = solo LLM · curva descriptiva sobre el conjunto común")
    return _guardar(fig, nombre)


def fig_07_confiabilidad(confiab: pd.DataFrame, calib: pd.DataFrame, nombre: str = "07_curvas_confiabilidad.png"):
    estilo()
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4), sharey=True)
    grupos = [("TF-IDF + LR", "TF-IDF"), ("bge-m3 + LR", "bge-m3 + LR"), ("LLM (probabilidad por logprobs)", "LLM")]
    colores = [GRIS, AZUL]
    for ax, (titulo, clave) in zip(axs, grupos):
        ax.plot([0, 1], [0, 1], color=GRIS_CLARO, ls="--", lw=1)
        modelos = [m for m in dict.fromkeys(confiab["modelo"]) if clave in m]
        cal = calib[calib["enfoque"].str.contains(clave, regex=False)].reset_index(drop=True)
        for i, m in enumerate(modelos):
            d = confiab[(confiab["modelo"] == m) & (confiab["n"] >= 10)]  # bins con menos de 10 quejas son ruido
            if clave == "LLM":
                color, etiqueta = (ROJO, VIOLETA)[i % 2], _corto(m.split(" · ")[0])
                etiqueta += f" · Brier {num(cal.loc[i, 'brier_sin_calibrar'], 3)}"
            else:
                color, etiqueta = colores[i % 2], m.split(" · ")[-1]
                etiqueta += f" · Brier {num(cal.loc[0, 'brier_sin_calibrar' if i == 0 else 'brier_calibrado'], 3)}"
            ax.plot(d["confianza_media"], d["exactitud"], color=color, label=etiqueta)
            ax.scatter(d["confianza_media"], d["exactitud"], s=12 + d["n"] / 2, color=color, zorder=3)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confianza media del bin")
        ax.legend(fontsize=7.4, loc="upper left")
        sub = f"Temperatura ajustada en validación: T = {num(cal.loc[0, 'temperatura'], 2)}" if clave != "LLM" else "Sin calibrar (no hay LLM en validación) · tamaño = n del bin"
        _titulo(ax, titulo, sub)
    axs[0].set_ylabel("Exactitud observada")
    return _guardar(fig, nombre)


def fig_08_senales_llm(senales: pd.DataFrame, nombre: str = "08_senales_confianza_llm.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    sen = list(dict.fromkeys(senales["senal"]))
    llms = list(dict.fromkeys(senales["llm"]))
    y = np.arange(len(sen))
    for j, (llm, color) in enumerate(zip(llms, (ROJO, VIOLETA))):
        v = senales[senales["llm"] == llm].set_index("senal").loc[sen, "auroc_acierto_vs_error"].to_numpy()
        ax.barh(y + (j - 0.5) * 0.36, v, height=0.36, color=color, label=_corto(llm))
        for i, val in enumerate(v):
            ax.text(val + 0.01, y[i] + (j - 0.5) * 0.36, num(val, 2), va="center", fontsize=8)
    ax.axvline(0.5, color=GRIS, ls="--", lw=1.1)
    ax.invert_yaxis()
    ax.set_yticks(y, sen, fontsize=8.6)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC para separar aciertos de errores del LLM (0,5 = azar)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    _titulo(ax, "¿Qué señal avisa cuándo el LLM se equivoca?", "La confianza que el LLM escribía en el JSON era 0,99 casi siempre")
    return _guardar(fig, nombre)


def fig_09_urgencia(urg: pd.DataFrame, nombre: str = "09_urgencia.png"):
    estilo()
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.0))
    for ax, conj in zip(axs, ("test_reciente", "test_comun")):
        d = urg[urg["conjunto"] == conj].reset_index(drop=True)
        y = np.arange(len(d))
        f1 = d["f1"].to_numpy() * 100
        lo = f1 - d["f1_ic95_bajo"].to_numpy() * 100
        hi = d["f1_ic95_alto"].to_numpy() * 100 - f1
        ax.barh(y, f1, xerr=[lo, hi], color=[GRIS, AZUL, ROJO, VIOLETA][: len(d)], height=0.55, capsize=3, ecolor=TEXTO2)
        ax.set_ylim(3.5, -0.5)  # mismo alto de barra en ambos paneles
        ax.set_yticks(y, [_corto(e) for e in d["enfoque"]], fontsize=8.6)
        _pct(ax, "x")
        ax.set_xlim(0, 115)
        ax.grid(axis="y", visible=False)
        for i, r in d.iterrows():
            ax.text(f1[i] + hi[i] + 1.5, i, f"P {num(r['precision'] * 100, 0)} % · R {num(r['exhaustividad'] * 100, 0)} %", va="center", fontsize=7.8)
        n, nu = int(d["n_quejas"].iloc[0]), int(d["n_urgentes"].iloc[0])
        _titulo(ax, f"F1 de «urgente» · {num(n, 0)} quejas ({nu} urgentes)",
                "Etiqueta: motivo elegido por el consumidor en el formulario del CFPB")
    return _guardar(fig, nombre)


def fig_10_espanol(es: pd.DataFrame, nombre: str = "10_transferencia_espanol.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    y = np.arange(len(es))
    ax.barh(y - 0.18, es["exactitud_ingles"] * 100, height=0.36, color=GRIS, label="Misma queja en inglés")
    v = es["exactitud_espanol"].to_numpy() * 100
    lo = v - es["exactitud_espanol_ic95_bajo"].to_numpy() * 100
    hi = es["exactitud_espanol_ic95_alto"].to_numpy() * 100 - v
    ax.barh(y + 0.18, v, xerr=[lo, hi], height=0.36, color=VERDE, capsize=3, ecolor=TEXTO2, label="Traducida al español (IC 95 %)")
    ax.invert_yaxis()
    ax.set_yticks(y, [_corto(e) for e in es["enfoque"]], fontsize=8.6)
    _pct(ax, "x")
    ax.set_xlim(0, 115)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    _titulo(ax, f"Exactitud en español sin entrenar en español (n = {int(es['n_quejas'].iloc[0])})",
            "Las mismas quejas reales del CFPB: original en inglés vs. traducción")
    return _guardar(fig, nombre)
