"""Figuras del README a partir de reports/*.json|csv (no recalcula nada).

    python -m rag_laboral.graficos
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

# Paleta categórica de referencia (orden fijo) y tintas de texto
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA2, SUPERFICIE, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
NOMBRES = {"bm25": "BM25", "vectorial": "Vectorial (bge-m3)", "hibrido": "Híbrido (RRF)", "rerank": "Híbrido + re-ranking"}
VARIANTES = {"no": "Sin sinónimos", "v1": "Sinónimos v1", "v2": "Sinónimos v2"}


def _estilo(ax, titulo: str):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REJILLA)
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _guardar(fig, nombre: str):
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / nombre, dpi=150)
    plt.close(fig)


def recuperacion():
    t = pd.DataFrame(json.loads((config.REPORTS / "recuperacion.json").read_text(encoding="utf-8"))["resultados"])
    t["sinonimos"] = t["sinonimos"].map(lambda s: "no" if s in (False, "no") else s)
    p = t[t.conjunto == "principal"]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    metodos = list(NOMBRES)
    x = np.arange(len(metodos))
    ancho = 0.26
    for ax, metrica, titulo in ((ejes[0], "recall@5", "Recall@5 (artículo correcto en el top 5)"),
                                (ejes[1], "mrr@20", "MRR@20")):
        for j, v in enumerate(VARIANTES):
            # el re-ranking no se corrió con sinónimos v1 (costo de CPU): esa barra queda vacía
            vals = [next(iter(p[(p.metodo == m) & (p.sinonimos == v)][metrica]), np.nan) for m in metodos]
            barras = ax.bar(x + (j - 1) * ancho, vals, ancho - 0.03, color=SERIES[j], label=VARIANTES[v],
                            edgecolor=SUPERFICIE, linewidth=1)
            if v == "v2":
                for b, val in zip(barras, vals):
                    if np.isnan(val):
                        continue
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=8, color=TINTA)
        ax.set_xticks(x, [NOMBRES[m].replace(" + ", "\n+ ") for m in metodos])
        ax.set_ylim(0, 1.05)
        _estilo(ax, titulo)
    ejes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TINTA2)
    fig.suptitle(f"Recuperación sobre {int(p['n'].iloc[0])} preguntas verificadas a mano (conjunto principal)",
                 x=0.01, ha="left", color=TINTA2, fontsize=9)
    _guardar(fig, "01_recuperacion_por_metodo.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = [1, 5, 10, 20]
    for j, m in enumerate(metodos):
        fila = p[(p.metodo == m) & (p.sinonimos == "v2")].iloc[0]
        ys = [fila[f"recall@{k}"] for k in ks]
        ax.plot(ks, ys, color=SERIES[j], linewidth=2, marker="o", markersize=6, label=NOMBRES[m])
    ax.set_xticks(ks)
    ax.set_xlabel("k (artículos recuperados)", color=TINTA2, fontsize=9)
    ax.set_ylim(0.3, 1.02)
    _estilo(ax, "Recall@k con sinónimos v2")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TINTA2)
    _guardar(fig, "02_recall_en_k.png")


def generacion():
    g = json.loads((config.REPORTS / "generacion.json").read_text(encoding="utf-8"))
    filas = [("cita_exacta", "Cita el artículo correcto"), ("abstencion_indebida", "Se abstiene sin deber (respondibles)"),
             ("rechazo_trampa_cercana", "Rechaza trampas cercanas (IESS, LOSEP…)"),
             ("rechazo_trampa_lejana", "Rechaza trampas lejanas (IVA, RUC…)")]
    conjuntos = [c for c in ("principal", "nuevas") if c in g["por_conjunto"]]
    fig, ejes = plt.subplots(1, len(conjuntos), figsize=(6.2 * len(conjuntos), 3.9), squeeze=False)
    for ax, c in zip(ejes[0], conjuntos):
        y = np.arange(len(filas))
        for j, (clave, nombre) in enumerate((("sin_umbral", "Solo el LLM decide"),
                                             ("con_umbral", f"+ umbral del re-ranker ({g['umbral']['valor']})"))):
            m = g["por_conjunto"][c][clave]
            vals = [m.get(k, np.nan) for k, _ in filas]
            ax.barh(y + (j - 0.5) * 0.36, vals, 0.33, color=SERIES[j], label=nombre, edgecolor=SUPERFICIE)
            for yy, v in zip(y, vals):
                if not np.isnan(v):
                    ax.text(v + 0.01, yy + (j - 0.5) * 0.36, f"{v:.0%}", va="center", fontsize=8, color=TINTA)
        ax.set_yticks(y, [n for _, n in filas])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        _estilo(ax, f"{c.capitalize()}: {g['por_conjunto'][c]['sin_umbral']['respondibles']} preguntas + "
                    f"{g['por_conjunto'][c]['sin_umbral']['trampas']} trampas")
        ax.grid(axis="x", color=REJILLA)
        ax.grid(axis="y", visible=False)
    manejadores, etiquetas = ejes[0][0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, frameon=False, fontsize=8, loc="lower center", ncol=2, labelcolor=TINTA2)
    fig.subplots_adjust(bottom=0.2)
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(config.FIGURAS / "03_generacion_y_rechazo.png", dpi=150)
    plt.close(fig)

    d = pd.read_csv(config.REPORTS / "generacion_detalle.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(d.puntaje_max_rerank.min(), d.puntaje_max_rerank.max(), 25)
    ax.hist(d[d.tipo == "respondible"].puntaje_max_rerank, bins=bins, color=SERIES[0], label="Respondibles",
            edgecolor=SUPERFICIE)
    ax.hist(d[d.tipo != "respondible"].puntaje_max_rerank, bins=bins, color=SERIES[1], label="Trampas",
            edgecolor=SUPERFICIE, alpha=0.9)
    if g["umbral"]["valor"] is not None:
        ax.axvline(g["umbral"]["valor"], color=TINTA2, linestyle="--", linewidth=1.2)
        ax.text(g["umbral"]["valor"], ax.get_ylim()[1] * 0.92, f"  umbral {g['umbral']['valor']}", fontsize=8, color=TINTA2)
    ax.set_xlabel("Mejor puntaje del re-ranker (bge-reranker-v2-m3)", color=TINTA2, fontsize=9)
    ax.set_ylabel("Preguntas", color=TINTA2, fontsize=9)
    _estilo(ax, "¿Separa el re-ranker lo que la ley cubre de lo que no?")
    ax.legend(frameon=False, fontsize=8, labelcolor=TINTA2)
    _guardar(fig, "04_umbral_reranker.png")

    t = d[d.segundos_llm > 0].segundos_llm
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(t, bins=20, color=SERIES[0], edgecolor=SUPERFICIE)
    ax.axvline(t.median(), color=TINTA2, linestyle="--", linewidth=1.2)
    ax.text(t.median(), ax.get_ylim()[1] * 0.9, f"  mediana {t.median():.0f} s", fontsize=8, color=TINTA2)
    ax.set_xlabel("Segundos por respuesta de qwen2.5:7b (CPU Ryzen 5 4600G, sin GPU)", color=TINTA2, fontsize=9)
    _estilo(ax, "Tiempo de generación por pregunta")
    _guardar(fig, "05_tiempo_por_respuesta.png")


if __name__ == "__main__":
    recuperacion()
    generacion()
    print("figuras en", config.FIGURAS.relative_to(config.RAIZ))
