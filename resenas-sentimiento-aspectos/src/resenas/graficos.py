"""Figuras del proyecto (matplotlib). Paleta de referencia validada para daltonismo: los tres enfoques usan los tres
primeros colores categóricos; positivo/negativo usan el par divergente azul–rojo con gris neutro."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
ENFOQUES = {"clasico": ("Clásico (reglas + TF-IDF)", "#2a78d6"),
            "embeddings": ("Embeddings bge-m3 + LR", "#eb6834"),
            "llm": ("LLM qwen2.5:7b", "#1baf7a")}
POSITIVO, NEGATIVO, NEUTRO = "#2a78d6", "#e34948", "#c3c2b7"
NOMBRE_ASPECTO = {"bateria": "Batería", "carga": "Carga", "pantalla": "Pantalla", "sonido": "Sonido",
                  "calidad": "Calidad / durabilidad", "ajuste": "Ajuste / compatibilidad", "diseno": "Diseño",
                  "proteccion": "Protección", "precio": "Precio", "envio": "Envío / estado al llegar",
                  "atencion": "Atención / garantía", "facilidad_uso": "Facilidad de uso"}


def _estilo(ax, eje_y=True):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9, length=0)
    ax.grid(axis="y" if eje_y else "x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _figura(ancho=9, alto=4.8):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    _estilo(ax)
    return fig, ax


def _guardar(fig, nombre):
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / nombre, facecolor=SUPERFICIE)
    plt.close(fig)


# --------------------------------------------------------------------------- evaluación
def figuras_evaluacion(ev: dict) -> None:
    res = ev["enfoques"]
    metricas = [("f1_aspectos", "F1 detección\nde aspectos"), ("f1_aspecto_polaridad", "F1 aspecto\n+ polaridad"),
                ("exactitud_polaridad", "Exactitud de polaridad\n(aspectos acertados)"),
                ("exactitud_global", "Exactitud del\nsentimiento global")]
    fig, ax = _figura(10, 5)
    x = np.arange(len(metricas))
    ancho = 0.26
    for k, (clave, (nombre, color)) in enumerate(ENFOQUES.items()):
        vals = [res[clave][m] for m, _ in metricas]
        ic = [res[clave]["ic95"][m] for m, _ in metricas]
        err = np.array([[v - lo for v, (lo, _) in zip(vals, ic)], [hi - v for v, (_, hi) in zip(vals, ic)]])
        pos = x + (k - 1) * (ancho + 0.02)
        ax.bar(pos, vals, ancho, color=color, label=nombre, edgecolor=SUPERFICIE, linewidth=1)
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=TINTA_2, elinewidth=1, capsize=3)
        for p, v in zip(pos, vals):
            ax.text(p, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=SUPERFICIE, fontweight="bold")
    ax.set_xticks(x, [m for _, m in metricas], fontsize=9, color=TINTA)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Los tres enfoques en el conjunto test ({ev['n_test']} reseñas etiquetadas a mano) · IC 95 %",
                 loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.12))
    _guardar(fig, "01_comparacion_enfoques.png")

    soporte = ev["soporte_test"]
    orden = sorted(config.ASPECTOS, key=lambda a: soporte[a])
    fig, ax = _figura(9, 6)
    _estilo(ax, eje_y=False)
    y = np.arange(len(orden))
    for k, (clave, (nombre, color)) in enumerate(ENFOQUES.items()):
        f1 = [res[clave]["por_aspecto"][a]["f1"] for a in orden]
        ax.scatter(f1, y + (k - 1) * 0.22, s=48, color=color, label=nombre, edgecolor=SUPERFICIE, linewidth=1.5,
                   zorder=3)
    ax.set_yticks(y, [f"{NOMBRE_ASPECTO[a]} (n={soporte[a]})" for a in orden], fontsize=9, color=TINTA)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("F1 de detección del aspecto", color=TINTA_2, fontsize=9)
    ax.set_title("F1 por aspecto en test (n = menciones en la verdad)", loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, -0.2), ncol=3)
    _guardar(fig, "02_f1_por_aspecto.png")

    fig, ax = _figura(8, 4.6)
    _estilo(ax, eje_y=True)
    desplaz = {"clasico": (10, -8, "top"), "embeddings": (-10, 14, "bottom"), "llm": (10, -8, "top")}
    for clave, (nombre, color) in ENFOQUES.items():
        s = res[clave]["costo"]["segundos_por_resena"]
        f = res[clave]["f1_aspecto_polaridad"]
        ax.scatter([s], [f], s=110, color=color, edgecolor=SUPERFICIE, linewidth=2, zorder=3)
        total = s * 55_082
        tiempo_total = f"{total / 60:.0f} min" if total < 3600 else f"{total / 3600:,.0f} h".replace(",", " ")
        txt = f"{nombre}\n{_fmt_seg(s)} por reseña · {tiempo_total} para las 55 mil"
        dx, dy, va = desplaz[clave]
        ax.annotate(txt, (s, f), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=TINTA, va=va,
                    ha="right" if dx < 0 else "left")
    s_cla, s_emb, s_llm = (res[k]["costo"]["segundos_por_resena"] for k in ("clasico", "embeddings", "llm"))
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segundos por reseña en CPU (escala log)", color=TINTA_2, fontsize=9)
    ax.set_ylabel("F1 aspecto + polaridad", color=TINTA_2, fontsize=9)
    ax.set_title(f"Calidad vs costo: embeddings {s_emb / s_cla:.0f}× más lento que el clásico; "
                 f"LLM {s_llm / s_emb:.0f}× más que embeddings", loc="left", fontsize=10.5, color=TINTA)
    _guardar(fig, "03_costo_vs_calidad.png")


def _fmt_seg(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 0.1 else f"{s:.2g} s" if s < 10 else f"{s:.1f} s"


# --------------------------------------------------------------------------- hallazgos
def figuras_hallazgos(t: dict[str, pd.DataFrame]) -> None:
    """t: tablas de hallazgos.py (ya agregadas)."""
    # 04 · Menciones por aspecto, divididas en positivas y negativas (barras divergentes)
    a = t["aspectos"].sort_values("menciones")
    fig, ax = _figura(9, 5.4)
    _estilo(ax, eje_y=False)
    y = np.arange(len(a))
    ax.barh(y, -a["pct_neg"], color=NEGATIVO, label="% de reseñas con queja", height=0.62)
    ax.barh(y, a["pct_pos"], color=POSITIVO, label="% de reseñas con elogio", height=0.62)
    for yi, (n, p) in enumerate(zip(a["pct_neg"], a["pct_pos"])):
        ax.text(-n - 0.4, yi, f"{n:.1f}", ha="right", va="center", fontsize=8, color=TINTA_2)
        ax.text(p + 0.4, yi, f"{p:.1f}", ha="left", va="center", fontsize=8, color=TINTA_2)
    ax.axvline(0, color=TINTA_2, linewidth=0.8)
    ax.set_yticks(y, [NOMBRE_ASPECTO[x] for x in a.index], fontsize=9, color=TINTA)
    lim = max(a["pct_neg"].max(), a["pct_pos"].max()) * 1.25
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
    ax.set_title("De qué hablan las reseñas: % de reseñas que se quejan o elogian cada aspecto", loc="left",
                 fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _guardar(fig, "04_aspectos_quejas_elogios.png")

    # 05 · Mapa de calor: % de reseñas con queja por tipo de producto × aspecto
    h = t["tipo_aspecto"]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    rampa = matplotlib.colors.LinearSegmentedColormap.from_list("rojos", ["#fcfcfb", "#f4b4b0", "#e34948", "#9c1f20"])
    im = ax.imshow(h.values, cmap=rampa, aspect="auto", vmin=0)
    ax.set_xticks(range(h.shape[1]), [NOMBRE_ASPECTO[c].split(" /")[0] for c in h.columns], rotation=35, ha="right",
                  fontsize=9, color=TINTA)
    ax.set_yticks(range(h.shape[0]), h.index, fontsize=9, color=TINTA)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=SUPERFICIE if v > h.values.max() * 0.55 else TINTA)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=TINTA_2)
    ax.set_title("% de reseñas con queja sobre cada aspecto, por tipo de producto", loc="left", fontsize=11, color=TINTA)
    _guardar(fig, "05_quejas_tipo_aspecto.png")

    # 06 · Relación con las estrellas: % de reseñas con queja por aspecto según estrellas (top 5 aspectos)
    e = t["estrellas_aspecto"]
    top = e.max().sort_values(ascending=False).index[:5]
    colores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = _figura(9, 4.8)
    for a_, c in zip(top, colores):
        ax.plot(e.index, e[a_], color=c, linewidth=2, marker="o", markersize=7, markeredgecolor=SUPERFICIE,
                markeredgewidth=1.5, label=NOMBRE_ASPECTO[a_])
    ax.set_xticks(e.index, [f"{i}★" for i in e.index])
    ax.set_xlim(0.8, 5.2)
    ax.set_ylabel("% de reseñas con queja", color=TINTA_2, fontsize=9)
    ax.set_title("Las quejas por aspecto caen con las estrellas, pero no todas igual", loc="left", fontsize=11,
                 color=TINTA)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _guardar(fig, "06_quejas_vs_estrellas.png")

    # 07 · Evolución anual del % de reseñas con queja (aspectos principales)
    ev = t["anio_aspecto_intra_producto"]
    top = ev.mean().sort_values(ascending=False).index[:4]
    fig, ax = _figura(9, 4.6)
    for a_, c in zip(top, colores):
        ax.plot(ev.index, ev[a_], color=c, linewidth=2, label=NOMBRE_ASPECTO[a_])
        ax.annotate(NOMBRE_ASPECTO[a_].split(" /")[0], (ev.index[-1], ev[a_].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TINTA)
    ax.set_xlim(ev.index.min() - 0.2, ev.index.max() + 1.0)
    ax.set_xticks(ev.index)
    ax.set_ylabel("% de reseñas con queja", color=TINTA_2, fontsize=9)
    ax.set_title("Quejas por año, dentro de cada producto (años con ≥ 1000 reseñas)", loc="left", fontsize=11,
                 color=TINTA)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _guardar(fig, "07_quejas_por_anio.png")

    # 08 · Marcas: índice neto (elogios − quejas) de calidad/durabilidad, marcas con ≥ 600 reseñas
    m = t["marcas"].sort_values("neto_calidad")
    fig, ax = _figura(9, 5.6)
    _estilo(ax, eje_y=False)
    y = np.arange(len(m))
    colores_m = [NEGATIVO if v < 0 else POSITIVO for v in m["neto_calidad"]]
    ax.barh(y, m["neto_calidad"], color=colores_m, height=0.62)
    for yi, v in enumerate(m["neto_calidad"]):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}", ha="left" if v >= 0 else "right", va="center",
                fontsize=8, color=TINTA_2)
    ax.axvline(0, color=TINTA_2, linewidth=0.8)
    ax.set_xlim(min(m["neto_calidad"].min() - 2.5, -1), m["neto_calidad"].max() + 2.5)
    ax.set_yticks(y, [f"{i} ({n:,})".replace(",", " ") for i, n in zip(m.index, m["resenas"])], fontsize=8.5,
                  color=TINTA)
    ax.set_xlabel("Puntos: % de reseñas que elogian − % que se quejan de la calidad/durabilidad", color=TINTA_2,
                  fontsize=9)
    ax.set_title("Calidad percibida por marca (marcas con ≥ 600 reseñas en la muestra)", loc="left", fontsize=11,
                 color=TINTA)
    _guardar(fig, "08_marcas_calidad.png")
