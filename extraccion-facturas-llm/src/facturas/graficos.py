"""Figuras de la evaluación (matplotlib, PNG estático en reports/figures)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .esquema import CAMPOS  # noqa: E402

SUPERFICIE, TINTA, TINTA_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORES = {"regex": "#eb6834", "llm": "#2a78d6", "cascada": "#1baf7a", "llm_1": "#9cc1ee", "llm_rep": "#6250d6"}
NOMBRES = {"regex": "Regex (línea base)", "llm_1": "LLM, 1.er intento", "llm": "LLM + validación + reintento",
           "cascada": "Cascada regex → LLM (+ reparaciones)", "llm_rep": "LLM + reparaciones"}
ETIQ_CAMPOS = {"ruc_emisor": "RUC emisor", "razon_social_emisor": "Razón social", "numero_factura": "N.º factura",
               "numero_autorizacion": "Autorización (49 díg.)", "fecha_emision": "Fecha", "cliente_nombre": "Cliente",
               "cliente_identificacion": "CI/RUC cliente", "subtotal": "Subtotal", "descuento": "Descuento",
               "iva": "IVA", "total": "Total"}


def _estilo(ax, titulo: str):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", fontsize=12, color=TINTA, pad=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.set_axisbelow(True)


def _barras_h(ax, etiquetas, series: dict[str, list[float]], titulo: str):
    n = len(series)
    alto = 0.8 / n
    y = np.arange(len(etiquetas))
    for k, (m, vals) in enumerate(series.items()):
        pos = y + (k - (n - 1) / 2) * alto
        ax.barh(pos, vals, height=alto * 0.9, color=COLORES[m], label=NOMBRES[m], edgecolor=SUPERFICIE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(min(v, 1) + 0.01, p, f"{v:.0%}", va="center", fontsize=7.5, color=TINTA_2)
    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    _estilo(ax, titulo)


def graficar(inf: dict) -> None:
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    m50 = inf["muestra_50"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SUPERFICIE})

    # 1 · exactitud por campo (muestra de 50: mismas facturas para todos)
    fig, ax = plt.subplots(figsize=(9, 7))
    etiquetas = [ETIQ_CAMPOS[c] for c in CAMPOS] + ["Ítems (F1)"]
    series = {m: [m50[m]["global"]["por_campo"][c] for c in CAMPOS] + [m50[m]["global"]["items_f1"]]
              for m in ("regex", "llm", "cascada")}
    _barras_h(ax, etiquetas, series, f"Exactitud por campo · muestra de {inf['datos']['muestra_llm']} facturas")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.14), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / "01_exactitud_por_campo.png", dpi=130)
    plt.close(fig)

    # 2 · facturas perfectas por plantilla y por condición
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [5, 2]})
    plantillas = list(m50["llm"]["por_plantilla"])
    series = {m: [m50[m]["por_plantilla"][p]["facturas_perfectas"] for p in plantillas]
              for m in ("regex", "llm", "cascada")}
    etq = [p.replace("_", " ").replace("E compacta", "E compacta\n(nunca vista)") for p in plantillas]
    _barras_h(axs[0], etq, series, "Facturas 100 % correctas, por plantilla")
    conds = list(m50["llm"]["por_condicion"])
    series = {m: [m50[m]["por_condicion"][c]["facturas_perfectas"] for c in conds]
              for m in ("regex", "llm", "cascada")}
    _barras_h(axs[1], conds, series, "…por condición")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.7, -0.16), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / "02_por_plantilla_y_condicion.png", dpi=130)
    plt.close(fig)

    # 3 · validación como semáforo: 1.er intento vs tras reintento
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cats = [("aprobada_correcta", "Aprobada y correcta", "#1baf7a"),
            ("marcada_con_error", "Marcada para revisión (con error)", "#eda100"),
            ("marcada_correcta", "Marcada sin necesidad", "#c3c2b7"),
            ("aprobada_con_error", "Error silencioso (aprobada con error)", "#e34948")]
    filas = [("llm", "LLM (medido)"), ("llm_rep", "LLM + reparaciones")]
    for k, (m, nombre) in enumerate(filas):
        v = inf["validacion"][m]
        tot = sum(v.get(c, 0) for c, _, _ in cats)
        izq = 0
        for c, lab, col in cats:
            w = v.get(c, 0) / tot
            ax.barh(k, w, left=izq, color=col, edgecolor=SUPERFICIE, linewidth=2, height=0.6,
                    label=lab if k == 0 else None)
            if w >= 0.03:
                ax.text(izq + w / 2, k, f"{v.get(c, 0)}", ha="center", va="center", fontsize=9, color=TINTA)
            izq += w
    ax.set_yticks(range(len(filas)), [n for _, n in filas])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _estilo(ax, "Validaciones de negocio como semáforo (n.º de facturas)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.55), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(config.FIGURAS / "03_validacion_semaforo.png", dpi=130)
    plt.close(fig)
