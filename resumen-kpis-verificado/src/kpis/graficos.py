"""Figuras del proyecto (matplotlib, PNG en reports/figures/)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import capa_kpis, config  # noqa: E402
from .generadores import ENFOQUES, NOMBRES  # noqa: E402

# Orden categórico fijo (azul, naranja, aqua, amarillo) y paleta de estado reservada
COLOR_ENFOQUE = dict(zip(ENFOQUES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
COLOR_ESTADO = {"correcta": "#0ca30c", "incorrecta": "#d03b3b", "sin_fuente": "#ec835a"}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2,
                     "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "axes.titleweight": "bold", "axes.titlesize": 11})


def _guardar(fig, nombre: str) -> str:
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = config.FIGURAS / nombre
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"reports/figures/{nombre}"


def ecuador_vs_region() -> str:
    claves = ["crecimiento", "inflacion", "desempleo", "cuenta_corriente", "deuda_externa", "remesas"]
    med = capa_kpis.mediana_regional()
    ecu = capa_kpis.tabla_pais_anio("ECU")
    fig, ejes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for ax, k in zip(ejes.flat, claves):
        ax.plot(ecu.index, ecu[k], color=COLOR_ENFOQUE["libre"], lw=2, label="Ecuador")
        ax.plot(med.index, med[k], color=TINTA_2, lw=1.5, ls="--", label="Mediana de 18 países")
        an = capa_kpis.anomalias("ECU")
        an = an[an["indicador"] == k]
        ax.scatter(an["anio"], an["valor"], s=60, facecolor="none", edgecolor=COLOR_ESTADO["incorrecta"], lw=2,
                   zorder=3, label="Anomalía (|z| ≥ 3)")
        ax.set_title(config.IND_POR_CLAVE[k].nombre.split(" (")[0], loc="left")
        ax.axhline(0, color=TINTA_2, lw=0.6)
    manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Ecuador frente a la mediana regional, 2000–2025 (Banco Mundial, WDI)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, "01_ecuador_vs_region.png")


def mapa_anomalias() -> str:
    a = capa_kpis.anomalias(desde=2005)
    tabla = a.groupby(["iso3", "anio"]).size().unstack(fill_value=0).reindex(list(config.PAISES), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(tabla.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(tabla.columns)), tabla.columns, rotation=90)
    ax.set_yticks(range(len(tabla.index)), [config.PAISES[i] for i in tabla.index])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(tabla.values):
        if v:
            ax.text(j, i, int(v), ha="center", va="center", fontsize=7, color="white" if v >= 3 else TINTA)
    fig.colorbar(im, ax=ax, label="Indicadores con cambio anómalo")
    ax.set_title("Cambios interanuales fuera de lo normal (|z robusto| ≥ 3 frente a la historia del país;\n"
                 "empieza en 2009 porque cada país necesita 8 años previos de variaciones)", loc="left")
    return _guardar(fig, "02_mapa_anomalias.png")


def _barras_ic(ax, df: pd.DataFrame, metrica: str, etiqueta_pct: bool = True) -> None:
    conjuntos = ["validacion", "prueba"]
    ancho = 0.2
    for k, enf in enumerate(ENFOQUES):
        for j, conj in enumerate(conjuntos):
            fila = df[(df["enfoque"] == enf) & (df["conjunto"] == conj)]
            if fila.empty:
                continue
            f = fila.iloc[0]
            x = j + (k - 1.5) * ancho
            v = f[metrica]
            lo, hi = f[f"{metrica}_ic"]
            ax.bar(x, v, ancho * 0.9, color=COLOR_ENFOQUE[enf], label=NOMBRES[enf] if j == 1 else None)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=TINTA, capsize=3, lw=1)
            ax.text(x, hi, f"{v:.1%}" if etiqueta_pct else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=TINTA)
    ax.set_xticks([0, 1], ["Validación (5 informes)", "Prueba (12 informes)"])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def alucinacion_y_cobertura(res: pd.DataFrame) -> str:
    df = res[res["version"].isin(["-", res.attrs.get("version_elegida", "v2")])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    _barras_ic(a1, df, "tasa_alucinacion")
    a1.set_title("Cifras alucinadas (incorrectas + sin fuente), IC 95 %", loc="left")
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _barras_ic(a2, df, "cobertura")
    a2.set_title("Cobertura de hallazgos de referencia, IC 95 %", loc="left")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylim(0, 1.15)
    manejadores, etiquetas = a2.get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               fontsize=9)
    fig.tight_layout()
    return _guardar(fig, "03_alucinacion_y_cobertura.png")


def tipos_de_error(cifras: pd.DataFrame, comps: pd.DataFrame) -> str:
    """Izquierda: cifras alucinadas (pocas). Derecha: lecturas cualitativas al revés (muchas más)."""
    c = cifras[cifras["conjunto"] == "prueba"]
    tabla = c.groupby(["enfoque", "estado"]).size().unstack(fill_value=0).reindex(ENFOQUES).fillna(0)
    k = comps[comps["conjunto"] == "prueba"].assign(mal=lambda d: ~d["correcta"].astype(bool))
    tk = k.groupby("enfoque").agg(n=("mal", "size"), mal=("mal", "sum")).reindex(ENFOQUES).fillna(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(ENFOQUES))
    izq = np.zeros(len(ENFOQUES))
    for est, etiqueta in [("incorrecta", "✗ incorrecta"), ("sin_fuente", "? sin fuente")]:
        v = tabla.get(est, pd.Series(0, index=tabla.index)).to_numpy()
        a1.barh(y, v, left=izq, color=COLOR_ESTADO[est], edgecolor="#fcfcfb", lw=2, label=etiqueta)
        izq += v
    tot = tabla.sum(axis=1).to_numpy()
    for i, (m, t) in enumerate(zip(izq, tot)):
        a1.text(m + 0.15, i, f"{int(m)} de {int(t)} cifras", va="center", fontsize=8, color=TINTA)
    a1.set_xlim(0, max(izq.max(), 1) * 2.2)
    a1.set_title("Cifras alucinadas", loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    a1.set_yticks(y, [NOMBRES[e] for e in ENFOQUES])
    a1.invert_yaxis()
    a2.barh(y, tk["n"] - tk["mal"], color=COLOR_ESTADO["correcta"], edgecolor="#fcfcfb", lw=2,
            label="✓ dirección correcta")
    a2.barh(y, tk["mal"], left=tk["n"] - tk["mal"], color=COLOR_ESTADO["incorrecta"], edgecolor="#fcfcfb", lw=2,
            label="✗ al revés o PIB en USD como crecimiento")
    for i, (n, m) in enumerate(zip(tk["n"], tk["mal"])):
        a2.text(n + 0.4, i, f"{int(m)} de {int(n)}", va="center", fontsize=8, color=TINTA)
    a2.set_xlim(0, tk["n"].max() * 1.25)
    a2.set_title("Lecturas cualitativas revisadas (frente a la región, al año anterior, PIB)", loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("Prueba: el LLM casi no inventa cifras, pero sí lecturas engañosas", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _guardar(fig, "04_cifras_y_lecturas_enganosas.png")


def reintentos(intentos: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for conj, estilo in [("validacion", "--"), ("prueba", "-")]:
        d = intentos[intentos["conjunto"] == conj]
        if d.empty:
            continue
        g = d.groupby("intento").agg(err=("errores", "sum"), n=("n_cifras", "sum"), inf=("pais", "count"))
        ax.plot(g.index, g["err"] / g["n"], estilo, marker="o", ms=8, color=COLOR_ENFOQUE["verificado"], lw=2,
                label=f"{'Validación' if conj == 'validacion' else 'Prueba'}")
        for x, (e, n, k) in g.iterrows():
            ax.annotate(f"{int(k)} informe{'s' if k != 1 else ''}", (x, e / n), textcoords="offset points", xytext=(6, 6), fontsize=8,
                        color=TINTA_2)
    ax.set_xticks(range(config.MAX_REINTENTOS + 1), ["1.ª respuesta", "reintento 1", "reintento 2"][
        :config.MAX_REINTENTOS + 1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Cifras alucinadas en los informes que llegan a ese intento")
    ax.legend(frameon=False)
    ax.set_title("(c) Verificación y reintento: error por intento", loc="left")
    return _guardar(fig, "05_reintentos.png")


def mejora_prompt(res: pd.DataFrame) -> str:
    d = res[(res["conjunto"] == "validacion") & (res["enfoque"].isin(["estructurado", "verificado"]))]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ancho = 0.35
    for k, ver in enumerate(["v1", "v2"]):
        sub = d[d["version"] == ver].set_index("enfoque").reindex(["estructurado", "verificado"])
        x = np.arange(2) + (k - 0.5) * ancho
        ax.bar(x, sub["tasa_alucinacion"], ancho * 0.9, color=["#9ec5f4", "#2a78d6"][k], label=f"prompt {ver}")
        for xi, v in zip(x, sub["tasa_alucinacion"]):
            ax.text(xi, v + 0.005, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(2), [NOMBRES["estructurado"], NOMBRES["verificado"]])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Ajuste del prompt en validación: cifras alucinadas", loc="left")
    ax.legend(frameon=False)
    return _guardar(fig, "06_mejora_prompt_validacion.png")


def legibilidad(rub: pd.DataFrame) -> str:
    criterios = ["claridad", "concision", "utilidad", "tono"]
    g = rub.groupby("enfoque")[criterios].mean().reindex(ENFOQUES)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ancho = 0.2
    for k, enf in enumerate(ENFOQUES):
        x = np.arange(len(criterios)) + (k - 1.5) * ancho
        ax.bar(x, g.loc[enf], ancho * 0.9, color=COLOR_ENFOQUE[enf], label=NOMBRES[enf])
    ax.set_xticks(range(len(criterios)), ["Claridad", "Concisión", "Utilidad para decidir", "Tono ejecutivo"])
    ax.set_ylim(0, 5.4)
    ax.set_ylabel("Puntaje medio (1–5)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Legibilidad juzgada a mano (rúbrica a ciegas, 6 informes de prueba)", loc="left")
    return _guardar(fig, "07_legibilidad_rubrica.png")


def tiempo(detalle: pd.DataFrame) -> str:
    d = detalle[detalle["conjunto"] == "prueba"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, enf in enumerate(ENFOQUES):
        v = d[d["enfoque"] == enf]["segundos"].to_numpy()
        ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=40, color=COLOR_ENFOQUE[enf],
                   edgecolor="#fcfcfb", lw=1.5, zorder=3)
        ax.hlines(np.mean(v), k - 0.25, k + 0.25, color=TINTA, lw=2)
        ax.text(k + 0.28, np.mean(v), f"{np.mean(v):.0f} s", va="center", fontsize=8)
    ax.set_xticks(range(len(ENFOQUES)), [NOMBRES[e].replace(") ", ")\n", 1).replace(" + ", "\n+ ")
                                         for e in ENFOQUES], fontsize=8)
    ax.set_ylabel("Segundos por informe (CPU)")
    ax.set_title("Tiempo por informe en prueba (línea = media)", loc="left")
    return _guardar(fig, "08_tiempo_por_informe.png")
