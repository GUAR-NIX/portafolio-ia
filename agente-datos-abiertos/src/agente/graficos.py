"""Generación de las 7 figuras analíticas (300 DPI) en reports/figures/ para IA_06_agente_datos_abiertos."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, datos, evaluacion  # noqa: E402

AZUL, NARANJA, VERDE, AMARILLO, ROSA, VIOLETA, VERDE_OSC, ROJO = config.COLORES
GRIS, GRIS_CLARO = "#8a8984", "#c9c8c3"
TEXTO, TEXTO2, SUP = "#0b0b0b", "#52514e", "#fcfcfb"

def num(v: float, dec: int = 1) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUP,
        "axes.facecolor": SUP,
        "savefig.facecolor": SUP,
        "savefig.dpi": 300,
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
        "axes.titlesize": 11.6,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 24,
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
        config.FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURAS_DIR / nombre, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def _pct(ax, eje: str = "y", dec: int = 0, escala: float = 100.0) -> None:
    f = mtick.FuncFormatter(lambda v, _: f"{num(v * escala, dec)} %")
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def fig_01_cobertura_datos_banco_mundial_y_clima(
    nombre: str = "01_cobertura_datos_banco_mundial_y_clima.png",
):
    """Panorama de los datos reales descargados de Banco Mundial (12 países) y Open-Meteo (12 ciudades)."""
    estilo()
    bm = datos.cargar_instantanea_bm()
    df_clima = datos.construir_dataframe_clima_anual()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.6))

    # Panel izquierdo: PIB per cápita 2023 (miles USD) en los 11 países de LatAm
    paises_latam = [p for p in config.PAISES.keys() if p != "USA"]
    valores_pib = []
    for p in paises_latam:
        serie_pib = bm["series"][p]["pib_per_capita_usd"]
        val_2023 = serie_pib.get("2023") or serie_pib.get("2022") or 0.0
        valores_pib.append((config.PAISES[p]["nombre"], p, float(val_2023) / 1000.0))
    valores_pib.sort(key=lambda x: x[2], reverse=True)

    y1 = np.arange(len(valores_pib))
    etiquetas_p = [f"{nom} ({iso})" for nom, iso, _ in valores_pib]
    cifras_k = [v for _, _, v in valores_pib]
    a1.barh(y1, cifras_k, color=AZUL, height=0.62)
    a1.invert_yaxis()
    a1.set_yticks(y1, etiquetas_p, fontsize=8.8)
    a1.set_xlabel("PIB per cápita 2023 (miles de USD corrientes, Banco Mundial)")
    a1.set_xlim(0, max(cifras_k) * 1.24)
    a1.grid(axis="y", visible=False)
    for i, v in enumerate(cifras_k):
        a1.text(v + 0.3, i, f"{num(v * 1000, 0)} USD", va="center", fontsize=8.2, color=TEXTO)
    _titulo(
        a1,
        "Banco Mundial WDI: PIB per cápita en América Latina (2023)",
        "Referencia EE. UU. (USA): 82.769 USD · 3.178 observaciones anuales (1990-2024)",
    )

    # Panel derecho: Clima 2020-2024 en las 12 ciudades (Precipitación media anual mm y Temp media °C)
    resumen_c = (
        df_clima.groupby("ciudad", as_index=False)
        .agg(
            precip_media=("precipitacion_mm", "mean"),
            temp_media=("temp_media_c", "mean"),
        )
        .sort_values("precip_media", ascending=False)
    )
    y2 = np.arange(len(resumen_c))
    precip_vals = resumen_c["precip_media"].to_numpy()
    temp_vals = resumen_c["temp_media"].to_numpy()
    ciudades_lbl = resumen_c["ciudad"].tolist()

    a2.barh(y2, precip_vals, color=VERDE, height=0.62)
    a2.invert_yaxis()
    a2.set_yticks(y2, ciudades_lbl, fontsize=8.8)
    a2.set_xlabel("Precipitación media anual 2020-2024 (mm/año, Open-Meteo ERA5)")
    a2.set_xlim(0, max(precip_vals) * 1.28)
    a2.grid(axis="y", visible=False)
    for i, (pr, tm) in enumerate(zip(precip_vals, temp_vals)):
        a2.text(
            pr + 25,
            i,
            f"{num(pr, 0)} mm · {num(tm, 1)} °C",
            va="center",
            fontsize=8.1,
            color=TEXTO,
        )
    _titulo(
        a2,
        "Open-Meteo Archive: lluvia anual y temperatura media (2020-2024)",
        "Agregación sobre 21.924 registros diarios reales en 12 ciudades",
    )

    return _guardar(fig, nombre)


def fig_07_cruce_real_clima_economia_latam(
    nombre: str = "07_cruce_real_clima_economia_latam.png",
):
    """Análisis empírico cruzando datos reales del Banco Mundial y Open-Meteo en América Latina."""
    estilo()
    bm = datos.cargar_instantanea_bm()
    df_clima = datos.construir_dataframe_clima_anual()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.5))

    # Panel izquierdo: Evolución histórica del PIB per cápita (1995-2023) en 6 economías clave
    paises_serie = [
        ("CHL", "Chile", AZUL),
        ("URY", "Uruguay", VERDE),
        ("CRI", "Costa Rica", NARANJA),
        ("MEX", "México", VIOLETA),
        ("COL", "Colombia", ROSA),
        ("ECU", "Ecuador", AMARILLO),
    ]
    for iso, nom, col in paises_serie:
        serie = bm["series"][iso]["pib_per_capita_usd"]
        anios_s = sorted(int(a) for a in serie.keys() if 1995 <= int(a) <= 2023)
        vals_s = [float(serie[str(a)]) / 1000.0 for a in anios_s]
        a1.plot(anios_s, vals_s, color=col, label=f"{nom} ({iso})", linewidth=2.1)

    a1.set_xlabel("Año (Banco Mundial WDI · NY.GDP.PCAP.CD)")
    a1.set_ylabel("PIB per cápita (miles de USD corrientes)")
    a1.legend(ncol=2, loc="upper left", fontsize=8.3)
    _titulo(
        a1,
        "Trayectoria histórica del PIB per cápita (1995-2023)",
        "Series anuales oficiales consultadas por consultar_indicador_banco_mundial",
    )

    # Panel derecho: Cruce Energía Renovable (% BM 2021) vs Precipitación anual en capitales (Open-Meteo)
    clima_cap = df_clima.groupby(["codigo_pais", "ciudad"], as_index=False).agg(
        precip_mm=("precipitacion_mm", "mean"),
        rad_solar=("radiacion_solar_mj_m2", "mean"),
    )
    clima_cap_unicos = clima_cap.drop_duplicates(subset=["codigo_pais"], keep="first")

    xs, ys, tamanhos, etiquetas_pt = [], [], [], []
    for _, fila in clima_cap_unicos.iterrows():
        iso = fila["codigo_pais"]
        serie_ren = bm["series"].get(iso, {}).get("energias_renovables_pct", {})
        val_ren = serie_ren.get("2021") or serie_ren.get("2020")
        if val_ren is not None:
            xs.append(float(fila["precip_mm"]))
            ys.append(float(val_ren))
            tamanhos.append(float(fila["rad_solar"]) * 8.5)
            etiquetas_pt.append(f"{iso} ({fila['ciudad']})")

    a2.scatter(xs, ys, s=tamanhos, color=VERDE, alpha=0.82, edgecolors=TEXTO, linewidth=1.0)
    for x_v, y_v, lbl in zip(xs, ys, etiquetas_pt):
        a2.annotate(
            lbl,
            xy=(x_v, y_v),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.0,
            color=TEXTO,
        )

    a2.set_xlabel("Precipitación media anual en la ciudad principal (mm/año, Open-Meteo)")
    a2.set_ylabel("Participación de energía renovable (%, Banco Mundial)")
    a2.set_xlim(0, max(xs) * 1.22)
    a2.set_ylim(0, max(ys) * 1.18)
    _titulo(
        a2,
        "Cruce Multi-API: régimen hídrico/solar vs. matriz renovable",
        "Tamaño del círculo proporcional a la radiación solar media (MJ/m²)",
    )

    return _guardar(fig, nombre)


def _fila(df_resumen: pd.DataFrame, conjunto: str, clave: str) -> pd.Series | None:
    sub = df_resumen[(df_resumen["conjunto"] == conjunto) & (df_resumen["clave_enfoque"] == clave)]
    return sub.iloc[0] if len(sub) else None


def _clave_agente_validacion(df_resumen: pd.DataFrame) -> str:
    return "agente_" + str(_fila(df_resumen, "prueba", "agente")["version_prompt"])


def fig_02_validacion_vs_prueba(df_resumen: pd.DataFrame, nombre: str = "02_validacion_vs_prueba.png"):
    """Respuesta correcta por enfoque en validación (ajuste) y prueba (no vista), con IC 95 %."""
    estilo()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    claves = [("zeroshot_sin_tools", "zeroshot_sin_tools"), ("enrutador_reglas", "enrutador_reglas"),
              ("react_texto", "react_texto"), (_clave_agente_validacion(df_resumen), "agente")]
    x = np.arange(len(claves))
    for desplaz, conjunto, color, etiqueta in ((-0.2, "validacion", GRIS_CLARO, "Validación (44, usada para ajustar)"),
                                               (0.2, "prueba", AZUL, "Prueba (60, no vista)")):
        vals, err_inf, err_sup = [], [], []
        for c_val, c_pru in claves:
            f = _fila(df_resumen, conjunto, c_val if conjunto == "validacion" else c_pru)
            v = float(f["respuesta_correcta"])
            vals.append(v)
            err_inf.append(v - float(f["respuesta_correcta_ic95_inf"]))
            err_sup.append(float(f["respuesta_correcta_ic95_sup"]) - v)
        ax.bar(x + desplaz, vals, width=0.38, color=color, label=etiqueta,
               yerr=[err_inf, err_sup], capsize=3, error_kw={"ecolor": TEXTO2, "elinewidth": 1})
        for xi, v, e_sup in zip(x + desplaz, vals, err_sup):
            ax.text(xi, v + e_sup + 0.025, f"{num(v * 100, 1)} %", ha="center", fontsize=8.4, color=TEXTO)
    ax.set_xticks(x, ["1 · LLM sin\nherramientas", "2 · Reglas\n(sin LLM)", "3 · ReAct en\ntexto", "4 · Agente\ntool-calling"])
    ax.set_ylim(0, 1.22)
    _pct(ax)
    ax.set_ylabel("Respuesta correcta (dominio + abstención)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.6)
    _titulo(ax, "Validación vs. prueba no vista: respuesta correcta por enfoque",
            "Barras de error: IC 95 % bootstrap (Wilson si todas las respuestas son iguales)")
    return _guardar(fig, nombre)


def fig_03_atribucion_llm_vs_codigo(df_resumen: pd.DataFrame, nombre: str = "03_atribucion_llm_vs_codigo.png"):
    """Qué hace bien el LLM por sí solo y qué aporta el código determinista (agente, ambos conjuntos)."""
    estilo()
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    metricas = [
        ("llm_seleccion_turno1", "LLM: herramientas\nexactas (turno 1)"),
        ("llm_seleccion_exacta", "LLM: herramientas\nexactas (bucle)"),
        ("llm_args_perfectos", "LLM: herramientas\n+ argumentos exactos"),
        ("llm_se_abstuvo_ood", "LLM: se abstiene\nsolo (fuera dominio)"),
        ("abstencion_correcta_ood", "Código: abstención\nfinal (fuera dominio)"),
        ("fidelidad_texto_llm", "LLM: cifras correctas\nen su propio texto"),
        ("respuesta_correcta", "Sistema: respuesta\ncorrecta"),
    ]
    x = np.arange(len(metricas))
    for desplaz, conjunto, clave, color, etiqueta in (
        (-0.2, "validacion", _clave_agente_validacion(df_resumen), GRIS_CLARO, "Validación"),
        (0.2, "prueba", "agente", AZUL, "Prueba"),
    ):
        f = _fila(df_resumen, conjunto, clave)
        vals = [float(f[m]) if f[m] is not None and not pd.isna(f[m]) else 0.0 for m, _ in metricas]
        colores = [color if not m.startswith("abstencion") else (VERDE if conjunto == "prueba" else GRIS_CLARO)
                   for m, _ in metricas]
        ax.bar(x + desplaz, vals, width=0.38, color=colores, label=etiqueta)
        for xi, v in zip(x + desplaz, vals):
            ax.text(xi, v + 0.02, f"{num(v * 100, 0)} %", ha="center", fontsize=8.0, color=TEXTO)
    ax.set_xticks(x, [e for _, e in metricas], fontsize=8.4)
    ax.set_ylim(0, 1.15)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.6)
    _titulo(ax, "Atribución: decisiones del LLM frente a validación y verificación del código",
            "Métricas del LLM medidas sobre lo que propone antes de cualquier corrección del código")
    return _guardar(fig, nombre)


def fig_04_por_categoria(df_cat: pd.DataFrame, nombre: str = "04_respuesta_correcta_por_categoria.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    cats = list(evaluacion.ETIQUETAS_CATEGORIAS)
    x = np.arange(len(cats))
    series = [("prueba", "zeroshot_sin_tools", ROJO), ("prueba", "enrutador_reglas", NARANJA),
              ("prueba", "react_texto", AMARILLO), ("prueba", "agente", AZUL)]
    ancho = 0.2
    for i, (conjunto, clave, color) in enumerate(series):
        sub = df_cat[(df_cat["conjunto"] == conjunto) & (df_cat["clave_enfoque"] == clave)].set_index("categoria")
        vals = [float(sub.loc[c, "respuesta_correcta"]) if c in sub.index else 0.0 for c in cats]
        ax.bar(x + (i - 1.5) * ancho, vals, width=ancho, color=color,
               label={"zeroshot_sin_tools": "1 · LLM sin herramientas", "enrutador_reglas": "2 · Reglas",
                      "react_texto": "3 · ReAct", "agente": "4 · Agente"}[clave])
    ax.set_xticks(x, [evaluacion.ETIQUETAS_CATEGORIAS[c].replace(" (", "\n(") for c in cats], fontsize=8.4)
    ax.set_ylim(0, 1.12)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", fontsize=8.3)
    _titulo(ax, "Conjunto de prueba (60): respuesta correcta por tipo de pregunta",
            "Fuera de dominio = 16 preguntas de tipos que no aparecen en validación")
    return _guardar(fig, nombre)


def fig_05_mejora_prompt(df_prompt: pd.DataFrame, nombre: str = "05_mejora_prompt_validacion.png"):
    estilo()
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    metricas = df_prompt[~df_prompt["metrica"].isin(["pasos_llm_medios", "latencia_media_s"])].reset_index(drop=True)
    x = np.arange(len(metricas))
    ax.bar(x - 0.2, metricas["v1"].astype(float), width=0.38, color=GRIS_CLARO, label="v1 · descripciones originales")
    ax.bar(x + 0.2, metricas["v2"].astype(float), width=0.38, color=AZUL, label="v2 · reglas de uso + ejemplo por herramienta")
    for i, (a, b) in enumerate(zip(metricas["v1"].astype(float), metricas["v2"].astype(float))):
        ax.text(i - 0.2, a + 0.02, f"{num(a * 100, 0)}", ha="center", fontsize=7.8, color=TEXTO2)
        ax.text(i + 0.2, b + 0.02, f"{num(b * 100, 0)}", ha="center", fontsize=7.8, color=TEXTO)
    nombres = {
        "respuesta_correcta": "Respuesta\ncorrecta", "llm_seleccion_turno1": "LLM herram.\nturno 1",
        "llm_seleccion_exacta": "LLM herram.\n(bucle)", "llm_args_perfectos": "LLM herram.\n+ args",
        "llm_se_abstuvo_ood": "LLM se abstiene\n(fuera dom.)", "abstencion_correcta_ood": "Abstención\nfinal",
        "falsa_abstencion_dominio": "Falsa abstención\n(en dominio)", "fidelidad_texto_llm": "Cifras correctas\ntexto LLM",
    }
    ax.set_xticks(x, [nombres.get(m, m) for m in metricas["metrica"]], fontsize=8.2)
    ax.set_ylim(0, 1.15)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.4)
    _titulo(ax, "Mejora de prompt medida en VALIDACIÓN (44 preguntas)",
            "La versión ganadora en validación es la única que se corre en el conjunto de prueba")
    return _guardar(fig, nombre)


def fig_06_latencia_y_pasos(df_detalle: pd.DataFrame, df_resumen: pd.DataFrame,
                            nombre: str = "06_latencia_y_pasos_por_pregunta.png"):
    estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.3))
    sub = df_detalle[(df_detalle["conjunto"] == "prueba")]
    claves = [("zeroshot_sin_tools", "1 · Sin herram.", ROJO), ("react_texto", "3 · ReAct", AMARILLO),
              ("agente", "4 · Agente", AZUL)]
    datos_box = [sub[sub["clave_enfoque"] == c]["latencia_ms"].to_numpy() / 1000 for c, _, _ in claves]
    bp = a1.boxplot(datos_box, tick_labels=[e for _, e, _ in claves], patch_artist=True, widths=0.5)
    for parche, (_, _, col) in zip(bp["boxes"], claves):
        parche.set_facecolor(col)
        parche.set_alpha(0.75)
    a1.set_ylabel("Segundos por pregunta (CPU, qwen2.5:7b)")
    _titulo(a1, "Latencia por pregunta en prueba", "El enrutador por reglas (<1 ms) no aparece en esta escala")
    agente = sub[sub["clave_enfoque"] == "agente"]
    conteo = agente["n_pasos_llm"].value_counts().sort_index()
    a2.bar(conteo.index.astype(str), conteo.values, color=AZUL, width=0.55)
    for xi, v in zip(conteo.index.astype(str), conteo.values):
        a2.text(xi, v + 0.4, str(int(v)), ha="center", fontsize=8.6)
    f = _fila(df_resumen, "prueba", "agente")
    a2.set_xlabel("Turnos del LLM por pregunta")
    a2.set_ylabel("Preguntas")
    a2.grid(axis="x", visible=False)
    _titulo(a2, f"Agente: {num(float(f['pasos_llm_medios']), 2)} turnos del LLM de media",
            "1 turno = se abstiene o responde sin herramientas; 2 = datos + respuesta; 3 = calculadora")
    return _guardar(fig, nombre)


def generar_todas_las_figuras(
    df_resumen: pd.DataFrame,
    df_cat: pd.DataFrame,
    df_detalle: pd.DataFrame,
    df_prompt: pd.DataFrame,
) -> list[str]:
    """Genera las 7 figuras PNG (300 DPI) y devuelve sus rutas relativas."""
    archivos = [
        "01_cobertura_datos_banco_mundial_y_clima.png",
        "02_validacion_vs_prueba.png",
        "03_atribucion_llm_vs_codigo.png",
        "04_respuesta_correcta_por_categoria.png",
        "05_mejora_prompt_validacion.png",
        "06_latencia_y_pasos_por_pregunta.png",
        "07_cruce_real_clima_economia_latam.png",
    ]
    fig_01_cobertura_datos_banco_mundial_y_clima(archivos[0])
    fig_02_validacion_vs_prueba(df_resumen, archivos[1])
    fig_03_atribucion_llm_vs_codigo(df_resumen, archivos[2])
    fig_04_por_categoria(df_cat, archivos[3])
    fig_05_mejora_prompt(df_prompt, archivos[4])
    fig_06_latencia_y_pasos(df_detalle, df_resumen, archivos[5])
    fig_07_cruce_real_clima_economia_latam(archivos[6])
    return [str((config.FIGURAS_DIR / a).relative_to(config.RAIZ)).replace("\\", "/") for a in archivos]
