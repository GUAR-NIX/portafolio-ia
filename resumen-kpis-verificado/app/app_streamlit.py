"""App Streamlit: elige país y año, mira el informe y, junto a cada cifra, si fue verificada.

    streamlit run app/app_streamlit.py

Sin Ollama funcionan la plantilla (cualquier país y año) y los informes del LLM ya grabados
(los 17 de evaluación). Con Ollama encendido y la casilla marcada, genera y graba informes nuevos.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpis import capa_kpis, comparaciones, config, evaluacion, generadores, llm  # noqa: E402
from kpis.numeros import normalizar  # noqa: E402

st.set_page_config(page_title="Resumen ejecutivo de KPIs verificado", page_icon="📊", layout="wide")

ESTILO = {"correcta": ("#e3f5e3", "#0a6b0a", "✓"), "incorrecta": ("#fbe4e4", "#a32020", "✗"),
          "sin_fuente": ("#fdeee5", "#9a4a1c", "?")}


def ollama_activo() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def resaltar(texto: str, cifras: list[dict]) -> str:
    """HTML con cada cifra marcada: verde = coincide con la capa de KPIs; rojo = no coincide; naranja = sin fuente."""
    t = normalizar(texto)
    partes, pos = [], 0
    for c in sorted(cifras, key=lambda c: c["inicio"]):
        if c["inicio"] < pos:
            continue
        fondo, tinta, icono = ESTILO[c["estado"]]
        if c["estado"] == "correcta":
            nota = f"verificada: {c['real_formateado']} ({c['hecho_id'].replace('|', ' · ')})"
            sufijo = icono
        elif c["estado"] == "incorrecta":
            nota = f"NO coincide; valor real {c['real_formateado']} ({c['hecho_id'].replace('|', ' · ')})"
            sufijo = f"{icono} real: {c['real_formateado']}"
        else:
            nota = "sin fuente en la capa de KPIs"
            sufijo = f"{icono} sin fuente"
        partes.append(html.escape(t[pos:c["inicio"]]))
        partes.append(f'<span title="{html.escape(nota)}" style="background:{fondo};color:{tinta};'
                      f'border-radius:4px;padding:1px 4px;font-weight:600">{html.escape(t[c["inicio"]:c["fin"]])}'
                      f' <small>{html.escape(sufijo)}</small></span>')
        pos = c["fin"]
    partes.append(html.escape(t[pos:]))
    return "".join(partes)


st.title("📊 Resumen ejecutivo de KPIs con verificación de cada cifra")
st.caption("Datos: Banco Mundial, World Development Indicators (CC BY 4.0). Región = mediana de 18 países de "
           "América Latina. Verde: la cifra coincide con la capa de KPIs; rojo: no coincide (se muestra el valor "
           "real); naranja: no tiene fuente.")

evaluados = {f"{config.PAISES[p]} {a}": (p, a) for p, a in config.PRUEBA + config.VALIDACION}
with st.sidebar:
    modo = st.radio("Informe", ["De la evaluación (grabado)", "Cualquier país y año"])
    if modo.startswith("De la"):
        pais, anio = evaluados[st.selectbox("País y año", list(evaluados))]
    else:
        pais = st.selectbox("País", list(config.PAISES), format_func=lambda p: config.PAISES[p],
                            index=list(config.PAISES).index("ECU"))
        anio = st.slider("Año", 2010, 2025, 2024)
    enfoque = st.radio("Enfoque", generadores.ENFOQUES, index=2, format_func=lambda e: generadores.NOMBRES[e])
    permitir = st.checkbox("Permitir llamar a Ollama si no está grabado", value=False)
    if permitir and not ollama_activo():
        st.warning("Ollama no responde en " + config.OLLAMA_URL)

try:
    version = evaluacion.elegir_version_prompt()
except llm.SinGrabacion:
    version = "v2"
try:
    r = generadores.generar(enfoque, pais, anio, version, proveedor="ollama" if permitir else "grabado")
except llm.SinGrabacion:
    st.info("Este informe del LLM no está grabado. Elige un informe de la evaluación, usa la plantilla o "
            "enciende Ollama y marca la casilla.")
    st.stop()

tab_inf, tab_kpi, tab_ecu = st.tabs(["Informe verificado", "Tabla de KPIs (fuente de verdad)", "Ecuador vs región"])
with tab_inf:
    v = r["verificacion"]
    if v is None:
        st.error("El LLM no devolvió un JSON válido.")
        st.code(r["texto"])
        st.stop()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cifras", v["n_cifras"])
    c2.metric("✓ Correctas", v["n_correcta"])
    c3.metric("✗ Incorrectas", v["n_incorrecta"])
    c4.metric("? Sin fuente", v["n_sin_fuente"])
    c5.metric("Tiempo", f"{r['segundos_llm'] + r['segundos_verificacion']:.0f} s",
              help="Tiempo del LLM en CPU (grabado) + verificación")
    if r["enfoque"] == "verificado":
        st.caption(f"Intentos del LLM: {r['llamadas_llm']} · " + " → ".join(
            f"{(i['tasa_alucinacion'] or 0):.0%}" for i in r["intentos"]) + " de cifras alucinadas por intento")
    titulos = {"titular": "", "hallazgo": "Hallazgos", "riesgo": "Riesgos", "recomendacion": "Recomendaciones"}
    ultimo = None
    for s in v["secciones"]:
        tipo = s["seccion"].split("_")[0]
        if tipo in titulos and tipo != ultimo and titulos[tipo]:
            st.markdown(f"#### {titulos[tipo]}")
        ultimo = tipo
        cuerpo = resaltar(s["texto"], s["cifras"])
        if tipo == "titular":
            st.markdown(f"### {cuerpo}", unsafe_allow_html=True)
        else:
            st.markdown(("• " if tipo != "linea" else "") + cuerpo, unsafe_allow_html=True)
    st.caption("Ojo: una cifra verde solo dice que el número existe en la capa de KPIs para ese indicador; "
               "la redacción puede seguir siendo engañosa (p. ej., llamar «crecimiento del PIB» a la variación "
               "del PIB en dólares corrientes).")
    with st.expander("Detalle de cada cifra"):
        filas = [{"sección": s["seccion"], **{k: c[k] for k in ("texto", "estado", "real_formateado", "hecho_id",
                                                               "motivo")}, "marcas": ", ".join(c["marcas"])}
                 for s in v["secciones"] for c in s["cifras"]]
        st.dataframe(pd.DataFrame(filas), width="stretch")
    comps = comparaciones.revisar_resultado(r)
    if comps:
        malas = [c for c in comps if not c["correcta"]]
        st.markdown(f"**Dirección de las comparaciones**: {len(comps) - len(malas)} de {len(comps)} correctas")
        for c in malas:
            st.markdown(f"✗ «{c['clausula']}» — dice *{c['dice']}*, pero el dato va *{c['real']}* "
                        f"({config.IND_POR_CLAVE[c['indicador']].nombre})")

with tab_kpi:
    st.markdown(capa_kpis.tabla_para_prompt(pais, anio))
    st.caption("Es la misma tabla que recibe el LLM. Puesto 1 = valor más alto de la región.")
    an = capa_kpis.anomalias(pais, desde=anio - 5)
    if len(an):
        st.markdown("**Anomalías de los últimos 5 años**")
        st.dataframe(an, width="stretch")

with tab_ecu:
    comp = capa_kpis.comparacion_con_region("ECU", anio)
    comp["indicador"] = comp["indicador"].map(lambda k: config.IND_POR_CLAVE[k].nombre)
    st.dataframe(comp.round(2), width="stretch")
    fig = config.FIGURAS / "01_ecuador_vs_region.png"
    if fig.exists():
        st.image(str(fig))
