"""App Streamlit: pregunta al agente y muestra el rastro completo de herramientas de cada respuesta.

    streamlit run app/app_streamlit.py

Las preguntas de los conjuntos de validación y prueba están grabadas y funcionan sin Ollama ni red.
Una pregunta nueva necesita `ollama serve` con qwen2.5:7b; el modo "vivo" necesita internet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from agente import config, datos, evaluacion, motor_agente  # noqa: E402

st.set_page_config(page_title="Agente de datos abiertos", page_icon="🌐", layout="wide")


@st.cache_data
def _preguntas_grabadas() -> list[str]:
    return [q["pregunta"] for n in ("prueba", "validacion") for q in evaluacion.cargar_conjunto(n)]


@st.cache_data
def _resumen() -> pd.DataFrame | None:
    ruta = config.REPORTES_DIR / "resumen_enfoques.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


st.title("🌐 Agente de datos abiertos · Banco Mundial + Open-Meteo")
st.caption(
    "qwen2.5:7b (Ollama, local) elige las herramientas y sus argumentos; el código valida con Pydantic, "
    "verifica que cada argumento esté en la pregunta y ejecuta. La respuesta final se arma con plantilla "
    "a partir de los datos devueltos; el texto propio del LLM se muestra aparte."
)

with st.sidebar:
    st.header("Configuración")
    modo = st.radio(
        "Datos",
        ["instantanea", "vivo"],
        help="instantanea: archivo versionado (BM 1990-2024, clima 2020-2024). vivo: APIs HTTP con caché "
        "(BM desde 1960, clima desde 1940).",
    )
    enfoque = st.selectbox(
        "Enfoque",
        ["tool_calling_nativo", "react_texto", "enrutador_reglas", "zeroshot_sin_tools"],
        format_func=lambda e: evaluacion.ETIQUETAS_ENFOQUES[e],
    )
    st.markdown(f"Prompt del agente: **{config.VERSION_PROMPT_AGENTE}** · máximo {config.MAX_PASOS_AGENTE} turnos del LLM")
    df_res = _resumen()
    if df_res is not None and {"conjunto", "clave_enfoque"} <= set(df_res.columns):
        fila = df_res[(df_res["conjunto"] == "prueba") & (df_res["clave_enfoque"] == "agente")]
        if len(fila):
            f = fila.iloc[0]
            st.metric("Respuesta correcta en prueba (60)", f"{f['respuesta_correcta'] * 100:.1f} %")
            st.metric("Latencia media por pregunta", f"{f['latencia_media_s']:.0f} s")

origen = st.radio("Pregunta", ["Elegir una grabada (sin Ollama)", "Escribir una nueva"], horizontal=True)
if origen.startswith("Elegir"):
    pregunta = st.selectbox("Pregunta de los conjuntos de evaluación", _preguntas_grabadas())
else:
    pregunta = st.text_input("Tu pregunta", "¿Cuál fue la inflación de Ecuador entre 2019 y 2023?")

if st.button("Consultar", type="primary") and pregunta:
    datos.establecer_modo(modo)
    try:
        with st.spinner("Consultando (en CPU el agente tarda 30-50 s si la pregunta no está grabada)..."):
            salida = motor_agente.consultar_agente(pregunta, enfoque=enfoque)
    except Exception as exc:  # Ollama apagado, sin red, etc.
        st.error(str(exc))
        st.stop()
    finally:
        datos.establecer_modo("instantanea")

    if salida["se_abstuvo"]:
        st.warning(salida["respuesta_final"])
    else:
        st.success(salida["respuesta_final"])
    if salida.get("texto_llm"):
        st.markdown("**Texto que redactó el LLM** (no verificado, solo informativo):")
        st.info(salida["texto_llm"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Turnos del LLM", salida.get("n_pasos_llm", 0 if enfoque == "enrutador_reglas" else 1))
    c2.metric("Herramientas ejecutadas", len(salida["llamadas_herramientas"]))
    c3.metric("Latencia (grabada o real)", f"{salida['latencia_ms'] / 1000:.1f} s")

    st.subheader("🔎 Rastro de herramientas")
    if salida.get("pasos"):
        for paso in salida["pasos"]:
            with st.expander(f"Turno {paso['paso']} del LLM · {len(paso['llamadas'])} llamada(s)", expanded=True):
                if not paso["llamadas"]:
                    st.write("El LLM no pidió herramientas en este turno.")
                for c in paso["llamadas"]:
                    icono = "✅" if c["exito"] else "⛔"
                    st.markdown(f"{icono} `{c['herramienta']}`")
                    st.json({"argumentos_del_LLM": c["argumentos_llm"], "argumentos_validados": c["argumentos_validados"]},
                            expanded=False)
                    if not c["exito"]:
                        st.caption(f"{c['error_tipo']}: {c['mensaje_error']}")
    elif salida["llamadas_herramientas"]:
        for c in salida["llamadas_herramientas"]:
            st.markdown(f"✅ `{c['herramienta']}`")
            st.json(c["argumentos"], expanded=False)
    else:
        st.write("Ninguna herramienta llamada.")

    if salida.get("ejecuciones_herramientas"):
        st.subheader("📦 Observaciones devueltas por las herramientas")
        for ej in salida["ejecuciones_herramientas"]:
            if ej.get("exito"):
                res = {k: v for k, v in ej["resultado"].items() if k != "cifras_clave"}
                st.markdown(f"`{ej['herramienta']}` · {res.get('fuente', '')}")
                st.code(json.dumps(res, ensure_ascii=False, indent=1)[:3000], language="json")
