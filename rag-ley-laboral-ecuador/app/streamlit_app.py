"""
App de chat con citas clicables: cada artículo citado se abre con su texto completo.

    streamlit run app/streamlit_app.py
    $env:RAG_PROVEEDOR="grabado"; streamlit run app/streamlit_app.py     # demo sin Ollama: preguntas de la evaluación
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_laboral import config  # noqa: E402
from rag_laboral.generar import Asistente  # noqa: E402
from rag_laboral.llm import ErrorLLM  # noqa: E402

st.set_page_config(page_title="Asistente laboral Ecuador", page_icon="⚖️", layout="wide")


@st.cache_resource
def cargar() -> Asistente:
    umbral = None
    ruta = config.REPORTS / "generacion.json"
    if ruta.exists():
        umbral = json.loads(ruta.read_text(encoding="utf-8")).get("umbral", {}).get("valor")
    return Asistente(umbral_rerank=umbral)


asistente = cargar()
preguntas = pd.read_csv(config.PREGUNTAS, dtype=str).fillna("")

st.title("⚖️ Asistente de la ley laboral del Ecuador")
st.caption("Responde solo con el Código del Trabajo (codificación 2005-017, última reforma 2026-04-09) y los artículos "
           "laborales de la Constitución. Cita cada artículo y dice cuando la ley cargada no cubre la pregunta. "
           "**No reemplaza la asesoría de un abogado.**")

with st.sidebar:
    estado = asistente.llm.disponible()
    st.markdown(f"**Modo:** `{estado['proveedor']}` · {'✅' if estado['ok'] else '⚠️'}")
    st.markdown(f"**Artículos indexados:** {len(asistente.buscador.articulos)}")
    if asistente.umbral is not None:
        st.markdown(f"**Umbral de rechazo (re-ranker):** {asistente.umbral:.3f}")
    st.markdown("---")
    st.markdown("**Ejemplos (respuestas grabadas):**")
    ejemplo = st.selectbox("Pregunta de la evaluación", [""] + preguntas.pregunta.tolist())

pregunta = st.chat_input("Pregunta sobre vacaciones, décimos, horas extra, despido, liquidación…") or ejemplo

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en la ley y redactando (en CPU tarda ~1 min)…"):
                r = asistente.responder(pregunta)
        except (ErrorLLM, RuntimeError) as e:
            st.error(str(e))
            st.stop()
        if r.cubierto_por_la_ley:
            st.markdown(r.respuesta)
            st.caption(f"Confianza del modelo: {r.confianza} · recuperación {r.segundos_recuperacion:.2f} s · "
                       f"LLM {r.segundos_llm:.0f} s{' (grabada)' if r.desde_grabacion else ''}")
        else:
            st.warning(r.respuesta)
            if r.rechazo_por_umbral:
                st.caption(f"Rechazo sin llamar al LLM: el artículo más parecido tiene puntaje {r.puntaje_max_rerank:.3f}, "
                           f"bajo el umbral.")
        if r.citas_eliminadas:
            st.caption(f"Se eliminaron citas que no estaban en el contexto: {', '.join(r.citas_eliminadas)}")
        if r.articulos_citados:
            st.markdown("**Artículos citados** (clic para ver el texto):")
            for c in r.articulos_citados:
                a = asistente.buscador.por_id[c]
                with st.expander(f"📜 {a['fuente']}, art. {a['numero']} — {a.get('epigrafe') or ''}"):
                    st.caption(" · ".join(x for x in (a.get("titulo"), a.get("capitulo"), a.get("seccion")) if x))
                    st.markdown(a["texto"].replace("\n", "  \n"))
        with st.expander("Artículos que leyó el modelo"):
            st.write(", ".join(r.contexto))
