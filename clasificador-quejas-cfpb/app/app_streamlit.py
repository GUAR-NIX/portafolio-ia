"""App Streamlit: pega una queja y mira la ruta completa (TF-IDF → LLM o triaje) y la urgencia.

    streamlit run app/app_streamlit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from quejas import config, predecir  # noqa: E402

EJEMPLOS = {
    "Hipoteca con riesgo de ejecución": (
        "My mortgage servicer refused my loan modification application twice and now they scheduled a "
        "foreclosure sale of my home next month even though I sent all the documents they asked for."
    ),
    "Cobranza por deuda de robo de identidad": (
        "A collection agency keeps calling me about a credit card debt that is not mine. Someone opened "
        "the account using my identity and I already filed a police report, but they keep threatening me."
    ),
    "Queja ambigua (cuenta y tarjeta)": (
        "I was charged a fee and the payment I made was not credited to my account. Customer service "
        "transferred me several times and nobody could explain the balance."
    ),
    "Queja en español": (
        "Me cobraron dos veces la cuota de mi tarjeta de crédito y el banco no quiere devolverme el dinero; "
        "además me pusieron un cargo por pago atrasado que no corresponde."
    ),
}

st.set_page_config(page_title="Enrutador de quejas CFPB", page_icon="📨", layout="wide")
st.title("📨 Enrutador de quejas financieras")
st.caption(
    "TF-IDF + LR decide si está seguro; si no, la queja va al LLM (qwen2.5:7b few-shot). "
    "Modelos entrenados con quejas reales del CFPB en inglés."
)

meta = predecir.cargar_metadata()
with st.sidebar:
    st.subheader("Configuración")
    consultar = st.checkbox(
        "Consultar Ollama en vivo si la queja va al LLM",
        value=False,
        help="Sin esta opción solo se usan respuestas grabadas; si no hay, la queja va a triaje humano.",
    )
    st.metric("Umbral de cascada (confianza TF-IDF)", f"{meta.get('umbral_cascada', 0):.2f}")
    st.metric("Umbral de urgencia", f"{meta.get('umbral_urgencia', 0):.2f}")
    st.caption("Ambos umbrales se eligieron en el conjunto de validación, no en el de prueba.")

opcion = st.selectbox("Ejemplo", ["(escribir mi propia queja)"] + list(EJEMPLOS))
texto = st.text_area(
    "Texto de la queja",
    value=EJEMPLOS.get(opcion, ""),
    height=160,
    placeholder="Pega aquí la narrativa del cliente (idealmente en inglés)...",
)

if st.button("Enrutar", type="primary") and texto.strip():
    try:
        res = predecir.enrutar_con_cascada(texto, permitir_ollama=consultar)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Destino", res["destino_operativo"].replace("Mesa especialista: ", ""))
    c2.metric("Decidido por", res["decidido_por"])
    c3.metric("Prioridad", "🔴 Alta" if res["urgente"] else "🟢 Normal")

    st.subheader("Ruta")
    for i, paso in enumerate(res["pasos"], start=1):
        st.markdown(f"{i}. {paso}")

    base = res["clasificador_barato"]
    probs = pd.DataFrame({
        "mesa": [config.ETIQUETAS_CATEGORIA[c] for c in config.CATEGORIAS],
        "probabilidad TF-IDF (calibrada)": [base["probabilidades"][c] for c in config.CATEGORIAS],
    }).set_index("mesa")
    st.subheader("Probabilidades del clasificador barato")
    st.bar_chart(probs)
    if res["llm"]:
        with st.expander("Respuesta del LLM"):
            st.code(json.dumps(res["llm"], indent=2, ensure_ascii=False), language="json")
    st.caption(
        "Aviso: el clasificador léxico se entrenó solo en inglés; con textos en español pierde exactitud "
        "y la cascada tiende a mandarlos al LLM (ver README)."
    )
