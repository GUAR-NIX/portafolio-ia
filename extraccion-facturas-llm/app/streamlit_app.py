"""
App: sube una factura en PDF → texto (pypdf u OCR) → JSON validado con el LLM local, comparado con la regex.

    streamlit run app/streamlit_app.py
    FACTURAS_PROVEEDOR=grabado streamlit run app/streamlit_app.py      # sin Ollama: solo las facturas de ejemplo

Parámetros de URL (útiles para capturas): ?muestra=FAC_002&auto=1
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from facturas import config  # noqa: E402
from facturas.esquema import CAMPOS  # noqa: E402
from facturas.extractor import extraer_de_texto  # noqa: E402
from facturas.llm import LLM, ErrorLLM  # noqa: E402
from facturas.metricas import campo_correcto, comparar  # noqa: E402
from facturas.regex_base import extraer_regex  # noqa: E402
from facturas.texto import extraer_texto, imagenes_de_paginas, ocr_disponible  # noqa: E402
from facturas.validaciones import validar  # noqa: E402

st.set_page_config(page_title="Extractor de facturas", page_icon="🧾", layout="wide")

ETIQUETAS = {"ruc_emisor": "RUC emisor", "razon_social_emisor": "Razón social", "numero_factura": "N.º de factura",
             "numero_autorizacion": "N.º de autorización", "fecha_emision": "Fecha de emisión",
             "cliente_nombre": "Cliente", "cliente_identificacion": "CI/RUC cliente", "subtotal": "Subtotal",
             "descuento": "Descuento", "iva": "IVA", "total": "Total"}
ESTADOS = {"valida": ("✅", "Válida: cuadran todas las reglas"),
           "con_advertencias": ("🟡", "Válida, con advertencias (falta un dato opcional)"),
           "revisar": ("🟠", "Para revisión humana: no pasa alguna regla de negocio"),
           "error": ("❌", "No se pudo extraer")}


@st.cache_data(show_spinner=False)
def leer(pdf_bytes: bytes):
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "factura.pdf"
        ruta.write_bytes(pdf_bytes)
        t = extraer_texto(ruta)
        img = imagenes_de_paginas(ruta, dpi=90)[0]
    return t, img


@st.cache_data(show_spinner=False)
def extraer_llm(texto: str, proveedor: str, reintentos: int):
    llm = LLM(proveedor=proveedor)
    r = extraer_de_texto(texto, llm, reintentos)
    return r.dict(), [i.__dict__ for i in r.intentos], llm.uso.__dict__


# ------------------------------------------------------------------ barra lateral
qp = st.query_params
with st.sidebar:
    st.header("🧾 Extractor de facturas")
    st.caption("PDF → texto → LLM local (qwen2.5:7b) → JSON validado con reglas del SRI")
    disp = LLM().disponible()
    proveedor = st.radio("Modelo", ["ollama", "grabado"],
                         index=0 if (config.PROVEEDOR == "ollama" and disp.get("ok")) else 1,
                         help="«grabado» reproduce las respuestas guardadas en la evaluación (sin Ollama).")
    if proveedor == "ollama":
        st.write("Ollama:", "🟢 listo" if disp.get("ok") else f"🔴 no disponible ({disp.get('error', 'falta el modelo')})")
    reintentos = st.slider("Reintentos con corrección", 0, 2, config.MAX_REINTENTOS)
    st.write("OCR de Windows:", "🟢" if ocr_disponible() else "🔴 (solo PDF digitales)")
    muestras = sorted(p.stem for p in config.DIR_MUESTRAS.glob("*.pdf"))
    elegida = st.selectbox("…o usa una factura de ejemplo", ["(ninguna)"] + muestras,
                           index=(muestras.index(qp["muestra"]) + 1) if qp.get("muestra") in muestras else 0)

st.title("Factura PDF → JSON validado")
subida = st.file_uploader("Sube una factura en PDF", type=["pdf"])
if subida:
    pdf, nombre, verdad = subida.getvalue(), subida.name, None
elif elegida != "(ninguna)":
    pdf, nombre = (config.DIR_MUESTRAS / f"{elegida}.pdf").read_bytes(), elegida
    verdad = json.loads((config.DIR_MUESTRAS / f"{elegida}.json").read_text(encoding="utf-8"))
else:
    st.info("Sube un PDF o elige una factura de ejemplo en la barra lateral. Las de ejemplo son sintéticas "
            "(5 diseños distintos, algunas escaneadas o giradas) y traen la respuesta correcta para comparar.")
    st.stop()

texto, imagen = leer(pdf)
col_pdf, col_res = st.columns([2, 3])
with col_pdf:
    st.image(imagen, caption=nombre, use_container_width=True)
    etq = {"pypdf": "capa de texto del PDF (pypdf)", "ocr": "OCR de Windows (sin capa de texto)",
           "sin_texto": "sin texto"}[texto.metodo]
    with st.expander(f"Texto extraído · {etq}", expanded=False):
        for a in texto.avisos:
            st.warning(a)
        st.code(texto.texto or "(vacío)", language=None)

with col_res:
    if not texto.texto:
        st.error(" ".join(texto.avisos))
        st.stop()
    if st.button("Extraer con el LLM", type="primary") or qp.get("auto") == "1":
        try:
            with st.spinner("El modelo está leyendo la factura (≈1 min por intento en CPU)…"):
                res, intentos, uso = extraer_llm(texto.texto, proveedor, reintentos)
        except ErrorLLM as e:
            st.error(str(e))
            st.stop()
        icono, desc = ESTADOS[res["estado"]]
        st.subheader(f"{icono} {desc}")
        st.caption(f"{uso['llamadas']} llamada(s) al LLM · {uso['segundos']:.0f} s · "
                   f"{uso['tokens_entrada']} tokens de entrada · {uso['tokens_salida']} de salida"
                   + (" · respuesta grabada" if uso["desde_grabacion"] else ""))
        f = res["factura"]
        if f is None:
            st.error(res["mensaje"])
            st.stop()
        rx = extraer_regex(texto.texto).model_dump(mode="json")
        filas = []
        for c in CAMPOS:
            fila = {"Campo": ETIQUETAS[c], "LLM": f[c]}
            if verdad:
                fila["LLM ✓"] = "✅" if campo_correcto(c, f[c], verdad[c]) else "❌"
            fila["Regex"] = rx[c]
            if verdad:
                fila["Regex ✓"] = "✅" if campo_correcto(c, rx[c], verdad[c]) else "❌"
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas).astype(str).replace("None", "—"), hide_index=True,
                     use_container_width=True, height=36 * (len(filas) + 1) + 4,
                     column_config={"LLM ✓": st.column_config.TextColumn(width="small"),
                                    "Regex ✓": st.column_config.TextColumn(width="small")})
        st.markdown(f"**Ítems ({len(f['items'])})**")
        st.dataframe(pd.DataFrame(f["items"]), hide_index=True, use_container_width=True)
        if verdad:
            cl, cr = comparar(f, verdad), comparar(rx, verdad)
            st.caption(f"Contra la respuesta correcta: LLM {cl['campos_ok']}/{len(CAMPOS)} campos, F1 de ítems "
                       f"{cl['items_f1']:.0%} · Regex {cr['campos_ok']}/{len(CAMPOS)} campos, F1 {cr['items_f1']:.0%}")
        st.markdown("**Validaciones de negocio**")
        if not res["problemas"]:
            st.success("RUC y clave de acceso con dígito verificador correcto · la clave coincide con RUC, fecha y "
                       "número · las líneas suman el subtotal · subtotal − descuento + IVA = total · IVA a la tasa "
                       "vigente en la fecha")
        for p in res["problemas"]:
            (st.error if p["nivel"] == "error" else st.warning)(p["mensaje"])
        if len(intentos) > 1:
            with st.expander(f"Reintentos ({len(intentos) - 1})"):
                for i in intentos:
                    st.write(f"Intento {i['numero'] + 1}: " + ("sin errores" if not i["errores"]
                                                               else "; ".join(i["errores"])))
        st.download_button("Descargar JSON", json.dumps(f, ensure_ascii=False, indent=1),
                           file_name=f"{Path(nombre).stem}.json", mime="application/json")
    else:
        st.write("La regex (instantánea) encontró esto; el LLM tarda ≈1 min en CPU.")
        rx = extraer_regex(texto.texto)
        probs = validar(rx)
        st.json(rx.model_dump(mode="json"), expanded=False)
        st.caption(f"Regex: {sum(p.nivel == 'error' for p in probs)} error(es) de validación")
