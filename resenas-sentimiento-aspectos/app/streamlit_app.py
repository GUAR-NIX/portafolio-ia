"""
App: sentimiento por aspecto en reseñas de teléfonos y accesorios (Amazon Reviews 2023).

    streamlit run app/streamlit_app.py
    RESENAS_PROVEEDOR=grabado streamlit run app/streamlit_app.py     # sin Ollama: respuestas grabadas del LLM

Pestañas: analizar una reseña con los tres enfoques · evaluación · hallazgos de negocio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from resenas import config  # noqa: E402
from resenas.enfoque_clasico import RUTA_MODELO, ModeloClasico  # noqa: E402
from resenas.enfoque_llm import analizar as analizar_llm  # noqa: E402
from resenas.etiquetas import cargar_muestra, leer_jsonl  # noqa: E402
from resenas.graficos import NOMBRE_ASPECTO  # noqa: E402
from resenas.llm import LLM, ErrorLLM  # noqa: E402

st.set_page_config(page_title="Aspectos en reseñas", page_icon="🔎", layout="wide")
ICONO = {"positivo": "🟦 positivo", "negativo": "🟥 negativo", "neutro": "⬜ neutro"}
NOMBRES = {"clasico": "Clásico (reglas + TF-IDF)", "embeddings": "Embeddings bge-m3 + LR", "llm": "LLM qwen2.5:7b"}


@st.cache_resource
def modelo_clasico():
    return ModeloClasico.cargar(RUTA_MODELO) if RUTA_MODELO.exists() else None


@st.cache_data
def ejemplos() -> dict[str, dict]:
    """Reseñas del test con la verdad y las predicciones grabadas de los tres enfoques."""
    ruta = config.DIR_REPORTES / "predicciones_test.jsonl"
    if not ruta.exists():
        return {}
    try:
        muestra = cargar_muestra()          # con texto, si ya se descargó y reconstruyó
    except FileNotFoundError:               # sin descarga: solo metadatos del índice (el texto no se versiona)
        muestra = {d["id"]: {**d, "titulo_resena": "", "texto": ""} for d in leer_jsonl(config.INDICE_MUESTRA)}
    return {d["id"]: {**d, "resena": muestra[d["id"]]} for d in leer_jsonl(ruta)}


@st.cache_data
def evaluacion() -> dict | None:
    p = config.DIR_REPORTES / "evaluacion.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def hallazgo(nombre: str) -> pd.DataFrame:
    return pd.read_csv(config.DIR_REPORTES / "hallazgos" / f"{nombre}.csv", index_col=0)


def tabla_menciones(res: dict) -> pd.DataFrame:
    filas = [{"Aspecto": NOMBRE_ASPECTO.get(m["aspecto"], m["aspecto"]), "Polaridad": ICONO[m["polaridad"]],
              "Cita": m.get("cita", "")} for m in res.get("aspectos", [])]
    return pd.DataFrame(filas or [{"Aspecto": "—", "Polaridad": "", "Cita": "(ningún aspecto del catálogo)"}])


def mostrar_resultado(titulo: str, res: dict | None, nota: str = ""):
    st.markdown(f"**{titulo}**")
    if res is None:
        st.caption(nota or "No disponible")
        return
    st.caption(f"Sentimiento global: {ICONO[res['sentimiento_global']]}" + (f" · {nota}" if nota else ""))
    st.dataframe(tabla_menciones(res), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- barra lateral
llm = LLM()
estado = llm.disponible()
with st.sidebar:
    st.header("🔎 Aspectos en reseñas")
    st.write("Amazon Reviews 2023 · Cell Phones & Accessories · 55 082 reseñas de 320 productos.")
    if estado["proveedor"] == "grabado":
        st.info(f"Modo demo: {estado['grabadas']} respuestas grabadas del LLM.")
    elif estado.get("ok"):
        st.success(f"Ollama conectado · {llm.modelo}")
    else:
        st.warning("Ollama no responde: el LLM solo funciona con reseñas del test (grabadas).")
        llm = LLM(proveedor="grabado")
    st.caption("Catálogo de 12 aspectos y guía de anotación en docs/GUIA_ANOTACION.md")

tab1, tab2, tab3 = st.tabs(["Analizar una reseña", "Evaluación", "Hallazgos de negocio"])

# --------------------------------------------------------------------------- 1 · analizar
with tab1:
    ej = ejemplos()
    fuente = st.radio("Reseña", ["Elegir del test (etiquetada a mano)", "Escribir una"], horizontal=True)
    if fuente.startswith("Elegir") and ej:
        ids = list(ej)
        etiqueta = {i: f"{i} · {ej[i]['resena']['estrellas']}★ · {ej[i]['resena']['tipo_producto']} · "
                       f"{ej[i]['resena']['titulo_resena'][:60]}".rstrip(" ·") for i in ids}
        inicial = ids.index("R015879") if "R015879" in ids else 0   # ejemplo con elogios y quejas a la vez
        elegido = st.selectbox("Reseña del conjunto test", ids, index=inicial, format_func=etiqueta.get)
        r = ej[elegido]["resena"]
        if r["texto"]:
            cuerpo = "  \n> ".join(x for x in r["texto"].splitlines() if x.strip())
            st.markdown(f"> **{r['titulo_resena']}**  \n> {cuerpo}")
        else:
            st.info("El texto de las reseñas no se incluye en el repositorio (licencia de Amazon Reviews 2023, solo "
                    "investigación). Para verlo: `python descargar_datos.py` y `python -m resenas.muestra`. Abajo "
                    "se ven las citas etiquetadas y lo que predijo cada enfoque.")
        st.caption(f"{r['estrellas']}★ · {r['marca']} · {r['tipo_producto']}")
        fuentes = {"✍️ A mano (verdad)": ej[elegido]["verdad"], "Clásico": ej[elegido]["clasico"],
                   "Embeddings": ej[elegido]["embeddings"], "LLM": ej[elegido]["llm"]}
        dicts = {k: {m["aspecto"]: m["polaridad"] for m in v["aspectos"]} for k, v in fuentes.items()}
        aspectos = [a for a in config.ASPECTOS if any(a in d for d in dicts.values())]
        tabla = pd.DataFrame([{"Aspecto": NOMBRE_ASPECTO[a], **{k: ICONO.get(d.get(a), "") for k, d in dicts.items()}}
                              for a in aspectos] +
                             [{"Aspecto": "Sentimiento global",
                               **{k: ICONO[v["sentimiento_global"]] for k, v in fuentes.items()}}])
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        st.caption("Celda vacía = el enfoque no detectó ese aspecto. Costo por reseña en CPU: clásico ~3 ms · "
                   "embeddings ~1 s · LLM ~20 s (aquí se muestran los resultados grabados de la evaluación).")
        with st.expander("Citas que justifican cada aspecto"):
            for k, v in fuentes.items():
                st.markdown(f"**{k}**")
                st.dataframe(tabla_menciones(v), hide_index=True, use_container_width=True)
    else:
        titulo = st.text_input("Título", "Great case but terrible seller")
        texto = st.text_area("Reseña (en inglés, como las de Amazon)",
                             "Fits my iPhone 13 perfectly and the purple color is beautiful. It already saved my "
                             "phone from two drops. But it arrived two weeks late and the seller never answered.")
        c1, c2 = st.columns(2)
        with c1:
            m = modelo_clasico()
            mostrar_resultado(NOMBRES["clasico"], m.analizar(titulo, texto) if m else None,
                              "" if m else "Falta data/modelo_clasico.joblib (python -m resenas.evaluar)")
        with c2:
            if st.button("Analizar con el LLM (~20 s en CPU)"):
                try:
                    with st.spinner("qwen2.5:7b pensando…"):
                        res = analizar_llm(titulo, texto, llm)
                    mostrar_resultado(NOMBRES["llm"], res, f"{res['segundos']:.0f} s · {res['tokens_salida']} tokens")
                except ErrorLLM as e:
                    st.error(str(e))

# --------------------------------------------------------------------------- 2 · evaluación
with tab2:
    ev = evaluacion()
    if not ev:
        st.info("Corre `python -m resenas.evaluar` para generar la evaluación.")
    else:
        st.subheader(f"Test: {ev['n_test']} reseñas etiquetadas a mano")
        filas = []
        for k, r in ev["enfoques"].items():
            filas.append({"Enfoque": NOMBRES[k], "F1 aspectos": r["f1_aspectos"],
                          "F1 aspecto + polaridad": r["f1_aspecto_polaridad"],
                          "Exactitud polaridad": r["exactitud_polaridad"], "Exactitud global": r["exactitud_global"],
                          "Segundos por reseña": r["costo"]["segundos_por_resena"]})
        st.dataframe(pd.DataFrame(filas).style.format(precision=3), hide_index=True, use_container_width=True)
        a = ev["acuerdo_anotacion"]
        st.caption(f"Acuerdo entre dos pasadas de etiquetado ({a['n_resenas']} reseñas): F1 de aspectos "
                   f"{a['f1_aspectos_entre_pasadas']:.2f} · kappa de polaridad {a['kappa_polaridad']:.2f}")
        for f in ("01_comparacion_enfoques", "02_f1_por_aspecto", "03_costo_vs_calidad"):
            st.image(str(config.DIR_FIGURAS / f"{f}.png"))

# --------------------------------------------------------------------------- 3 · hallazgos
with tab3:
    if not (config.DIR_REPORTES / "hallazgos" / "aspectos.csv").exists():
        st.info("Corre `python -m resenas.aplicar` y `python -m resenas.hallazgos`.")
    else:
        resumen = json.loads((config.DIR_REPORTES / "hallazgos.json").read_text(encoding="utf-8"))
        c = st.columns(4)
        c[0].metric("Reseñas analizadas", f"{resumen['resenas']:,}".replace(",", " "))
        c[1].metric("Con al menos un aspecto", f"{resumen['pct_resenas_con_algun_aspecto']} %")
        c[2].metric("Con al menos una queja", f"{resumen['pct_resenas_con_alguna_queja']} %")
        c[3].metric("Estrellas medias", resumen["estrellas_medias"])
        st.image(str(config.DIR_FIGURAS / "04_aspectos_quejas_elogios.png"))
        st.image(str(config.DIR_FIGURAS / "05_quejas_tipo_aspecto.png"))
        st.markdown("**Productos con más quejas** (≥ 150 reseñas en la muestra)")
        prods = hallazgo("productos")
        tipo = st.selectbox("Tipo de producto", ["(todos)"] + sorted(prods["tipo"].unique()))
        if tipo != "(todos)":
            prods = prods[prods["tipo"] == tipo]
        prods = prods.assign(queja_principal=prods["queja_principal"].map(NOMBRE_ASPECTO))
        st.dataframe(prods.head(20), use_container_width=True)
        st.markdown("**Marcas** (≥ 600 reseñas)")
        st.dataframe(hallazgo("marcas").assign(queja_principal=lambda d: d["queja_principal"].map(NOMBRE_ASPECTO)),
                     use_container_width=True)
        for f in ("06_quejas_vs_estrellas", "07_quejas_por_anio", "08_marcas_calidad"):
            st.image(str(config.DIR_FIGURAS / f"{f}.png"))
        st.markdown("**Cuántas estrellas se asocian a cada queja o elogio** (regresión lineal con todas a la vez)")
        imp = hallazgo("impacto_estrellas")[["coef"]].rename(columns={"coef": "55 mil reseñas (modelo)"}).join(
            hallazgo("impacto_estrellas_a_mano")[["coef", "ic_bajo", "ic_alto"]].rename(
                columns={"coef": "620 etiquetadas a mano", "ic_bajo": "IC 95 % bajo", "ic_alto": "IC 95 % alto"}))
        st.dataframe(imp.drop(index="intercepto"), use_container_width=True)
        st.caption("La columna del modelo es en parte circular (su clasificador de polaridad aprendió de las estrellas); "
                   "la de etiquetas a mano no, porque se etiquetó sin ver las estrellas.")
