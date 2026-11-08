"""Evaluación completa sobre VALIDACIÓN (44, ajuste) y PRUEBA (60, no vista) a partir de las grabaciones del LLM.

    python -m agente.ejecutar_evaluacion

No llama a Ollama si todas las respuestas están en data/grabacion_llm.jsonl (ver agente.correr_llm).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import config, datos, evaluacion, graficos, herramientas, motor_agente

# (conjunto, clave de enfoque, enfoque del motor, versión de prompt)
CORRIDAS: list[tuple[str, str, str, str | None]] = [
    ("validacion", "zeroshot_sin_tools", "zeroshot_sin_tools", None),
    ("validacion", "enrutador_reglas", "enrutador_reglas", None),
    ("validacion", "react_texto", "react_texto", None),
    ("validacion", "agente_v1", "tool_calling_nativo", "v1"),
    ("validacion", "agente_v2", "tool_calling_nativo", "v2"),
    ("prueba", "zeroshot_sin_tools", "zeroshot_sin_tools", None),
    ("prueba", "enrutador_reglas", "enrutador_reglas", None),
    ("prueba", "react_texto", "react_texto", None),
    ("prueba", "agente", "tool_calling_nativo", "ELEGIDA"),
]

ETIQUETAS = {
    "zeroshot_sin_tools": "1 · LLM sin herramientas",
    "enrutador_reglas": "2 · Enrutador por reglas (sin LLM)",
    "react_texto": "3 · ReAct en texto (LLM, 1 turno)",
    "agente_v1": "4 · Agente tool-calling · prompt v1",
    "agente_v2": "4 · Agente tool-calling · prompt v2",
    "agente": "4 · Agente tool-calling",
}

METRICAS_IC = ["respuesta_correcta", "llm_seleccion_exacta", "llm_args_perfectos", "abstencion_correcta"]


def _media(sub: pd.DataFrame, col: str) -> float | None:
    if col not in sub or sub[col].dropna().empty:
        return None
    return round(float(sub[col].dropna().mean()), 4)


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (conjunto, clave), sub in df.groupby(["conjunto", "clave_enfoque"], sort=False):
        dom = sub[~sub["debe_abstenerse"]]
        ood = sub[sub["debe_abstenerse"]]
        fila: dict[str, Any] = {
            "conjunto": conjunto,
            "clave_enfoque": clave,
            "etiqueta": ETIQUETAS[clave],
            "version_prompt": sub["version_prompt"].iloc[0],
            "n_preguntas": len(sub),
            "n_dominio": len(dom),
            "n_fuera_dominio": len(ood),
            "respuesta_correcta": _media(sub, "respuesta_correcta"),
            "respuesta_correcta_dominio": _media(dom, "respuesta_correcta"),
            "abstencion_correcta_ood": _media(ood, "abstencion_correcta"),
            "ejecuto_herramienta_ood": _media(ood, "ejecuto_herramienta_ood"),
            "falsa_abstencion_dominio": _media(dom, "falsa_abstencion"),
            "llm_seleccion_turno1": _media(dom, "llm_seleccion_turno1"),
            "llm_seleccion_exacta": _media(dom, "llm_seleccion_exacta"),
            "llm_f1_herramientas": _media(dom, "llm_f1_herramientas"),
            "llm_exactitud_args": _media(dom, "llm_exactitud_args"),
            "llm_args_perfectos": _media(dom, "llm_args_perfectos"),
            "llm_se_abstuvo_ood": _media(ood, "llm_se_abstuvo_ood"),
            "sistema_seleccion_exacta": _media(dom, "sistema_seleccion_exacta"),
            "sistema_exactitud_args": _media(dom, "sistema_exactitud_args"),
            "fidelidad_respuesta_final": _media(dom, "fidelidad_respuesta_final"),
            "fidelidad_texto_llm": _media(dom, "fidelidad_texto_llm"),
            "cifras_sin_respaldo_texto_llm_media": _media(dom, "cifras_sin_respaldo_texto_llm"),
            "pasos_llm_medios": round(float(sub["n_pasos_llm"].mean()), 2),
            "pasos_llm_medios_dominio": round(float(dom["n_pasos_llm"].mean()), 2) if len(dom) else None,
            "latencia_media_s": round(float(sub["latencia_ms"].mean()) / 1000, 2),
            "latencia_mediana_s": round(float(sub["latencia_ms"].median()) / 1000, 2),
            "latencia_p90_s": round(float(sub["latencia_ms"].quantile(0.9)) / 1000, 2),
            "tokens_prompt_medio": round(float(sub["tokens_prompt"].mean()), 1),
            "tokens_salida_medio": round(float(sub["tokens_salida"].mean()), 1),
        }
        for col in METRICAS_IC:
            base = ood if col == "abstencion_correcta" else (sub if col == "respuesta_correcta" else dom)
            if col in base and not base[col].dropna().empty:
                inf, sup, metodo = evaluacion.ic95(base[col].to_numpy())
                fila[f"{col}_ic95_inf"], fila[f"{col}_ic95_sup"], fila[f"{col}_ic95_metodo"] = inf, sup, metodo
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_atribucion(df_resumen: pd.DataFrame) -> pd.DataFrame:
    """Qué parte del resultado del agente se debe al LLM y qué parte al código determinista."""
    filas = []
    for conjunto, clave in (("validacion", None), ("prueba", "agente")):
        if clave is None:
            clave = "agente_" + df_resumen.loc[df_resumen["conjunto"] == "prueba", "version_prompt"].dropna().iloc[-1]
        r = df_resumen[(df_resumen["conjunto"] == conjunto) & (df_resumen["clave_enfoque"] == clave)].iloc[0]
        etapas = [
            ("LLM", "Elige la combinación exacta de herramientas en el 1.er turno", r["llm_seleccion_turno1"]),
            ("LLM", "Elige la combinación exacta de herramientas (todos los turnos)", r["llm_seleccion_exacta"]),
            ("LLM", "Herramientas y argumentos exactos (año, país, ciudad, cifras)", r["llm_args_perfectos"]),
            ("LLM", "Exactitud media de argumentos", r["llm_exactitud_args"]),
            ("LLM", "Se abstiene solo (no llama herramientas) fuera de dominio", r["llm_se_abstuvo_ood"]),
            ("Código", "Abstención final fuera de dominio (tras Pydantic + anclaje)", r["abstencion_correcta_ood"]),
            ("Código", "Falsa abstención en dominio (anclaje/validación demasiado estrictos)", r["falsa_abstencion_dominio"]),
            ("LLM", "Fidelidad numérica del texto que redacta el LLM", r["fidelidad_texto_llm"]),
            ("Código", "Fidelidad de la respuesta final (plantilla: por construcción)", r["fidelidad_respuesta_final"]),
            ("Sistema", "Respuesta correcta (todas las preguntas)", r["respuesta_correcta"]),
        ]
        for quien, etapa, valor in etapas:
            filas.append({"conjunto": conjunto, "version_prompt": r["version_prompt"], "responsable": quien,
                          "etapa": etapa, "valor": valor})
    return pd.DataFrame(filas)


def ejecutar_pipeline_completo() -> dict[str, Any]:
    config.REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    datos.establecer_modo("instantanea")

    print("1/5 · Construyendo conjuntos de validación (44) y prueba (60) con verdad terreno...")
    conjuntos = {n: evaluacion.construir_conjunto(n, guardar=True) for n in ("validacion", "prueba")}
    version = evaluacion.elegir_version_prompt()
    print(f"    versión de prompt elegida en validación: {version}")

    print("2/5 · Evaluando enfoques (grabaciones del LLM)...")
    filas: list[dict[str, Any]] = []
    trazas: list[dict[str, Any]] = []
    for conjunto, clave, enfoque, ver in CORRIDAS:
        ver_efectiva = version if ver == "ELEGIDA" else ver
        for q in conjuntos[conjunto]:
            salida = motor_agente.consultar_agente(q["pregunta"], enfoque=enfoque, version_prompt=ver_efectiva)
            fila = evaluacion.evaluar_respuesta_individual(q, salida)
            fila.update({
                "conjunto": conjunto,
                "clave_enfoque": clave,
                "pregunta": q["pregunta"],
                "herramientas_esperadas": ", ".join(q["herramientas_esperadas"]) or "ninguna",
                "herramientas_ejecutadas": ", ".join(c["herramienta"] for c in salida["llamadas_herramientas"]) or "ninguna",
                "respuesta_final": salida["respuesta_final"],
                "texto_llm": salida.get("texto_llm", ""),
            })
            filas.append(fila)
            if enfoque == "tool_calling_nativo" and ver == "ELEGIDA":
                trazas.append({"id_pregunta": q["id_pregunta"], "pregunta": q["pregunta"], "pasos": salida["pasos"],
                               "respuesta_final": salida["respuesta_final"], "texto_llm": salida["texto_llm"]})
    df = pd.DataFrame(filas)
    df.to_csv(config.REPORTES_DIR / "detalle_evaluacion.csv", index=False, encoding="utf-8")

    print("3/5 · Métricas agregadas, IC 95 %, atribución y mejora de prompt...")
    df_resumen = resumir(df)
    df_resumen.to_csv(config.REPORTES_DIR / "resumen_enfoques.csv", index=False, encoding="utf-8")
    df_atrib = tabla_atribucion(df_resumen)
    df_atrib.to_csv(config.REPORTES_DIR / "atribucion_llm_vs_codigo.csv", index=False, encoding="utf-8")

    df_cat = (
        df.groupby(["conjunto", "clave_enfoque", "categoria"], sort=False)
        .agg(n=("respuesta_correcta", "size"), respuesta_correcta=("respuesta_correcta", "mean"),
             llm_seleccion_exacta=("llm_seleccion_exacta", "mean"), llm_args_perfectos=("llm_args_perfectos", "mean"),
             pasos_llm_medios=("n_pasos_llm", "mean"), latencia_media_s=("latencia_ms", lambda s: s.mean() / 1000))
        .round(4)
        .reset_index()
    )
    df_cat.to_csv(config.REPORTES_DIR / "desempeno_por_categoria.csv", index=False, encoding="utf-8")

    val = df_resumen[df_resumen["conjunto"] == "validacion"].set_index("clave_enfoque")
    cols_prompt = ["respuesta_correcta", "llm_seleccion_turno1", "llm_seleccion_exacta", "llm_args_perfectos",
                   "llm_se_abstuvo_ood", "abstencion_correcta_ood", "falsa_abstencion_dominio",
                   "fidelidad_texto_llm", "pasos_llm_medios", "latencia_media_s"]
    df_prompt = pd.DataFrame({
        "metrica": cols_prompt,
        "v1": [val.loc["agente_v1", c] for c in cols_prompt],
        "v2": [val.loc["agente_v2", c] for c in cols_prompt],
    })
    df_prompt["diferencia_v2_menos_v1"] = (df_prompt["v2"] - df_prompt["v1"]).round(4)
    df_prompt.to_csv(config.REPORTES_DIR / "mejora_prompt_validacion.csv", index=False, encoding="utf-8")

    # Errores del agente en prueba, para el análisis de errores del README
    sub_err = df[(df["conjunto"] == "prueba") & (df["clave_enfoque"] == "agente") & (df["respuesta_correcta"] < 1)]
    sub_err[["id_pregunta", "categoria", "pregunta", "herramientas_esperadas", "herramientas_ejecutadas",
             "respuesta_final"]].to_csv(config.REPORTES_DIR / "errores_agente_prueba.csv", index=False, encoding="utf-8")

    bm = datos.cargar_instantanea_bm()
    pd.DataFrame(bm["resumen_cobertura"]).to_csv(config.REPORTES_DIR / "cobertura_indicadores_bm.csv", index=False, encoding="utf-8")
    datos.construir_dataframe_clima_anual().to_csv(
        config.REPORTES_DIR / "resumen_climatico_ciudades.csv", index=False, encoding="utf-8"
    )

    print("4/5 · Figuras...")
    rutas_figuras = graficos.generar_todas_las_figuras(df_resumen=df_resumen, df_cat=df_cat, df_detalle=df,
                                                       df_prompt=df_prompt)

    print("5/5 · Artefactos en models/ y reports/resultados.json...")
    esquemas = herramientas.obtener_esquemas_ollama(version)
    (config.MODELOS_DIR / "catalogo_herramientas.json").write_text(
        json.dumps(esquemas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fila_prueba = df_resumen[(df_resumen["conjunto"] == "prueba") & (df_resumen["clave_enfoque"] == "agente")].iloc[0]
    (config.MODELOS_DIR / "metadata_agente.json").write_text(json.dumps({
        "proyecto": "IA_06_agente_datos_abiertos",
        "version": "2.0.0",
        "modelo_llm": config.MODELO_LLM,
        "version_prompt": version,
        "max_pasos_llm": config.MAX_PASOS_AGENTE,
        "herramientas": [t["function"]["name"] for t in esquemas],
        "paises_soportados": list(config.PAISES),
        "indicadores_bm_soportados": list(config.INDICADORES_BM),
        "ciudades_clima_soportadas": list(config.CIUDADES_CLIMA),
        "metricas_prueba": {k: (None if pd.isna(v) else v) for k, v in fila_prueba.to_dict().items()},
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _limpio(df_: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(df_.to_json(orient="records", force_ascii=False))

    resultados = {
        "proyecto": "IA_06_agente_datos_abiertos",
        "modelo_llm": config.MODELO_LLM,
        "version_prompt_elegida": version,
        "n_validacion": len(conjuntos["validacion"]),
        "n_prueba": len(conjuntos["prueba"]),
        "datos_instantanea": {
            "observaciones_banco_mundial": int(sum(r["n_observaciones"] for r in bm["resumen_cobertura"])),
            "registros_diarios_open_meteo": int(sum(
                c["total_dias_historico"] for c in datos.cargar_instantanea_clima()["ciudades"].values()
            )),
        },
        "resumen_enfoques": _limpio(df_resumen),
        "atribucion_llm_vs_codigo": _limpio(df_atrib),
        "mejora_prompt_validacion": _limpio(df_prompt),
        "desempeno_por_categoria": _limpio(df_cat),
        "trazas_agente_prueba": trazas,
        "figuras_generadas": rutas_figuras,
    }
    (config.REPORTES_DIR / "resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("Evaluación finalizada.")
    return resultados


def podar_grabaciones() -> int:
    """Deja en data/grabacion_llm.jsonl solo las respuestas usadas por la evaluación (tras correrla)."""
    usadas = motor_agente.HUELLAS_USADAS
    lineas = config.GRABACION_LLM.read_text(encoding="utf-8").splitlines()
    conservadas = [ln for ln in lineas if ln.strip() and json.loads(ln).get("huella") in usadas]
    config.GRABACION_LLM.write_text("\n".join(conservadas) + "\n", encoding="utf-8")
    return len(lineas) - len(conservadas)


if __name__ == "__main__":
    ejecutar_pipeline_completo()
