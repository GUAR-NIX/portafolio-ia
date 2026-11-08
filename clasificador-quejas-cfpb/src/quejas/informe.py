"""Genera README.md, MODEL_CARD.md y docs/GUIA_PASO_A_PASO.md a partir de reports/resultados.json.

Las plantillas viven en scripts/plantillas/ y usan marcadores {{clave}}. Ninguna cifra del README se
escribe a mano: si cambia la evaluación, se vuelve a generar con

    python -m quejas.informe
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config

PLANTILLAS = config.RAIZ / "scripts" / "plantillas"
DESTINOS = {
    "README.md": config.RAIZ / "README.md",
    "MODEL_CARD.md": config.RAIZ / "MODEL_CARD.md",
    "GUIA_PASO_A_PASO.md": config.RAIZ / "docs" / "GUIA_PASO_A_PASO.md",
}

CLAVES_ENFOQUE = {
    "1. TF-IDF + LR (25.722 ej.)": "tfidf",
    "1a. TF-IDF + LR (1.800 ej.)": "tfidf_p",
    "2. Zero-Shot prototipos bge-m3": "zs",
    "3. bge-m3 + LR (1.800 ej.)": "emb",
    "4. LLM zero-shot (qwen2.5:7b)": "llm0",
    "5. LLM few-shot (qwen2.5:7b, 6 ej.)": "llm1",
    "6. Cascada TF-IDF → LLM zero-shot": "casc0",
    "7. Cascada TF-IDF → LLM few-shot": "casc1",
}


def n(v: float, dec: int = 3) -> str:
    """Número con formato español: 1.234,5."""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, dec: int = 1) -> str:
    return f"{n(v * 100.0, dec)} %"


def entero(v: float) -> str:
    return n(v, 0)


def ms(v: float) -> str:
    if v < 1:
        return f"{n(v, 2)} ms"
    if v < 1000:
        return f"{n(v, 0)} ms"
    return f"{n(v / 1000.0, 1)} s"


def _tabla(encabezado: list[str], filas: list[list[str]], alinear: str | None = None) -> str:
    alinear = alinear or "l" + "r" * (len(encabezado) - 1)
    sep = ["---" if a == "l" else "---:" for a in alinear]
    lineas = ["| " + " | ".join(encabezado) + " |", "| " + " | ".join(sep) + " |"]
    lineas += ["| " + " | ".join(f) + " |" for f in filas]
    return "\n".join(lineas)


def valores(r: dict[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}
    md = r["metadata"]
    tp = md["tamanos_particion"]
    for k, val in tp.items():
        v[f"n_{k}"] = entero(val)
    v["fecha_min"], v["fecha_max"] = md["fechas"]["fecha_min"], md["fechas"]["fecha_max"]
    v["n_parquet"] = entero(r["resumen_fuente"]["filas_totales_parquet"])
    v["pct_urgente_corpus"] = n(r["resumen_fuente"]["pct_urgente"], 1) + " %"

    # --- Comparación principal (mismo conjunto común) ---
    filas = []
    for fila in r["comparacion_enfoques"]:
        c = CLAVES_ENFOQUE[fila["enfoque"]]
        v[f"f1_{c}"] = n(fila["f1_macro"])
        v[f"f1ic_{c}"] = f"{n(fila['f1_ic95_bajo'])}–{n(fila['f1_ic95_alto'])}"
        v[f"acc_{c}"] = pct(fila["exactitud"])
        v[f"accic_{c}"] = f"{pct(fila['exactitud_ic95_bajo'])}–{pct(fila['exactitud_ic95_alto'])}"
        v[f"top2_{c}"] = pct(fila["exactitud_top2"])
        v[f"brier_{c}"] = n(fila["brier_multiclase"])
        v[f"ms_{c}"] = ms(fila["ms_por_queja"])
        filas.append([
            fila["enfoque"].split(". ", 1)[1], f"**{n(fila['f1_macro'])}** ({v[f'f1ic_{c}']})",
            f"{pct(fila['exactitud'])} ({v[f'accic_{c}']})", pct(fila["exactitud_top2"]),
            n(fila["brier_multiclase"]), ms(fila["ms_por_queja"]),
        ])
    v["tabla_principal"] = _tabla(
        ["Enfoque", "Macro-F1 (IC 95 %)", "Exactitud (IC 95 %)", "Top-2", "Brier", "Latencia CPU"], filas)

    # --- Diferencias pareadas ---
    filas = []
    for d in r["diferencias_pareadas"]:
        a, b = CLAVES_ENFOQUE[d["enfoque_a"]], CLAVES_ENFOQUE[d["enfoque_b"]]
        v[f"dif_{a}_{b}"] = f"{'+' if d['diferencia'] >= 0 else '−'}{n(abs(d['diferencia']))}"
        v[f"dific_{a}_{b}"] = f"{n(d['ic95_bajo'])} a {n(d['ic95_alto'])}"
        v[f"dist_{a}_{b}"] = "distinguible" if d["distinguible"] else "no distinguible"
        filas.append([
            f"{d['enfoque_a'].split('. ', 1)[1]} − {d['enfoque_b'].split('. ', 1)[1]}",
            v[f"dif_{a}_{b}"], f"[{v[f'dific_{a}_{b}']}]", "sí" if d["distinguible"] else "**no**",
        ])
    v["tabla_diferencias"] = _tabla(["Comparación (Macro-F1 A − B)", "Diferencia", "IC 95 % pareado", "¿Distinguible?"], filas)

    # --- Cascada ---
    v["tau_cascada"] = n(r["cascada"]["umbral"])
    v["pct_al_llm"] = n(r["cascada"]["pct_al_llm_test"], 1) + " %"
    v["fraccion_objetivo"] = n(md["fraccion_al_llm_objetivo"] * 100, 0) + " %"

    # --- Curva few-shot ---
    filas = []
    for f in r["curva_few_shot"]:
        filas.append([str(f["ejemplos_por_clase"]), entero(f["total_ejemplos"]),
                      f"{n(f['f1_macro_bge_m3_media'])} ± {n(f['f1_macro_bge_m3_std'])}",
                      f"{n(f['f1_macro_tfidf_media'])} ± {n(f['f1_macro_tfidf_std'])}"])
        v[f"fs_emb_{f['ejemplos_por_clase']}"] = n(f["f1_macro_bge_m3_media"])
        v[f"fs_tf_{f['ejemplos_por_clase']}"] = n(f["f1_macro_tfidf_media"])
    v["tabla_few_shot"] = _tabla(["Ejemplos por mesa", "Total", "bge-m3 + LR", "TF-IDF + LR"], filas)

    # --- Señales de confianza del LLM ---
    auto = r["confianza_autodeclarada"]
    v["auto_n"], v["auto_099"] = str(auto.get("n_respuestas", 0)), str(auto.get("n_con_0_99", 0))
    filas = []
    senales = r["senales_confianza_llm"]
    for s in dict.fromkeys(x["senal"] for x in senales):
        fila = [s]
        for llm in ("4. LLM zero-shot (qwen2.5:7b)", "5. LLM few-shot (qwen2.5:7b, 6 ej.)"):
            x = next(e for e in senales if e["senal"] == s and e["llm"] == llm)
            fila.append(n(x["auroc_acierto_vs_error"], 2))
        filas.append(fila)
        clave = {"Probabilidad de la ficha (logprobs del LLM)": "logprob", "Acuerdo zero-shot = few-shot": "acuerdo_modos",
                 "Acuerdo LLM = TF-IDF": "acuerdo_tfidf", "Probabilidad calibrada de TF-IDF": "prob_tfidf"}[s]
        v[f"auroc_{clave}_llm0"], v[f"auroc_{clave}_llm1"] = fila[1], fila[2]
    v["tabla_senales"] = _tabla(["Señal", "AUROC (LLM zero-shot)", "AUROC (LLM few-shot)"], filas)
    lp = next(e for e in senales if e["senal"].startswith("Probabilidad de la ficha") and e["llm"].startswith("4."))
    v["pct_logprob_099_llm0"] = n(lp["pct_con_senal_maxima"], 1) + " %"
    ac = r["acuerdo_zero_few"]
    v["pct_acuerdo_modos"] = n(ac["pct_acuerdo_zero_few"], 1) + " %"
    v["acc_si_acuerdan"] = pct(ac["exactitud_few_si_acuerdan"])
    v["acc_si_discrepan"] = pct(ac["exactitud_few_si_discrepan"])
    v["n_discrepan"] = str(ac["n_discrepan"])
    v["llm_invalidas"] = str(r["llm_respuestas_invalidas"])

    # --- Confusiones del LLM ---
    conf = [c for c in r["confusiones_principales"] if c["enfoque"].startswith("4.")][:3]
    v["confusiones_llm0"] = "; ".join(
        f"{config.ETIQUETAS_CATEGORIA[c['real']]} → {config.ETIQUETAS_CATEGORIA[c['predicha']]} ({c['n']} quejas, {n(c['pct_de_errores'], 0)} % de sus errores)"
        for c in conf)
    conf_t = [c for c in r["confusiones_principales"] if c["enfoque"].startswith("1.")][:3]
    v["confusiones_tfidf"] = "; ".join(
        f"{config.ETIQUETAS_CATEGORIA[c['real']]} → {config.ETIQUETAS_CATEGORIA[c['predicha']]} ({c['n']})" for c in conf_t)

    # --- Deriva ---
    vieja, nueva = r["deriva_temporal"]
    v["f1_deriva_vieja"], v["f1_deriva_nueva"] = n(vieja["f1_macro"]), n(nueva["f1_macro"])
    v["f1ic_deriva_vieja"] = f"{n(vieja['f1_ic95_bajo'])}–{n(vieja['f1_ic95_alto'])}"
    v["f1ic_deriva_nueva"] = f"{n(nueva['f1_ic95_bajo'])}–{n(nueva['f1_ic95_alto'])}"
    v["acc_deriva_vieja"], v["acc_deriva_nueva"] = pct(vieja["exactitud"]), pct(nueva["exactitud"])
    lo, hi = r["deriva_diferencia_ic95"]
    v["dif_deriva"] = n(nueva["f1_macro"] - vieja["f1_macro"])
    v["dif_deriva_ic"] = f"{n(lo)} a {n(hi)}"
    v["tabla_trimestres"] = _tabla(["Trimestre de recepción", "Quejas", "Macro-F1", "Exactitud"],
                                   [[t["trimestre"], entero(t["n_quejas"]), n(t["f1_macro"]), pct(t["exactitud"])]
                                    for t in r["deriva_por_trimestre"]])
    v["tabla_deriva_mesa"] = _tabla(["Mesa", "F1 misma época", "F1 recientes", "Cambio"],
                                    [[config.ETIQUETAS_CATEGORIA[f["categoria"]], n(f["f1_misma_epoca"]), n(f["f1_recientes"]),
                                      f"{'+' if f['f1_recientes'] >= f['f1_misma_epoca'] else '−'}{n(abs(f['f1_recientes'] - f['f1_misma_epoca']))}"]
                                     for f in r["deriva_f1_por_mesa"]])

    # --- Calibración ---
    for c in r["calibracion"]:
        k = CLAVES_ENFOQUE[c["enfoque"]]
        v[f"brier_sin_{k}"] = n(c["brier_sin_calibrar"])
        v[f"ece_sin_{k}"] = n(c["ece_sin_calibrar"])
        v[f"confmedia_sin_{k}"] = pct(c["confianza_media_sin_calibrar"])
        v[f"accc_{k}"] = pct(c["exactitud"])
        if c["brier_calibrado"] == c["brier_calibrado"]:  # no NaN
            v[f"brier_cal_{k}"] = n(c["brier_calibrado"])
            v[f"ece_cal_{k}"] = n(c["ece_calibrado"])
            v[f"temp_{k}"] = n(c["temperatura"], 2)
            v[f"confmedia_cal_{k}"] = pct(c["confianza_media_calibrado"])

    # --- Abstención ---
    for a in r["abstencion_umbral_validacion"]:
        k = CLAVES_ENFOQUE[a["enfoque"]]
        v[f"tau_abst_{k}"] = n(a["umbral"], 2)
        v[f"cob_abst_{k}"] = n(a["cobertura_automatizada_pct"], 1) + " %"
        v[f"accauto_abst_{k}"] = pct(a["exactitud_en_automatizadas"])
        v[f"acc0_abst_{k}"] = pct(a["exactitud_sin_abstencion"])
        v[f"ahorro_abst_{k}"] = n(a["ahorro_pct"], 1) + " %"

    # --- Urgencia ---
    filas = []
    for u in r["urgencia"]:
        filas.append([u["enfoque"].split(". ", 1)[-1], "reciente completo" if u["conjunto"] == "test_reciente" else "común",
                      f"{entero(u['n_quejas'])} ({u['n_urgentes']})", pct(u["precision"]),
                      f"{pct(u['exhaustividad'])} ({pct(u['exhaustividad_ic95_bajo'], 0)}–{pct(u['exhaustividad_ic95_alto'], 0)})",
                      f"**{n(u['f1'])}** ({n(u['f1_ic95_bajo'])}–{n(u['f1_ic95_alto'])})",
                      n(u["pr_auc"]) if u["pr_auc"] == u["pr_auc"] else "—"])
    v["tabla_urgencia"] = _tabla(["Detector", "Conjunto", "Quejas (urgentes)", "Precisión", "Exhaustividad (IC 95 %)",
                                  "F1 (IC 95 %)", "PR-AUC"], filas, "llrrrrr")
    for u in r["urgencia"]:
        k = {"Reglas (palabras clave)": "reglas", "TF-IDF + LR (umbral en val)": "tfidf"}.get(u["enfoque"]) or CLAVES_ENFOQUE[u["enfoque"]]
        suf = "rec" if u["conjunto"] == "test_reciente" else "co"
        v[f"urg_f1_{k}_{suf}"] = n(u["f1"])
        v[f"urg_rec_{k}_{suf}"] = pct(u["exhaustividad"])
        v[f"urg_prec_{k}_{suf}"] = pct(u["precision"])
        v[f"urg_n_{suf}"] = entero(u["n_quejas"])
        v[f"urg_pos_{suf}"] = entero(u["n_urgentes"])
    v["umbral_urgencia"] = n(md["umbral_urgencia"], 2)
    nombres_tipo = {"fraude": "Fraude / robo de identidad", "ejecucion_hipotecaria": "Ejecución hipotecaria / modificación",
                    "reposesion": "Reposesión de vehículo", "amenaza_legal_embargo": "Amenaza legal / embargo"}
    v["tabla_urgencia_tipo"] = _tabla(["Tipo (motivo CFPB)", "Quejas urgentes en test", "Exhaustividad TF-IDF", "Exhaustividad reglas"],
                                      [[nombres_tipo[t["tipo_urgencia"]], entero(t["n"]), pct(t["exhaustividad_tfidf"]),
                                        pct(t["exhaustividad_reglas"])] for t in r["urgencia_por_tipo"]])

    # --- Español ---
    filas = []
    for e in r["transferencia_espanol"]:
        k = CLAVES_ENFOQUE[e["enfoque"]]
        v[f"es_acc_{k}"] = pct(e["exactitud_espanol"])
        v[f"es_aciertos_{k}"] = str(e["aciertos_espanol"])
        v[f"en_aciertos_{k}"] = str(e["aciertos_ingles"])
        v[f"esic_{k}"] = f"{pct(e['exactitud_espanol_ic95_bajo'], 0)}–{pct(e['exactitud_espanol_ic95_alto'], 0)}"
        filas.append([e["enfoque"].split(". ", 1)[1], f"{e['aciertos_ingles']}/{e['n_quejas']}",
                      f"{e['aciertos_espanol']}/{e['n_quejas']}", f"{pct(e['exactitud_espanol'])} ({v[f'esic_{k}']})"])
    v["tabla_espanol"] = _tabla(["Enfoque (sin entrenar en español)", "Aciertos en inglés", "Aciertos en español",
                                 "Exactitud en español (IC 95 %)"], filas)

    pal = r.get("espanol_pct_al_llm_en_cascada", {})
    if pal:
        v["pct_es_al_llm"] = n(pal["espanol"], 0) + " %"
        v["pct_en_al_llm"] = n(pal["ingles"], 0) + " %"

    # --- Mapeo producto CFPB → mesa (agrupación propia) ---
    import descargar_datos as dd  # noqa: PLC0415  (script en la raíz del proyecto)

    dist = r["resumen_fuente"]["distribucion_original_por_producto"]
    filas = [[f"`{p}`", config.ETIQUETAS_CATEGORIA[m], entero(dist.get(p, 0))]
             for p, m in sorted(dd.MAPEO_PRODUCTO.items(), key=lambda x: (config.CATEGORIAS.index(x[1]), -dist.get(x[0], 0)))]
    filas += [[f"`{p}`", "*(excluido)*", entero(dist.get(p, 0))] for p in ("Other financial service", "Virtual currency")]
    v["tabla_mapeo"] = _tabla(["Producto del CFPB", "Mesa (agrupación propia)", "Filas en el Parquet"], filas, "llr")
    v["tabla_particiones"] = _tabla(["Mesa", "train", "reserva misma época", "val", "test"],
                                    [[config.ETIQUETAS_CATEGORIA[p["categoria"]], entero(p["train"]), entero(p["test_misma_epoca"]),
                                      entero(p["val"]), entero(p["test"])] for p in r["particiones_temporales"]])
    return v


def rellenar(plantilla: str, v: dict[str, str]) -> str:
    def reemplazo(m: re.Match) -> str:
        clave = m.group(1)
        if clave not in v:
            raise KeyError(f"Marcador sin valor en la plantilla: {{{{{clave}}}}}")
        return v[clave]

    return re.sub(r"\{\{(\w+)\}\}", reemplazo, plantilla)


def generar(ruta_resultados: Path = config.REPORTES / "resultados.json") -> dict[str, str]:
    import sys

    sys.path.insert(0, str(config.RAIZ))
    r = json.loads(Path(ruta_resultados).read_text(encoding="utf-8"))
    v = valores(r)
    return {nombre: rellenar((PLANTILLAS / nombre).read_text(encoding="utf-8"), v) for nombre in DESTINOS}


def main() -> None:
    for nombre, texto in generar().items():
        DESTINOS[nombre].write_text(texto, encoding="utf-8")
        print(f"[OK] {DESTINOS[nombre].relative_to(config.RAIZ)}")


if __name__ == "__main__":
    main()
