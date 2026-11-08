"""Reproduce toda la evaluación desde las respuestas grabadas (sin Ollama) y escribe reports/.

    python -m kpis.ejecutar_evaluacion
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import comparaciones, config, evaluacion as ev, generadores, graficos

ENF_LLM = ("libre", "estructurado", "verificado")


def _json_ic(par) -> list[float]:
    return [round(par[0], 4), round(par[1], 4)]


def correr_todo() -> dict:
    version = ev.elegir_version_prompt()
    resultados: list[tuple[str, dict]] = []
    for v in generadores.VERSIONES_PROMPT:   # ajuste del prompt: solo validación
        for enf in ("estructurado", "verificado"):
            resultados += [("validacion", r) for r in ev.correr(enf, config.VALIDACION, v)]
    resultados += [("validacion", r) for r in ev.correr("libre", config.VALIDACION, version)]
    resultados += [("validacion", r) for r in ev.correr("plantilla", config.VALIDACION, None)]
    for enf in ENF_LLM:
        resultados += [("prueba", r) for r in ev.correr(enf, config.PRUEBA, version)]
    resultados += [("prueba", r) for r in ev.correr("plantilla", config.PRUEBA, None)]
    return {"version": version, "resultados": resultados}


def main() -> dict:
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    corrida = correr_todo()
    version, resultados = corrida["version"], corrida["resultados"]
    detalle = pd.DataFrame([ev.fila_informe(r, c) for c, r in resultados])
    cifras = pd.DataFrame([f for c, r in resultados for f in ev.filas_cifras(r, c)])
    comps = pd.DataFrame([{"conjunto": c, "enfoque": r["enfoque"], "version": r["version"] or "-", "pais": r["pais"],
                           **x} for c, r in resultados for x in comparaciones.revisar_resultado(r)])
    intentos = pd.DataFrame([{"conjunto": c, "pais": r["pais"], "version": r["version"], "intento": i["intento"],
                              "n_cifras": i["n_cifras"], "errores": round((i["tasa_alucinacion"] or 0) *
                                                                          (i["n_cifras"] or 0)),
                              "json_valido": i["json_valido"]}
                             for c, r in resultados if r["enfoque"] == "verificado" for i in r["intentos"]])
    res = ev.resumen(detalle)
    res.attrs["version_elegida"] = version

    # Comparaciones pareadas en prueba (mismos 12 informes)
    prueba = detalle[detalle["conjunto"] == "prueba"]
    por = {e: prueba[prueba["enfoque"] == e] for e in generadores.ENFOQUES}
    pares = [("verificado", "estructurado"), ("estructurado", "libre"), ("verificado", "libre"),
             ("plantilla", "verificado")]
    pareadas = []
    for a, b in pares:
        for m in ("tasa_alucinacion", "cobertura"):
            d, lo, hi = ev.bootstrap_pareado(por[a], por[b], ev.METRICAS[m])
            pareadas.append({"a": a, "b": b, "metrica": m, "diferencia": d, "ic_inf": lo, "ic_sup": hi})
    pareadas = pd.DataFrame(pareadas)

    # Legibilidad (rúbrica a mano, a ciegas)
    rub = None
    if config.RUBRICA.exists():
        rub = pd.read_csv(config.RUBRICA)
        mapa = json.loads((config.DATA / "rubrica_mapa.json").read_text(encoding="utf-8"))
        rub["enfoque"] = rub["id"].map(lambda i: mapa[i]["enfoque"])
        rub["pais"] = rub["id"].map(lambda i: mapa[i]["pais"])
        rub["total"] = rub[["claridad", "concision", "utilidad", "tono"]].mean(axis=1)

    # Guardar tablas
    detalle.to_csv(config.REPORTS / "detalle_informes.csv", index=False, float_format="%.4f")
    cifras.to_csv(config.REPORTS / "detalle_cifras.csv", index=False)
    comps.to_csv(config.REPORTS / "detalle_comparaciones.csv", index=False, float_format="%.3f")
    intentos.to_csv(config.REPORTS / "reintentos.csv", index=False)
    pareadas.to_csv(config.REPORTS / "comparaciones_pareadas.csv", index=False, float_format="%.4f")
    res_csv = res.copy()
    for col in [c for c in res_csv.columns if c.endswith("_ic")]:
        res_csv[col] = res_csv[col].map(lambda t: f"{t[0]:.4f}–{t[1]:.4f}")
    res_csv.to_csv(config.REPORTS / "resumen_enfoques.csv", index=False, float_format="%.4f")

    textos = [{"conjunto": c, "enfoque": r["enfoque"], "version": r["version"], "pais": r["pais"], "anio": r["anio"],
               "texto": r["texto"]} for c, r in resultados
              if r["enfoque"] == "plantilla" or r["version"] in (version, None)]
    (config.REPORTS / "informes_generados.json").write_text(json.dumps(textos, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")

    figuras = [graficos.ecuador_vs_region(), graficos.mapa_anomalias(), graficos.alucinacion_y_cobertura(res),
               graficos.tipos_de_error(cifras, comps),graficos.reintentos(intentos), graficos.mejora_prompt(res),
               graficos.tiempo(detalle)]
    if rub is not None:
        figuras.append(graficos.legibilidad(rub))

    # resultados.json: cifras clave (el README y la guía salen de aquí)
    def fila(conj, enf, ver=None):
        sel = res[(res["conjunto"] == conj) & (res["enfoque"] == enf)]
        if ver is not None:
            sel = sel[sel["version"] == ver]
        f = sel.iloc[0]
        out = {k: (round(float(f[k]), 4) if isinstance(f[k], (float, np.floating)) else
                   (int(f[k]) if isinstance(f[k], (int, np.integer)) else f[k]))
               for k in ["informes", "cifras", "correctas", "incorrectas", "sin_fuente", "cifras_por_informe",
                         "llamadas_llm", "palabras", "referencias_ok", "refs_en_linea", "comparaciones",
                         "comparaciones_mal_n", *ev.METRICAS]}
        for m in ev.METRICAS:
            if f"{m}_ic" in f:
                out[f"{m}_ic"] = _json_ic(f[f"{m}_ic"])
        return out

    salida = {
        "version_prompt_elegida": version, "regla_eleccion": ev.REGLA_ELECCION,
        "modelo": config.MODELO,
        "descargado": json.loads(config.METADATOS.read_text(encoding="utf-8"))["descargado"],
        "n_validacion": len(config.VALIDACION), "n_prueba": len(config.PRUEBA),
        "validacion": {f"{e}_{v}": fila("validacion", e, v) for e in ("estructurado", "verificado")
                       for v in generadores.VERSIONES_PROMPT} |
                      {"libre": fila("validacion", "libre"), "plantilla": fila("validacion", "plantilla")},
        "prueba": {e: fila("prueba", e) for e in generadores.ENFOQUES},
        "pareadas_prueba": pareadas.round(4).to_dict(orient="records"),
        "reintentos_prueba": {
            "informes_que_reintentan": int((intentos[(intentos.conjunto == "prueba")].groupby("pais")["intento"]
                                            .max() > 0).sum()),
            "informes_aprobados_primer_intento": int(prueba[prueba.enfoque == "estructurado"]["aprobado"].sum()),
            "informes_aprobados_final": int(prueba[prueba.enfoque == "verificado"]["aprobado"].sum()),
        },
        "marcas_unidad_prueba": {e: int(prueba[prueba.enfoque == e]["n_marcas_unidad"].sum())
                                 for e in generadores.ENFOQUES},
        "atribucion_debil_prueba": {e: int(prueba[prueba.enfoque == e]["n_atribucion_debil"].sum())
                                    for e in generadores.ENFOQUES},
        "figuras": figuras,
    }
    cp = comps[comps["conjunto"] == "prueba"]
    salida["lecturas_mal_por_tipo_prueba"] = {
        e: {t: int(((cp["enfoque"] == e) & (cp["tipo"] == t) & ~cp["correcta"].astype(bool)).sum())
            for t in ("frente_a_region", "frente_al_anio_anterior", "pib_nominal_como_crecimiento")}
        for e in generadores.ENFOQUES}
    ruta_aud = config.DATA / "auditoria_verificador.csv"
    if ruta_aud.exists():
        aud = pd.read_csv(ruta_aud)
        malas = aud[aud["estado_verificador"] != "correcta"]
        buenas = aud[aud["estado_verificador"] == "correcta"]
        salida["auditoria"] = {
            "marcadas_mal": len(malas), "marcadas_mal_confirmadas": int((malas["juicio_humano"] != "correcta").sum()),
            "muestra_correctas": len(buenas),
            "muestra_correctas_confirmadas": int((buenas["juicio_humano"] == "correcta").sum()),
            "enganosas_en_muestra": int(aud["nota"].fillna("").str.startswith("cifra correcta, lectura").sum()),
        }
        salida["auditoria"]["precision_marcadas_mal"] = round(
            salida["auditoria"]["marcadas_mal_confirmadas"] / max(len(malas), 1), 4)
        salida["auditoria"]["precision_correctas"] = round(
            salida["auditoria"]["muestra_correctas_confirmadas"] / max(len(buenas), 1), 4)
    if rub is not None:
        g = rub.groupby("enfoque")[["claridad", "concision", "utilidad", "tono", "total"]].mean().round(2)
        salida["legibilidad"] = {"n_informes": int(rub["pais"].nunique()), **g.to_dict(orient="index")}
    (config.REPORTS / "resultados.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps({k: salida[k] for k in ("version_prompt_elegida", "prueba")}, ensure_ascii=False, indent=1))
    return salida


if __name__ == "__main__":
    main()
