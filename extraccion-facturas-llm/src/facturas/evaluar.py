"""
Evaluación reproducible (no llama al LLM: lee reports/predicciones_llm.jsonl, que deja `python -m facturas.correr_llm`).

    python -m facturas.evaluar            # todo: métricas, figuras y detalle
    python -m facturas.evaluar --ic       # solo agrega/recalcula los IC 95 % en evaluacion.json desde el detalle por factura

Los IC (bootstrap de facturas para campos e ítems, Wilson para facturas perfectas) importan sobre todo en la muestra
de 50 del LLM (10 por plantilla): una diferencia de 5 puntos entre métodos o plantillas no es distinguible del azar.

Métodos comparados:
  * regex          línea base sin LLM (las 200 facturas y la muestra de 50)
  * llm_1          LLM, solo el primer intento (sin validación ni reintento)
  * llm            LLM + validación + reintento con corrección: lo MEDIDO en la corrida
  * llm_rep        lo mismo + reparaciones deterministas (escritas viendo los errores de `llm`: cifra optimista)
  * cascada        regex primero; si su resultado NO pasa las validaciones de negocio, se usa el LLM.
                   Es el diseño de producción: lo barato donde funciona, lo caro donde hace falta.

Salidas: reports/evaluacion.json, reports/evaluacion_detalle.csv, reports/figures/*.png
"""
from __future__ import annotations

import csv
import json
import statistics as st
import time
from collections import defaultdict

from . import config
from .correr_llm import SALIDA, catalogo, textos
from .esquema import CAMPOS, Factura
from .metricas import comparar, f1_micro, ic95
from .regex_base import extraer_regex
from .reparaciones import reparar_dict
from .validaciones import errores, validar

COND = {"digital": "digital", "escaneada": "escaneada/girada", "rotada": "escaneada/girada"}
CAMPOS_VALIDABLES = ["ruc_emisor", "numero_factura", "numero_autorizacion", "fecha_emision",
                     "cliente_identificacion", "subtotal", "descuento", "iva", "total"]


def verdad(fid: str) -> dict:
    return json.loads((config.DIR_VERDAD / f"{fid}.json").read_text(encoding="utf-8"))


def resumen(filas: list[dict]) -> dict:
    n = len(filas)
    if not n:
        return {"n": 0}
    f1 = f1_micro(filas)
    return {"n": n,
            "exactitud_campos": round(sum(f["campos_ok"] for f in filas) / (n * len(CAMPOS)), 4),
            "por_campo": {c: round(sum(f[c] for f in filas) / n, 4) for c in CAMPOS},
            "items_f1": round(f1["f1"], 4), "items_precision": round(f1["precision"], 4),
            "items_recall": round(f1["recall"], 4),
            "facturas_perfectas": round(sum(f["factura_perfecta"] for f in filas) / n, 4),
            "ic95": ic95(filas)}


def por_grupo(filas: list[dict], clave: str) -> dict:
    g = defaultdict(list)
    for f in filas:
        g[f[clave]].append(f)
    return {k: resumen(v) for k, v in sorted(g.items())}


def abstencion(filas: list[dict], preds: dict[str, dict]) -> dict:
    """Campos que NO están impresos (verdad null): ¿el método devuelve null o inventa algo?"""
    total = inventa = 0
    for f in filas:
        v = verdad(f["id"])
        p = preds[f["id"]] or {}
        for c in CAMPOS:
            if v[c] is None:
                total += 1
                inventa += p.get(c) is not None
    return {"campos_ausentes": total, "devuelve_null": total - inventa, "inventa": inventa,
            "tasa_null": round((total - inventa) / total, 4) if total else None}


def semaforo(filas: list[dict], estados: dict[str, str]) -> dict:
    """¿La validación separa lo bueno de lo malo? 'malo' = algún campo validable o ítem incorrecto."""
    m = defaultdict(int)
    for f in filas:
        malo = not all(f[c] for c in CAMPOS_VALIDABLES) or f["items_f1"] < 1
        marcada = estados[f["id"]] in ("revisar", "error")
        m[("marcada" if marcada else "aprobada") + "_" + ("con_error" if malo else "correcta")] += 1
    m = dict(m)
    aprob = m.get("aprobada_correcta", 0) + m.get("aprobada_con_error", 0)
    con_err = m.get("marcada_con_error", 0) + m.get("aprobada_con_error", 0)
    m["errores_silenciosos_pct_aprobadas"] = round(m.get("aprobada_con_error", 0) / aprob, 4) if aprob else None
    m["errores_detectados_pct"] = round(m.get("marcada_con_error", 0) / con_err, 4) if con_err else None
    return m


def main():
    cat = {r["id"]: r for r in catalogo()}
    tx = textos()
    detalle = []

    # ---------------------------------------------------------------- regex en las 200
    preds_regex, estado_regex, t0 = {}, {}, time.perf_counter()
    for fid in cat:
        p = extraer_regex(tx[fid]["texto"])
        preds_regex[fid] = p.model_dump(mode="json")
        estado_regex[fid] = "revisar" if errores(validar(p)) else "valida"
    ms_regex = (time.perf_counter() - t0) / len(cat) * 1000

    def fila(fid, metodo, pred):
        r = cat[fid]
        f = comparar(pred, verdad(fid))
        f.update(id=fid, metodo=metodo, plantilla=r["plantilla"], condicion=COND[r["condicion"]],
                 faltantes="con faltantes" if r["faltantes"] else "completa")
        return f

    regex_200 = [fila(fid, "regex", preds_regex[fid]) for fid in cat]

    # ---------------------------------------------------------------- LLM en la muestra
    llm = [json.loads(ln) for ln in SALIDA.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [d["id"] for d in llm]
    por_id = {d["id"]: d for d in llm}
    preds = {"regex": {i: preds_regex[i] for i in ids},
             "llm_1": {i: por_id[i]["intentos"][0]["factura"] if por_id[i]["intentos"] else None for i in ids},
             "llm": {i: por_id[i]["factura"] for i in ids}}
    preds["llm_rep"] = {i: reparar_dict(preds["llm"][i]) for i in ids}
    preds["cascada"] = {i: preds_regex[i] if estado_regex[i] == "valida" else preds["llm_rep"][i] for i in ids}
    filas = {m: [fila(i, m, preds[m][i]) for i in ids] for m in preds}
    for fs in filas.values():
        detalle += fs

    estados_llm = {i: por_id[i]["estado"] for i in ids}
    estado_llm_1 = {}
    for i in ids:
        f1 = preds["llm_1"][i]
        estado_llm_1[i] = "error" if f1 is None else ("revisar" if errores(validar(Factura.model_validate(f1)))
                                                       else "valida")
    estados_casc = {i: "valida" if estado_regex[i] == "valida" else estados_llm[i] for i in ids}

    seg = [d["segundos_llm"] for d in llm]
    reint = [d["reintentos"] for d in llm]
    con_reint = [d for d in llm if d["reintentos"]]
    arreglados = sum(1 for d in con_reint if not d["intentos"][-1]["errores"])
    texto_ocr = [t["segundos"] for t in tx.values() if t["metodo"] == "ocr"]
    rotadas = [r for r in cat.values() if r["condicion"] == "rotada"]

    informe = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "modelo": config.MODELO,
        "datos": {"facturas": len(cat), "muestra_llm": len(ids),
                  "plantillas": sorted({r["plantilla"] for r in cat.values()}),
                  "escaneadas_o_giradas": sum(r["condicion"] != "digital" for r in cat.values()),
                  "con_faltantes": sum(bool(r["faltantes"]) for r in cat.values())},
        "texto": {"pypdf": sum(t["metodo"] == "pypdf" for t in tx.values()),
                  "ocr": len(texto_ocr), "ocr_segundos_medio": round(st.mean(texto_ocr), 2) if texto_ocr else None,
                  "rotaciones_detectadas": sum(1 for r in rotadas if tx[r["id"]]["rotacion"] == (360 - int(r["rotacion"])) % 360),
                  "rotadas": len(rotadas)},
        "regex_200": {"global": resumen(regex_200), "por_plantilla": por_grupo(regex_200, "plantilla"),
                      "por_condicion": por_grupo(regex_200, "condicion"), "ms_por_factura": round(ms_regex, 2),
                      "pasa_validacion": round(sum(e == "valida" for e in estado_regex.values()) / len(cat), 4)},
        "muestra_50": {m: {"global": resumen(fs), "por_plantilla": por_grupo(fs, "plantilla"),
                           "por_condicion": por_grupo(fs, "condicion"), "por_faltantes": por_grupo(fs, "faltantes"),
                           "abstencion": abstencion(fs, preds[m])} for m, fs in filas.items()},
        "validacion": {"llm_1": semaforo(filas["llm_1"], estado_llm_1), "llm": semaforo(filas["llm"], estados_llm),
                       "llm_rep": semaforo(filas["llm_rep"], estados_llm),
                       "regex": semaforo(filas["regex"], {i: estado_regex[i] for i in ids}),
                       "cascada": semaforo(filas["cascada"], estados_casc)},
        "reintentos": {"facturas_con_reintento": len(con_reint), "arregladas_por_reintento": arreglados,
                       "reintentos_medios": round(st.mean(reint), 3),
                       "estados_finales": {e: sum(d["estado"] == e for d in llm) for e in
                                           ("valida", "con_advertencias", "revisar", "error")}},
        "costo": {"llm_segundos_medio": round(st.mean(seg), 1), "llm_segundos_mediana": round(st.median(seg), 1),
                  "llm_segundos_p90": round(sorted(seg)[int(0.9 * (len(seg) - 1))], 1),
                  "llamadas_medias": round(st.mean(d["llamadas"] for d in llm), 2),
                  "tokens_entrada_medios": round(st.mean(d["tokens_entrada"] for d in llm)),
                  "tokens_salida_medios": round(st.mean(d["tokens_salida"] for d in llm)),
                  "cascada_usa_llm_pct_200": round(sum(e != "valida" for e in estado_regex.values()) / len(cat), 4),
                  "cascada_usa_llm_pct_muestra": round(sum(estado_regex[i] != "valida" for i in ids) / len(ids), 4)},
    }
    config.REPORTES.mkdir(exist_ok=True)
    (config.REPORTES / "evaluacion.json").write_text(json.dumps(informe, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    columnas = ["id", "metodo", "plantilla", "condicion", "faltantes", *CAMPOS, "items_real", "items_pred",
                "items_ok", "items_f1", "campos_ok", "factura_perfecta"]
    with open(config.REPORTES / "evaluacion_detalle.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(detalle + [dict(f, metodo="regex_200") for f in regex_200])
    from .graficos import graficar
    graficar(informe)
    imprimir(informe)
    return informe


def imprimir(inf: dict) -> None:
    m = inf["muestra_50"]
    print(f"Muestra LLM: {inf['datos']['muestra_llm']} facturas")
    print(f"{'método':10s} {'campos':>7s} {'ítems F1':>9s} {'perfectas':>10s}")
    for k in ("regex", "llm_1", "llm", "llm_rep", "cascada"):
        g = m[k]["global"]
        print(f"{k:10s} {g['exactitud_campos']:7.1%} {g['items_f1']:9.1%} {g['facturas_perfectas']:10.1%}")
    g = inf["regex_200"]["global"]
    print(f"regex (200) {g['exactitud_campos']:6.1%} {g['items_f1']:9.1%} {g['facturas_perfectas']:10.1%}")
    print("validación LLM:", inf["validacion"]["llm"])
    print("reintentos:", inf["reintentos"])
    print("costo:", inf["costo"])


def solo_ic() -> None:
    """Agrega ic95 a cada resumen de reports/evaluacion.json usando reports/evaluacion_detalle.csv (sin reevaluar)."""
    informe = json.loads((config.REPORTES / "evaluacion.json").read_text(encoding="utf-8"))
    with open(config.REPORTES / "evaluacion_detalle.csv", newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    claves = {"global": lambda f: True, "por_plantilla": "plantilla", "por_condicion": "condicion",
              "por_faltantes": "faltantes"}

    def poner(bloque: dict, sel: list[dict]) -> None:
        for nombre, clave in claves.items():
            if nombre not in bloque:
                continue
            if nombre == "global":
                bloque["global"]["ic95"] = ic95(sel)
            else:
                for valor, resumen_g in bloque[nombre].items():
                    resumen_g["ic95"] = ic95([f for f in sel if f[clave] == valor])

    poner(informe["regex_200"], [f for f in filas if f["metodo"] == "regex_200"])
    for m, bloque in informe["muestra_50"].items():
        poner(bloque, [f for f in filas if f["metodo"] == m])
    (config.REPORTES / "evaluacion.json").write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    g = informe["muestra_50"]["llm"]["global"]
    print("IC 95 % LLM (n = 50):", g["ic95"])


if __name__ == "__main__":
    import sys
    solo_ic() if "--ic" in sys.argv[1:] else main()
