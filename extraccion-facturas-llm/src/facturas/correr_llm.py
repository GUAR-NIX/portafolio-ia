"""
Corre el extractor con LLM sobre la muestra de evaluación y guarda cada resultado apenas termina
(reports/predicciones_llm.jsonl). Es reanudable: las facturas ya procesadas se saltan y, con FACTURAS_GRABAR=1,
las llamadas ya hechas se reutilizan desde data/grabacion_llm.jsonl.

    python -m facturas.correr_llm                # muestra de 50 facturas (10 por plantilla), ~1-2 h en CPU
    python -m facturas.correr_llm --n-por-plantilla 4

Muestra estratificada: por plantilla, las 2 facturas versionadas en data/muestras + hasta 3 escaneadas/giradas +
hasta 2 con campos faltantes + digitales al azar hasta completar. (La regex se evalúa sobre las 200.)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

from . import config
from .extractor import extraer_de_texto
from .llm import LLM, ErrorLLM
from .texto import extraer_texto

SALIDA = config.REPORTES / "predicciones_llm.jsonl"
TEXTOS = config.RAIZ / "data" / "textos.jsonl"


def catalogo() -> list[dict]:
    with open(config.CATALOGO, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def muestra_llm(n_por_plantilla: int = 10, semilla: int = config.SEMILLA) -> list[str]:
    rng = random.Random(semilla)
    muestras = {p.stem for p in config.DIR_MUESTRAS.glob("*.pdf")}
    elegidas = []
    filas = catalogo()
    for plantilla in sorted({r["plantilla"] for r in filas}):
        de_p = [r for r in filas if r["plantilla"] == plantilla]
        sel = [r["id"] for r in de_p if r["id"] in muestras]
        ruidosas = [r["id"] for r in de_p if r["condicion"] != "digital" and r["id"] not in sel]
        faltan = [r["id"] for r in de_p if r["faltantes"] and r["id"] not in sel and r["id"] not in ruidosas]
        sel += rng.sample(ruidosas, min(3, len(ruidosas)))
        sel += rng.sample(faltan, min(2, len(faltan)))
        resto = [r["id"] for r in de_p if r["id"] not in sel and r["condicion"] == "digital" and not r["faltantes"]]
        sel += rng.sample(resto, max(0, n_por_plantilla - len(sel)))
        elegidas.append(sorted(sel[:n_por_plantilla]))
    # intercaladas por plantilla: si la corrida se corta, todas las plantillas quedan igual de cubiertas
    return [ids[k] for k in range(n_por_plantilla) for ids in elegidas if k < len(ids)]


def textos(ids: list[str] | None = None) -> dict[str, dict]:
    """Texto de cada factura (pypdf u OCR). Se guarda en data/textos.jsonl para no repetir el OCR."""
    previos = {}
    if TEXTOS.exists():
        for ln in TEXTOS.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            previos[d["id"]] = d
    ids = ids or [r["id"] for r in catalogo()]
    faltan = [i for i in ids if i not in previos]
    for fid in faltan:
        t0 = time.perf_counter()
        t = extraer_texto(config.DIR_FACTURAS / f"{fid}.pdf")
        previos[fid] = {"id": fid, "metodo": t.metodo, "rotacion": t.rotacion, "avisos": t.avisos,
                        "segundos": round(time.perf_counter() - t0, 2), "texto": t.texto}
    if faltan:
        with open(TEXTOS, "w", encoding="utf-8") as fh:
            for fid in sorted(previos):
                fh.write(json.dumps(previos[fid], ensure_ascii=False) + "\n")
    return previos


def hechas() -> set[str]:
    if not SALIDA.exists():
        return set()
    return {json.loads(ln)["id"] for ln in SALIDA.read_text(encoding="utf-8").splitlines() if ln.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-por-plantilla", type=int, default=10)
    a = ap.parse_args()
    os.environ.setdefault("FACTURAS_GRABAR", "1")
    ids = muestra_llm(a.n_por_plantilla)
    tx = textos()
    ya = hechas()
    pendientes = [i for i in ids if i not in ya]
    print(f"Muestra: {len(ids)} facturas · ya hechas {len(ya & set(ids))} · pendientes {len(pendientes)}", flush=True)
    llm = LLM()
    t_ini = time.perf_counter()
    for k, fid in enumerate(pendientes, 1):
        u0 = (llm.uso.llamadas, llm.uso.tokens_entrada, llm.uso.tokens_salida)
        for intento in range(3):                 # un 500 pasajero de Ollama no debe tumbar una corrida de horas
            try:
                r = extraer_de_texto(tx[fid]["texto"], llm, con_reparaciones=False)   # se guarda lo medido
                break
            except ErrorLLM as e:
                print(f"  {fid}: {e} → reintento en 30 s", flush=True)
                time.sleep(30)
        else:
            print(f"  {fid}: se omite tras 3 fallos de Ollama", flush=True)
            continue
        d = {"id": fid, **r.dict(),
             "intentos": [{"numero": i.numero, "factura": i.factura, "errores": i.errores, "segundos": i.segundos}
                          for i in r.intentos],
             "llamadas": llm.uso.llamadas - u0[0], "tokens_entrada": llm.uso.tokens_entrada - u0[1],
             "tokens_salida": llm.uso.tokens_salida - u0[2]}
        d["metodo_texto"] = tx[fid]["metodo"]
        with open(SALIDA, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        trans = time.perf_counter() - t_ini
        print(f"[{k}/{len(pendientes)}] {fid} {r.estado:16s} reintentos={r.reintentos} {r.segundos_llm:5.0f}s "
              f"· transcurrido {trans / 60:.0f} min · faltan ~{trans / k * (len(pendientes) - k) / 60:.0f} min",
              flush=True)


if __name__ == "__main__":
    main()
