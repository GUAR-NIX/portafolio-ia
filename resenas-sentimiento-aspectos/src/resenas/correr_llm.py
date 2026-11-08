"""
Corre el LLM sobre las reseñas etiquetadas y guarda cada resultado en cuanto llega (reanudable).

    RESENAS_GRABAR=1 python -m resenas.correr_llm            # 40 de dev + test (320) + español (80), ~2,5 h
    python -m resenas.correr_llm --conjunto dev --n 40

Cada llamada queda además en data/grabacion_llm.jsonl: la evaluación y las pruebas se reproducen sin Ollama.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config
from .enfoque_llm import analizar
from .etiquetas import cargar_muestra, leer_jsonl
from .llm import LLM

SALIDA = config.DIR_REPORTES / "predicciones_llm.jsonl"


def hechos() -> set[str]:
    return {d["id"] for d in leer_jsonl(SALIDA)} if SALIDA.exists() else set()


def correr(ids_orden: list[str], muestra: dict[str, dict], limite_horas: float = 6.0) -> None:
    llm = LLM()
    ya = hechos()
    pendientes = [i for i in ids_orden if i not in ya]
    print(f"{len(ya)} ya hechas · {len(pendientes)} pendientes · modelo {llm.modelo}", flush=True)
    t0 = time.time()
    for k, i in enumerate(pendientes, 1):
        r = muestra[i]
        res = analizar(r["titulo_resena"], r["texto"], llm)
        res["id"] = i
        res["conjunto"] = r["conjunto"]
        with open(SALIDA, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        transcurrido = time.time() - t0
        print(f"[{k}/{len(pendientes)}] {i} {res['segundos']:.0f} s · {len(res['aspectos'])} aspectos"
              f"{' · ERROR ' + res['error'][:80] if res['error'] else ''} · media {transcurrido / k:.0f} s", flush=True)
        if transcurrido > limite_horas * 3600:
            print("Límite de tiempo alcanzado; vuelve a correr para seguir.")
            break


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=["dev", "test", "es", "todo"], default="todo")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--horas", type=float, default=6.0)
    a = ap.parse_args()
    muestra = cargar_muestra()
    dev = [i for i, d in muestra.items() if d["conjunto"] == "dev"]
    test = [i for i, d in muestra.items() if d["conjunto"] == "test"]
    es = [i for i, d in muestra.items() if d["conjunto"] == "es"]
    if a.conjunto == "es":
        orden = es[:a.n]
    elif a.conjunto == "dev":
        orden = dev[:a.n]
    elif a.conjunto == "test":
        orden = test[:a.n]
    else:
        orden = dev[:40] + test + es     # 40 de dev para revisar el prompt; el test completo; el conjunto en español
    correr(orden, muestra, a.horas)


if __name__ == "__main__":
    main()
