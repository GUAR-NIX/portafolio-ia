"""Corre (y graba) las llamadas a Ollama que necesita la evaluación. Es reanudable: cada respuesta se
guarda al instante en data/grabacion_llm.jsonl y las ya grabadas no se repiten.

    python -m agente.correr_llm validacion      # agente v1 y v2 + líneas base en las 44 de validación
    python -m agente.correr_llm prueba_base     # zero-shot y ReAct en las 60 de prueba
    python -m agente.correr_llm prueba_agente   # agente con la versión de prompt elegida en validación
"""
from __future__ import annotations

import sys
import time

from . import config, evaluacion, motor_agente


def _correr(conjunto: list[dict], enfoque: str, version: str | None = None) -> None:
    t0 = time.perf_counter()
    for i, q in enumerate(conjunto, start=1):
        motor_agente.consultar_agente(q["pregunta"], enfoque=enfoque, version_prompt=version)
        print(f"[{enfoque} {version or ''}] {i}/{len(conjunto)} {q['id_pregunta']} "
              f"({time.perf_counter() - t0:.0f} s acumulados)", flush=True)


def main(fase: str) -> None:
    config.MODO_DATOS = "instantanea"
    if fase == "validacion":
        val = evaluacion.cargar_conjunto("validacion")
        _correr(val, "tool_calling_nativo", "v2")
        _correr(val, "tool_calling_nativo", "v1")
        _correr(val, "zeroshot_sin_tools")
        _correr(val, "react_texto")
    elif fase == "prueba_base":
        prueba = evaluacion.cargar_conjunto("prueba")
        _correr(prueba, "zeroshot_sin_tools")
        _correr(prueba, "react_texto")
    elif fase == "prueba_agente":
        version = evaluacion.elegir_version_prompt()
        print(f"Versión de prompt elegida en validación: {version}", flush=True)
        _correr(evaluacion.cargar_conjunto("prueba"), "tool_calling_nativo", version)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
