"""Corre (y graba) las llamadas a Ollama de la evaluación. Es reanudable: cada respuesta se guarda al
instante en data/grabacion_llm.jsonl y las ya grabadas no se repiten.

    python -m kpis.correr_llm validacion   # (b) con prompt v1 y v2 en validación
    python -m kpis.correr_llm validacion_c # (c) reintentos con v1 y v2 en validación (verificador congelado)
    python -m kpis.correr_llm prueba       # (a), (b) y (c) con la versión elegida en validación
"""
from __future__ import annotations

import sys
import time

from . import config, evaluacion, generadores


def _correr(conjunto: list[tuple[str, int]], enfoque: str, version: str) -> None:
    t0 = time.perf_counter()
    for i, (pais, anio) in enumerate(conjunto, start=1):
        r = generadores.generar(enfoque, pais, anio, version, proveedor="ollama")
        v = r["verificacion"] or {}
        print(f"[{enfoque} {version}] {i}/{len(conjunto)} {pais} {anio}: llamadas={r['llamadas_llm']} "
              f"cifras={v.get('n_cifras')} alucinacion={v.get('tasa_alucinacion', float('nan')):.2f} "
              f"({time.perf_counter() - t0:.0f} s acumulados)", flush=True)


def main(fase: str) -> None:
    if fase == "validacion":
        for version in generadores.VERSIONES_PROMPT:
            _correr(config.VALIDACION, "estructurado", version)
    elif fase == "validacion_c":
        for version in generadores.VERSIONES_PROMPT:
            _correr(config.VALIDACION, "verificado", version)
    elif fase == "prueba":
        version = evaluacion.elegir_version_prompt()
        print(f"Versión de prompt elegida en validación: {version}", flush=True)
        _correr(config.PRUEBA, "estructurado", version)
        _correr(config.VALIDACION + config.PRUEBA, "libre", version)
        _correr(config.PRUEBA, "verificado", version)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
