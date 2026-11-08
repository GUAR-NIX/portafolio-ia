"""
Los datos de este proyecto no se descargan: se GENERAN (facturas sintéticas, sin datos de personas reales).
Equivale al descargar_datos.py de los otros proyectos.

    python generar_datos.py          # 200 PDF + verdad JSON en data/raw/ (~10 s), catálogo y 10 muestras
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from facturas.generar import generar  # noqa: E402

if __name__ == "__main__":
    filas = generar()
    print(f"{len(filas)} facturas generadas en data/raw/facturas (verdad en data/raw/verdad)")
