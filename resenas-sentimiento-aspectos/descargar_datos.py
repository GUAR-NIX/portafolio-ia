"""Descarga el subconjunto de Amazon Reviews 2023 (~60 mil reseñas de 320 productos de 8 tipos). Reanudable. ~15 min."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from resenas.descarga import main  # noqa: E402

if __name__ == "__main__":
    main()
