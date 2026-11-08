import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from facturas import config  # noqa: E402
from facturas.llm import LLM  # noqa: E402


class LLMFalso(LLM):
    """LLM simulado: devuelve en orden las respuestas dadas (la última se repite) y guarda lo que recibió."""

    def __init__(self, respuestas: list):
        super().__init__(proveedor="falso")
        self.respuestas = respuestas
        self.recibidos = []

    def chat(self, sistema, mensajes, json_schema=None, max_tokens=1500):
        self.recibidos.append(mensajes)
        r = self.respuestas[min(self.uso.llamadas, len(self.respuestas) - 1)]
        self.uso.llamadas += 1
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


@pytest.fixture
def llm_falso():
    return LLMFalso


def _muestras():
    import csv
    with open(config.CATALOGO, encoding="utf-8") as fh:
        cat = {r["id"]: r for r in csv.DictReader(fh)}
    return [dict(cat[p.stem], pdf=p, verdad=json.loads(p.with_suffix(".json").read_text(encoding="utf-8")))
            for p in sorted(config.DIR_MUESTRAS.glob("*.pdf"))]


@pytest.fixture(scope="session")
def muestras():
    """Las 10 facturas versionadas en data/muestras (2 por plantilla) con su verdad y su fila del catálogo."""
    return _muestras()


@pytest.fixture
def factura_ok(muestras):
    return json.loads(json.dumps(next(m for m in muestras if m["plantilla"] == "B_moderna")["verdad"]))
