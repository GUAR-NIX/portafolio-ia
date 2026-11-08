import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from resenas.llm import LLM  # noqa: E402


class LLMFalso(LLM):
    """LLM simulado: devuelve en orden las respuestas dadas (la última se repite) y guarda lo que recibió."""

    def __init__(self, respuestas: list):
        super().__init__(proveedor="falso")
        self.respuestas = respuestas
        self.recibidos = []

    def chat(self, sistema, mensajes, json_schema=None, max_tokens=600):
        self.recibidos.append({"sistema": sistema, "mensajes": mensajes, "esquema": json_schema})
        r = self.respuestas[min(self.uso.llamadas, len(self.respuestas) - 1)]
        self.uso.llamadas += 1
        self.uso.segundos += 1.5
        self.uso.tokens_salida += 40
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


@pytest.fixture
def llm_falso():
    return LLMFalso


@pytest.fixture(scope="session")
def muestra_pequena() -> dict[str, dict]:
    """8 reseñas de Amazon Reviews 2023 (ver tests/datos/LEEME.md): el texto completo de la muestra no se versiona."""
    ruta = RAIZ / "tests" / "datos" / "muestra_pequena.jsonl"
    return {d["id"]: d for d in (json.loads(x) for x in ruta.read_text(encoding="utf-8").splitlines() if x.strip())}
