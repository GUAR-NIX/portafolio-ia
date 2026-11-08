"""Modelo pydantic de la anotación: el mismo para las etiquetas hechas a mano, la salida del LLM y la de los demás
enfoques. Su JSON Schema se pasa a Ollama como salida estructurada."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import config

Aspecto = Enum("Aspecto", {a: a for a in config.ASPECTOS}, type=str)
Polaridad = Literal["positivo", "negativo", "neutro"]


class MencionAspecto(BaseModel):
    aspecto: Aspecto
    polaridad: Polaridad
    cita: str = Field(description="Fragmento literal de la reseña que justifica el aspecto")

    model_config = {"use_enum_values": True}


class AnalisisResena(BaseModel):
    # El global va al final: Ollama genera en el orden del esquema y así decide después de revisar los aspectos.
    aspectos: list[MencionAspecto] = Field(default_factory=list)
    sentimiento_global: Polaridad

    @field_validator("aspectos")
    @classmethod
    def _un_aspecto_una_vez(cls, v: list[MencionAspecto]) -> list[MencionAspecto]:
        """Si el modelo repite un aspecto, se queda la primera mención (la guía pide una por reseña)."""
        vistos, salida = set(), []
        for m in v:
            if m.aspecto not in vistos:
                vistos.add(m.aspecto)
                salida.append(m)
        return salida

    def como_dict(self) -> dict[str, str]:
        """{aspecto: polaridad}: la forma que usan las métricas."""
        return {m.aspecto: m.polaridad for m in self.aspectos}


def esquema_ollama() -> dict:
    """JSON Schema para el parámetro `format` de Ollama: la cita primero dentro de cada mención."""
    return {
        "type": "object",
        "properties": {
            "aspectos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cita": {"type": "string"},
                        "aspecto": {"type": "string", "enum": list(config.ASPECTOS)},
                        "polaridad": {"type": "string", "enum": list(config.POLARIDADES)},
                    },
                    "required": ["cita", "aspecto", "polaridad"],
                },
            },
            "sentimiento_global": {"type": "string", "enum": list(config.POLARIDADES)},
        },
        "required": ["aspectos", "sentimiento_global"],
    }
