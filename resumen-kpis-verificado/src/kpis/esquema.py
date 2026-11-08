"""Salida estructurada del informe (pydantic). El esquema JSON se pasa a Ollama en `format`
para que el modelo solo pueda devolver un objeto válido."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Metrica = Literal["valor", "variacion", "mediana_region", "brecha_region", "ranking"]


class Referencia(BaseModel):
    indicador: str = Field(description="clave del indicador, p. ej. 'inflacion'")
    pais: str = Field(description="código ISO3 del país, o 'LAT' para la mediana regional")
    anio: int
    metrica: Metrica
    valor: float = Field(description="valor tal como aparece en la tabla (sin unidades)")


class Afirmacion(BaseModel):
    texto: str = Field(description="una o dos oraciones en español para gerencia")
    referencias: list[Referencia] = Field(description="una referencia por cada cifra del texto")


class Informe(BaseModel):
    titular: Afirmacion
    hallazgos: list[Afirmacion] = Field(min_length=3, max_length=5)
    riesgos: list[Afirmacion] = Field(min_length=1, max_length=3)
    recomendaciones: list[Afirmacion] = Field(min_length=1, max_length=3)

    def afirmaciones(self) -> list[tuple[str, Afirmacion]]:
        out = [("titular", self.titular)]
        out += [(f"hallazgo_{i}", a) for i, a in enumerate(self.hallazgos, 1)]
        out += [(f"riesgo_{i}", a) for i, a in enumerate(self.riesgos, 1)]
        out += [(f"recomendacion_{i}", a) for i, a in enumerate(self.recomendaciones, 1)]
        return out

    def texto_markdown(self) -> str:
        partes = [f"## {self.titular.texto}", "", "**Hallazgos**"]
        partes += [f"- {a.texto}" for a in self.hallazgos]
        partes += ["", "**Riesgos**"] + [f"- {a.texto}" for a in self.riesgos]
        partes += ["", "**Recomendaciones**"] + [f"- {a.texto}" for a in self.recomendaciones]
        return "\n".join(partes)


def esquema_json() -> dict:
    return Informe.model_json_schema()
