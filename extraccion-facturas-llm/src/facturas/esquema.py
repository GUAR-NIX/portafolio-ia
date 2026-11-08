"""
Esquema de salida (pydantic). Es a la vez:
  * el formato de la "verdad" de cada factura sintética,
  * el JSON Schema que se le pasa a Ollama (salida estructurada),
  * el validador de tipos de lo que devuelve el modelo.

El ORDEN de los campos importa: Ollama genera el JSON en el orden del esquema (lección del proyecto IA/), así que va
en el orden en que aparecen en una factura: emisor → número → autorización → fecha → cliente → ítems → totales.
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator


class Item(BaseModel):
    descripcion: str
    cantidad: float
    precio_unitario: float
    total: float = Field(description="cantidad × precio unitario (precio total de la línea)")


class Factura(BaseModel):
    ruc_emisor: str | None = Field(description="RUC del emisor, 13 dígitos")
    razon_social_emisor: str | None
    numero_factura: str | None = Field(description="formato 001-001-000000123")
    numero_autorizacion: str | None = Field(description="49 dígitos; null si no aparece")
    fecha_emision: date | None = Field(description="AAAA-MM-DD")
    cliente_nombre: str | None
    cliente_identificacion: str | None = Field(description="cédula, RUC o 9999999999999; null si no aparece")
    items: list[Item]
    subtotal: float | None = Field(description="suma de las líneas, antes de descuento e IVA")
    descuento: float | None = Field(description="0 si no hay descuento")
    iva: float | None
    total: float | None

    @field_validator("ruc_emisor", "numero_autorizacion", "cliente_identificacion", mode="before")
    @classmethod
    def _solo_digitos(cls, v):
        if v is None:
            return None
        d = re.sub(r"\D", "", str(v))
        return d or None

    @field_validator("numero_factura", mode="before")
    @classmethod
    def _numero(cls, v):
        if v is None:
            return None
        d = re.sub(r"\D", "", str(v))
        if len(d) == 15:
            return f"{d[:3]}-{d[3:6]}-{d[6:]}"
        partes = re.findall(r"\d+", str(v))
        if len(partes) == 3:
            return f"{int(partes[0]):03d}-{int(partes[1]):03d}-{int(partes[2]):09d}"
        return str(v).strip() or None

    @field_validator("razon_social_emisor", "cliente_nombre", mode="before")
    @classmethod
    def _texto(cls, v):
        if v is None:
            return None
        s = re.sub(r"\s+", " ", str(v)).strip()
        return s or None

    @field_validator("descuento", mode="after")
    @classmethod
    def _descuento_positivo(cls, v):
        # muchas facturas lo imprimen como "-$3.14"; el modelo copia el signo y el total deja de cuadrar
        return abs(v) if v is not None else None

    @field_validator("fecha_emision", mode="before")
    @classmethod
    def _fecha_vacia(cls, v):
        return None if v in ("", "null", "None") else v


def esquema_llm() -> dict:
    """JSON Schema compacto para el parámetro `format` de Ollama (sin títulos ni descripciones largas)."""
    s = Factura.model_json_schema()

    def limpiar(o):
        if isinstance(o, dict):
            o.pop("title", None)
            o.pop("default", None)
            return {k: limpiar(v) for k, v in o.items()}
        if isinstance(o, list):
            return [limpiar(x) for x in o]
        return o

    s = limpiar(s)
    s["required"] = list(Factura.model_fields)          # todos obligatorios (null explícito si falta)
    s["$defs"]["Item"]["required"] = list(Item.model_fields)
    return s


CAMPOS = [c for c in Factura.model_fields if c != "items"]
CAMPOS_MONTO = ["subtotal", "descuento", "iva", "total"]
