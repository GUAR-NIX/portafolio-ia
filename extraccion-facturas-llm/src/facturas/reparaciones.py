"""
Reparaciones deterministas sobre la salida del LLM (sin volver a llamarlo).

Se escribieron DESPUÉS de ver los errores de la primera evaluación, así que el README reporta dos cifras:
"medida" (sin reparar) y "con reparaciones" (optimista: se diseñaron mirando esas mismas facturas).

1. Consumidor final: el modelo devuelve la identificación 9999999999999 pero deja el nombre en null.
   Por norma del SRI ese comprador se llama siempre "CONSUMIDOR FINAL".
2. Descripciones contaminadas: en tickets y en el formato "1. Humita — 1 × $1.40 = $1.40" el modelo a veces copia
   la cantidad y el precio dentro de la descripción ("Humita— 1 x $1.40"). Los montos ya están en sus campos.
"""
from __future__ import annotations

import re

from . import sri
from .esquema import Factura

_COLA_MONTOS = re.compile(r"\s*[—–-]?\s*[\dIl]+(?:[.,]\d+)?\s*[x×]\s*\$?\s*[\d.,]+(?:\s*=\s*\$?\s*[\d.,]+)?\s*$", re.I)


def limpiar_descripcion(d: str) -> str:
    limpia = _COLA_MONTOS.sub("", d).strip(" —–-")
    return re.sub(r"\s+", " ", limpia) or d


def reparar(f: Factura) -> Factura:
    f = f.model_copy(deep=True)
    if f.cliente_identificacion == sri.CONSUMIDOR_FINAL and not f.cliente_nombre:
        f.cliente_nombre = "CONSUMIDOR FINAL"
    for it in f.items:
        it.descripcion = limpiar_descripcion(it.descripcion)
    return f


def reparar_dict(d: dict | None) -> dict | None:
    if d is None:
        return None
    return reparar(Factura.model_validate(d)).model_dump(mode="json")
