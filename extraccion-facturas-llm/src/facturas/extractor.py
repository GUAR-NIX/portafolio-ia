"""
Extractor con LLM local: texto de la factura → JSON (salida estructurada con el esquema pydantic) → validaciones de
negocio → si hay errores, se le devuelven al modelo y se reintenta (máx. config.MAX_REINTENTOS) → se queda el intento
con menos errores. Lo que siga sin cuadrar se marca "revisar" (revisión humana) en vez de pasar en silencio.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import config
from .esquema import Factura, esquema_llm
from .llm import LLM, ErrorLLM, extraer_json
from .reparaciones import reparar
from .texto import TextoFactura, extraer_texto
from .validaciones import Problema, errores, validar

SISTEMA = """Extraes datos de facturas del Ecuador (texto sacado de un PDF o por OCR) y respondes SOLO con el JSON pedido.
Reglas:
- Copia los datos tal como aparecen; no inventes. Si un dato no aparece en el texto, usa null.
- ruc_emisor: los 13 dígitos del RUC de quien emite la factura (no el del cliente).
- razon_social_emisor: nombre de la empresa o persona que emite.
- numero_factura: establecimiento-punto de emisión-secuencial, por ejemplo 001-002-000012345.
- numero_autorizacion: los 49 dígitos del número de autorización o clave de acceso. Si está partido en varias líneas, únelos.
- fecha_emision: fecha de emisión en formato AAAA-MM-DD.
- cliente_nombre y cliente_identificacion: el comprador (cédula de 10 dígitos, RUC de 13, o 9999999999999 si es consumidor final).
- items: una entrada por cada línea de detalle, con su cantidad, precio unitario y total de la línea.
- subtotal: suma de las líneas antes de descuento e impuestos. descuento: monto positivo, 0 si no hay. iva: valor del IVA. total: valor a pagar.
- Los montos son números, sin símbolo de moneda ni separador de miles."""

CORRECCION = """Tu extracción no pasa estas validaciones:
{errores}
Vuelve a leer el texto de la factura y devuelve el JSON completo corregido. Corrige solo lo que está mal; si un dato no aparece en el texto, usa null."""


@dataclass
class Intento:
    numero: int
    crudo: str
    factura: dict | None
    errores: list[str]
    segundos: float


@dataclass
class Resultado:
    factura: Factura | None
    problemas: list[Problema]
    intentos: list[Intento]
    texto: TextoFactura | None = None
    estado: str = "error"                 # valida | con_advertencias | revisar | error
    segundos_llm: float = 0.0
    mensaje: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def reintentos(self) -> int:
        return max(0, len(self.intentos) - 1)

    def dict(self) -> dict:
        return {"estado": self.estado, "factura": json.loads(self.factura.model_dump_json()) if self.factura else None,
                "problemas": [p.dict() for p in self.problemas], "reintentos": self.reintentos,
                "segundos_llm": round(self.segundos_llm, 1), "metodo_texto": self.texto.metodo if self.texto else None,
                "mensaje": self.mensaje}


def _estado(problemas: list[Problema]) -> str:
    if errores(problemas):
        return "revisar"
    return "con_advertencias" if problemas else "valida"


def _parsear(crudo: str, con_reparaciones: bool = True) -> tuple[Factura | None, list[Problema]]:
    try:
        f = Factura.model_validate(extraer_json(crudo))
        if con_reparaciones:
            f = reparar(f)
    except (ErrorLLM, ValidationError) as e:
        msg = str(e).splitlines()[0][:300]
        return None, [Problema("json_valido", "error", f"El JSON no cumple el esquema: {msg}", [])]
    return f, validar(f)


def extraer_de_texto(texto: str, llm: LLM | None = None, max_reintentos: int = config.MAX_REINTENTOS,
                     con_reparaciones: bool = True) -> Resultado:
    llm = llm or LLM()
    esquema = esquema_llm()
    mensajes = [{"role": "user", "content": f"Texto de la factura:\n{texto}"}]
    intentos: list[Intento] = []
    mejor: tuple[int, Factura | None, list[Problema]] | None = None
    seg0 = llm.uso.segundos
    for n in range(max_reintentos + 1):
        t0 = time.perf_counter()
        crudo = llm.chat(SISTEMA, mensajes, json_schema=esquema, max_tokens=1800)
        f, probs = _parsear(crudo, con_reparaciones)
        errs = errores(probs)
        intentos.append(Intento(n, crudo, json.loads(f.model_dump_json()) if f else None,
                                [e.mensaje for e in errs], round(time.perf_counter() - t0, 2)))
        if f is not None and (mejor is None or len(errs) < mejor[0]):
            mejor = (len(errs), f, probs)
        if not errs:
            break
        mensajes += [{"role": "assistant", "content": crudo},
                     {"role": "user", "content": CORRECCION.format(errores="\n".join(f"- {e.mensaje}" for e in errs))}]
    seg = llm.uso.segundos - seg0
    if mejor is None:
        return Resultado(None, errores(probs), intentos, estado="error", segundos_llm=seg,
                         mensaje="El modelo no devolvió un JSON válido.")
    _, f, probs = mejor
    return Resultado(f, probs, intentos, estado=_estado(probs), segundos_llm=seg)


def extraer_pdf(ruta: str | Path, llm: LLM | None = None, max_reintentos: int = config.MAX_REINTENTOS) -> Resultado:
    t = extraer_texto(ruta)
    if not t.texto:
        return Resultado(None, [], [], texto=t, estado="error", mensaje=" ".join(t.avisos) or "PDF sin texto.")
    r = extraer_de_texto(t.texto, llm, max_reintentos)
    r.texto = t
    return r


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extrae una factura PDF a JSON validado")
    ap.add_argument("pdf")
    ap.add_argument("--sin-reintentos", action="store_true")
    a = ap.parse_args()
    llm = LLM()
    r = extraer_pdf(a.pdf, llm, 0 if a.sin_reintentos else config.MAX_REINTENTOS)
    print(json.dumps(r.dict(), ensure_ascii=False, indent=1))
    print(f"\n{llm.uso.llamadas} llamadas · {llm.uso.segundos:.0f} s · {llm.uso.tokens_entrada} tokens de entrada · "
          f"{llm.uso.tokens_salida} de salida")
