"""
Validaciones de negocio sobre una factura extraída. Cada problema trae un mensaje en español que sirve para dos cosas:
mostrarlo al usuario y devolvérselo al LLM para que corrija (reintento con corrección).

* error       → la factura no cuadra: se reintenta y, si sigue, se marca para revisión humana
* advertencia → falta un dato opcional (p. ej. sin número de autorización): no se reintenta
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from . import config, sri
from .esquema import Factura

TOL = config.TOLERANCIA


@dataclass
class Problema:
    regla: str
    nivel: str          # "error" | "advertencia"
    mensaje: str
    campos: list[str]

    def dict(self) -> dict:
        return asdict(self)


def _cerca(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol + 1e-9


def validar(f: Factura) -> list[Problema]:
    p: list[Problema] = []
    E = lambda regla, msg, *campos: p.append(Problema(regla, "error", msg, list(campos)))          # noqa: E731
    A = lambda regla, msg, *campos: p.append(Problema(regla, "advertencia", msg, list(campos)))    # noqa: E731

    # --- identificación
    if not f.ruc_emisor:
        E("ruc_presente", "Falta el RUC del emisor (13 dígitos).", "ruc_emisor")
    elif not sri.ruc_valido(f.ruc_emisor):
        E("ruc_digito", f"El RUC del emisor {f.ruc_emisor} no es válido (longitud o dígito verificador); "
                        "revisa que copiaste los 13 dígitos exactos.", "ruc_emisor")
    if not f.numero_factura or not re.fullmatch(r"\d{3}-\d{3}-\d{9}", f.numero_factura):
        E("numero_formato", f"El número de factura '{f.numero_factura}' no tiene el formato 001-001-000000123.",
          "numero_factura")
    if f.fecha_emision is None:
        E("fecha_presente", "Falta la fecha de emisión.", "fecha_emision")
    if f.cliente_identificacion and not sri.identificacion_valida(f.cliente_identificacion):
        E("cliente_id", f"La identificación del cliente {f.cliente_identificacion} no es una cédula, RUC ni "
                        "consumidor final válido; revisa los dígitos.", "cliente_identificacion")

    # --- clave de acceso / autorización
    if not f.numero_autorizacion:
        A("autorizacion_presente", "La factura no muestra número de autorización.", "numero_autorizacion")
    elif not sri.clave_valida(f.numero_autorizacion):
        E("autorizacion_digito", f"El número de autorización tiene {len(f.numero_autorizacion)} dígitos o su "
                                 "dígito verificador no cuadra; deben ser 49 dígitos exactos (si está partido en "
                                 "varias líneas, únelos).", "numero_autorizacion")
    else:
        partes = sri.desarmar_clave(f.numero_autorizacion)
        if f.ruc_emisor and partes["ruc"] != f.ruc_emisor:
            E("autorizacion_ruc", f"El RUC dentro de la clave de acceso ({partes['ruc']}) no coincide con el RUC "
                                  f"del emisor ({f.ruc_emisor}).", "ruc_emisor", "numero_autorizacion")
        if f.fecha_emision and partes["fecha"] != f.fecha_emision.isoformat():
            E("autorizacion_fecha", f"La fecha dentro de la clave de acceso ({partes['fecha']}) no coincide con la "
                                    f"fecha de emisión ({f.fecha_emision}).", "fecha_emision")
        if f.numero_factura and partes["numero_factura"] != f.numero_factura:
            E("autorizacion_numero", f"El número dentro de la clave de acceso ({partes['numero_factura']}) no "
                                     f"coincide con el número de factura ({f.numero_factura}).", "numero_factura")

    # --- ítems y montos
    if not f.items:
        E("items_presentes", "No se extrajo ninguna línea de detalle (ítems).", "items")
    for k, it in enumerate(f.items, 1):
        esperado = round(it.cantidad * it.precio_unitario, 2)
        if not _cerca(esperado, it.total, max(TOL, 0.005 * abs(esperado))):
            E("item_linea", f"Ítem {k} ('{it.descripcion[:40]}'): cantidad {it.cantidad} × precio "
                            f"{it.precio_unitario} = {esperado:.2f}, pero el total de la línea es {it.total:.2f}.",
              "items")
    if f.subtotal is None or f.total is None or f.iva is None:
        E("montos_presentes", "Faltan montos (subtotal, IVA o total).", "subtotal", "iva", "total")
        return p
    desc = f.descuento or 0.0
    if f.items:
        suma = round(sum(it.total for it in f.items), 2)
        if not _cerca(suma, f.subtotal, max(TOL, 0.01 * len(f.items))):
            E("items_suman", f"La suma de las líneas ({suma:.2f}) no es igual al subtotal ({f.subtotal:.2f}): "
                             "falta o sobra algún ítem, o algún monto está mal copiado.", "items", "subtotal")
    if not _cerca(f.subtotal - desc + f.iva, f.total):
        E("total_cuadra", f"subtotal ({f.subtotal:.2f}) − descuento ({desc:.2f}) + IVA ({f.iva:.2f}) = "
                          f"{f.subtotal - desc + f.iva:.2f}, pero el total es {f.total:.2f}.",
          "subtotal", "descuento", "iva", "total")
    base = f.subtotal - desc
    tasas = [sri.tasa_iva(f.fecha_emision)] if f.fecha_emision else [0.12, 0.15]
    if not any(_cerca(round(base * t, 2), f.iva, 0.03) for t in tasas):
        E("iva_tasa", f"El IVA ({f.iva:.2f}) no corresponde al {'/'.join(f'{t:.0%}' for t in tasas)} de la base "
                      f"({base:.2f}) para la fecha {f.fecha_emision}.", "iva")
    return p


def errores(p: list[Problema]) -> list[Problema]:
    return [x for x in p if x.nivel == "error"]
