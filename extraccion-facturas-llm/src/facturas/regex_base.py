"""
Línea base SIN LLM: expresiones regulares escritas mirando las plantillas A–D (como haría un analista en una tarde).
La plantilla E no se miró: sirve para medir qué pasa cuando llega un proveedor con un formato nuevo.

Es instantánea y no alucina, pero depende de etiquetas y formatos concretos (punto decimal, "SUBTOTAL", "IVA 15%"...).
"""
from __future__ import annotations

import re

from .esquema import Factura, Item

MESES = {m: i for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                     "septiembre", "octubre", "noviembre", "diciembre"], 1)}
MONTO = r"-?\$?\s?(\d{1,3}(?:,\d{3})*\.\d{2})"
SEP = r"\s{2,}"           # el texto normalizado separa columnas con 4 espacios
INI = r"(?:^|\s{2,})"     # etiqueta al inicio de línea o en la columna derecha (RIDE)


def _num(s: str | None) -> float | None:
    return float(s.replace(",", "")) if s else None


def _buscar(patron: str, texto: str, flags=re.I | re.M) -> str | None:
    m = re.search(patron, texto, flags)
    return m.group(1).strip() if m else None


def _fecha(texto: str) -> str | None:
    m = re.search(r"Fecha(?: de)? Emisi[oó]n:?\s*(\d{2})/(\d{2})/(\d{4})", texto, re.I) or \
        re.search(r"FECHA:?\s*(\d{2})/(\d{2})/(\d{4})", texto, re.I)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r"Fecha:?\s*(\d{4})-(\d{2})-(\d{2})", texto, re.I)
    if m:
        return "-".join(m.groups())
    m = re.search(r"(\d{1,2}) de (" + "|".join(MESES) + r") de (\d{4})", texto, re.I)
    if m:
        return f"{m.group(3)}-{MESES[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    return None


def _razon_social(texto: str) -> str | None:
    lineas = [re.split(SEP, ln)[0].strip() for ln in texto.splitlines()]
    for k, ln in enumerate(lineas):
        # RIDE: el nombre va justo antes de "Dirección Matriz" y puede ocupar 2 líneas en mayúsculas
        if ln.lower().startswith("dirección matriz") and k > 0:
            partes, j = [lineas[k - 1]], k - 2
            while j >= 0 and lineas[j].isupper() and not re.match(r"(LOGO|R\.?U\.?C|FACTURA|NÚMERO|\d)", lineas[j]):
                partes.insert(0, lineas[j])
                j -= 1
            return " ".join(partes)
        # ticket: el nombre (1 o 2 líneas) va antes de "RUC: ..."
        if re.match(r"RUC:\s*\d{13}$", ln) and 0 < k <= 3:
            return " ".join(lineas[:k])
    for ln in lineas:                                     # resto: la primera línea que no es logo ni encabezado
        if ln and not re.match(r"(LOGO|R\.?U\.?C|FACTURA|COMPROBANTE)", ln, re.I):
            return ln
    return None


def _clave(texto: str) -> str | None:
    claves = re.findall(r"(?<!\d)(\d{49})(?!\d)", texto)
    if claves:
        return claves[0]
    m = re.search(r"CLAVE DE ACCESO:?\s*\n((?:\d+\n?){2,3})", texto, re.I)    # ticket: partida en 2 líneas
    if m:
        d = re.sub(r"\D", "", m.group(1))
        return d if len(d) == 49 else None
    return None


def _cliente(texto: str) -> str | None:
    m = re.search(r"CLIENTE:\s*(.+)\n(?!CI/RUC|-)([^:\n]+)\n", texto)       # ticket: nombre partido
    if m and not re.search(r"\d", m.group(2)):
        return f"{m.group(1).strip()} {m.group(2).strip()}"
    return _buscar(r"(?:Nombres y Apellidos|CLIENTE|Señor\(es\)):\s*(.+?)(?:\s{2,}|$)", texto) or \
        _buscar(r"Facturar a:.*\n(.+?)(?:\s{2,}|$)", texto)


def _items(texto: str) -> list[Item]:
    items = []
    lineas = texto.splitlines()
    for k, ln in enumerate(lineas):
        # A · RIDE: código  cant  descripción  p.unit  descuento  total
        m = re.match(rf"^\S+{SEP}(\d+(?:\.\d+)?){SEP}(.+?){SEP}{MONTO}{SEP}{MONTO}{SEP}{MONTO}$", ln)
        if m:
            items.append(Item(descripcion=m.group(2), cantidad=float(m.group(1)), precio_unitario=_num(m.group(3)),
                              total=_num(m.group(5))))
            continue
        # B · moderna: descripción  cant  $precio  $importe
        m = re.match(rf"^(.+?){SEP}(\d+(?:\.\d+)?){SEP}{MONTO}{SEP}{MONTO}$", ln)
        if m:
            items.append(Item(descripcion=m.group(1), cantidad=float(m.group(2)), precio_unitario=_num(m.group(3)),
                              total=_num(m.group(4))))
            continue
        # D · servicios: "1. Descripción — 2 × $10.00 = $20.00"
        m = re.match(rf"^\d+\.\s+(.+?)\s+—\s+(\d+(?:\.\d+)?)\s+×\s+{MONTO}\s+=\s+{MONTO}$", ln)
        if m:
            items.append(Item(descripcion=m.group(1), cantidad=float(m.group(2)), precio_unitario=_num(m.group(3)),
                              total=_num(m.group(4))))
            continue
        # C · ticket: la descripción va en la(s) línea(s) anterior(es) y luego "  2 x 1.50     3.00"
        m = re.match(rf"^(\d+(?:\.\d+)?)\s+x\s+{MONTO}{SEP}{MONTO}$", ln)
        if m:
            desc, j = [], k - 1
            while j >= 0 and not re.search(r"\d+\.\d{2}$|^CANT|^-+$", lineas[j]):
                desc.insert(0, lineas[j].strip())
                j -= 1
            items.append(Item(descripcion=" ".join(desc), cantidad=float(m.group(1)), precio_unitario=_num(m.group(2)),
                              total=_num(m.group(3))))
    return items


def extraer_regex(texto: str) -> Factura:
    ruc = _buscar(r"R\.?U\.?C\.?:?\s*(\d{13})", texto)
    claves = re.findall(r"(?<!\d)(\d{49})(?!\d)", texto)
    cliente = _buscar(r"(?:Nombres y Apellidos|CLIENTE|Señor\(es\)):\s*(.+?)(?:\s{2,}|$)", texto) or \
        _buscar(r"Facturar a:.*\n(.+?)(?:\s{2,}|$)", texto)
    ident = _buscar(r"(?:Identificaci[oó]n|CI/RUC|C\.I\./RUC|RUC/CI):\s*(\d{10,13})", texto)
    subtotal = _buscar(rf"{INI}SUBTOTAL:?\s+{MONTO}$", texto)
    descuento = _buscar(rf"{INI}(?:TOTAL DESCUENTO|DESC\.|Descuento:?)\s+{MONTO}", texto)
    iva = _buscar(rf"{INI}IVA\s*\(?\d{{1,2}}\s?%\)?:?\s+{MONTO}", texto)
    total = _buscar(rf"{INI}(?:VALOR TOTAL|Total a pagar|TOTAL USD|TOTAL:)\s+{MONTO}", texto)
    return Factura(
        ruc_emisor=ruc, razon_social_emisor=_razon_social(texto),
        numero_factura=_buscar(r"(\d{3}-\d{3}-\d{9})", texto),
        numero_autorizacion=_clave(texto), fecha_emision=_fecha(texto),
        cliente_nombre=_cliente(texto), cliente_identificacion=ident, items=_items(texto),
        subtotal=_num(subtotal), descuento=_num(descuento) if descuento else 0.0, iva=_num(iva), total=_num(total))
