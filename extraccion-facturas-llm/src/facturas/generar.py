"""
Genera N facturas SINTÉTICAS en PDF (estilo ecuatoriano) y su "verdad" en JSON.

    python -m facturas.generar            # 200 facturas en data/raw/ + data/catalogo.csv + 10 muestras

* 5 plantillas con diseños, etiquetas, formatos de fecha y de número distintos.
  La plantilla E ("compacta") se diseñó como **plantilla nunca vista**: ni la línea base de regex ni el prompt se
  ajustaron mirándola (montos con coma decimal 1.234,56, fecha "17-may-2024", etiquetas diferentes).
* Ruido: ~15 % escaneadas (imagen sin capa de texto, inclinada, con ruido y JPEG), parte de ellas giradas 90/180/270°;
  ~12 % con campos faltantes (la verdad es null: el extractor debe devolver null y no inventar).
* Todo es inventado: nombres por combinación aleatoria, RUC/cédulas aleatorios que solo cumplen el dígito verificador.
* Reproducible: semilla fija (config.SEMILLA).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
from datetime import date, timedelta

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from . import config, sri
from .esquema import Factura

PLANTILLAS = ["A_ride_sri", "B_moderna", "C_ticket", "D_servicios", "E_compacta"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]

NOMBRES = ["María", "José", "Ana", "Luis", "Carmen", "Jorge", "Lucía", "Diego", "Valeria", "Andrés", "Gabriela",
           "Fernando", "Daniela", "Santiago", "Paola", "Mateo", "Camila", "Esteban", "Verónica", "Javier", "Sofía",
           "Patricio", "Rocío", "Miguel", "Isabel", "Ricardo", "Elena", "Hernán", "Mónica", "Pablo"]
APELLIDOS = ["Zambrano", "Andrade", "Vera", "Cedeño", "Mora", "Paredes", "Salazar", "Torres", "Jaramillo", "Ortiz",
             "Villacís", "Benítez", "Cárdenas", "Moreira", "Chávez", "Guerrero", "Rivadeneira", "Loor", "Toapanta",
             "Quishpe", "Espinoza", "Calderón", "Mendoza", "Aguirre", "Naranjo", "Sarmiento", "Valencia", "Ponce"]
PALABRAS_EMPRESA = ["Andina", "Del Pacífico", "Los Ceibos", "Sierra Verde", "Tungurahua Sur", "Chimborazo",
                    "La Ensenada", "Cotopaxi", "Galápagos Azul", "El Condado", "Río Daule", "Pichincha Norte",
                    "Manabita", "Austral", "Amazonía", "Puerto Bolívar", "Cumbayá", "Los Álamos", "Samborondón Tech",
                    "Nueva Loja"]
SUFIJOS = ["S.A.", "Cía. Ltda.", "S.A.S.", "C.A."]
CIUDADES = ["Quito", "Guayaquil", "Cuenca", "Ambato", "Manta", "Loja", "Machala", "Riobamba", "Ibarra", "Portoviejo"]
CALLES = ["Av. Amazonas", "Av. 9 de Octubre", "Calle Bolívar", "Av. de las Américas", "Calle Sucre", "Av. Shyris",
          "Calle Olmedo", "Av. Solano", "Calle Rocafuerte", "Av. Quito"]

# sector → (prefijo comercial, [(descripción, precio mínimo, precio máximo, ¿cantidad decimal?)])
SECTORES = {
    "ferreteria": ("Ferretería", [
        ("Cemento gris saco 50 kg", 7.5, 8.9, False), ("Varilla corrugada 12 mm x 12 m", 9.0, 12.5, False),
        ("Cable eléctrico #12 AWG (metro)", 0.45, 0.85, True), ("Pintura látex blanca galón", 14.0, 22.0, False),
        ("Tubo PVC 1/2 pulgada x 6 m", 3.2, 5.1, False), ("Clavos de 2 pulgadas (kg)", 1.8, 2.6, True),
        ("Martillo de uña 16 oz", 6.5, 11.0, False), ("Cinta métrica 5 m", 3.0, 6.5, False),
        ("Llave de paso 1/2 pulgada", 4.5, 8.0, False), ("Bloque de hormigón 15 cm", 0.38, 0.55, False),
        ("Arena fina (m3)", 18.0, 25.0, True), ("Taladro percutor 600 W", 38.0, 75.0, False)]),
    "tecnologia": ("Tecnología", [
        ("Laptop 15.6 pulgadas 16 GB RAM", 520.0, 890.0, False), ("Mouse inalámbrico", 8.0, 19.0, False),
        ("Teclado USB español", 9.0, 22.0, False), ("Monitor LED 24 pulgadas", 115.0, 180.0, False),
        ("Disco SSD 1 TB", 55.0, 95.0, False), ("Memoria USB 64 GB", 6.0, 12.0, False),
        ("Router WiFi doble banda", 35.0, 70.0, False), ("Cable HDMI 2 m", 4.0, 9.0, False),
        ("Impresora multifunción tinta continua", 160.0, 260.0, False), ("Servicio de instalación de red", 40.0, 120.0, False),
        ("Licencia antivirus anual", 25.0, 45.0, False), ("Audífonos con micrófono", 12.0, 35.0, False)]),
    "papeleria": ("Papelería", [
        ("Resma papel bond A4 75 g", 4.2, 5.8, False), ("Cuaderno universitario 100 hojas", 1.6, 2.9, False),
        ("Esfero azul (caja x 12)", 3.0, 5.5, False), ("Carpeta archivadora lomo ancho", 2.5, 4.2, False),
        ("Tóner láser negro", 45.0, 85.0, False), ("Grapadora metálica", 5.5, 12.0, False),
        ("Marcadores permanentes (set x 4)", 2.8, 5.0, False), ("Sobres manila oficio (paquete x 25)", 3.5, 6.0, False),
        ("Calculadora científica", 14.0, 28.0, False), ("Cinta adhesiva transparente", 0.6, 1.4, False)]),
    "servicios": ("Servicios", [
        ("Asesoría contable mensual", 150.0, 450.0, False), ("Declaración de impuestos", 30.0, 90.0, False),
        ("Hora de consultoría en datos", 25.0, 60.0, True), ("Mantenimiento preventivo de equipos", 35.0, 120.0, False),
        ("Diseño de logotipo", 80.0, 250.0, False), ("Capacitación en Excel (hora)", 20.0, 45.0, True),
        ("Auditoría de inventario", 200.0, 600.0, False), ("Traducción de documento (página)", 8.0, 18.0, False),
        ("Desarrollo de reporte en Power BI", 120.0, 400.0, False), ("Soporte técnico remoto (hora)", 15.0, 35.0, True)]),
    "restaurante": ("Restaurante", [
        ("Almuerzo ejecutivo", 3.5, 6.5, False), ("Encebollado mixto", 4.5, 7.5, False),
        ("Seco de pollo", 5.0, 8.5, False), ("Jugo natural de naranjilla", 1.5, 2.8, False),
        ("Café pasado", 1.0, 2.2, False), ("Bolón de verde mixto", 2.5, 4.0, False),
        ("Ceviche de camarón", 7.0, 11.0, False), ("Agua sin gas 500 ml", 0.6, 1.2, False),
        ("Humita", 1.2, 2.0, False), ("Postre del día", 2.0, 3.5, False)]),
    "limpieza": ("Distribuidora", [
        ("Detergente en polvo 2 kg", 5.5, 8.5, False), ("Cloro galón", 2.2, 3.8, False),
        ("Desinfectante lavanda galón", 3.5, 6.0, False), ("Papel higiénico jumbo (paquete x 6)", 12.0, 19.0, False),
        ("Fundas de basura industriales (paquete x 10)", 2.5, 4.5, False), ("Escoba plástica", 2.0, 4.0, False),
        ("Jabón líquido de manos 1 L", 2.8, 5.0, False), ("Guantes de nitrilo (caja x 100)", 7.0, 12.0, False),
        ("Trapeador microfibra", 4.0, 7.5, False), ("Toallas de papel (rollo)", 1.8, 3.2, False)]),
}


# ======================================================================== verdad
def _persona(rng: random.Random) -> str:
    return f"{rng.choice(NOMBRES)} {rng.choice(NOMBRES)} {rng.choice(APELLIDOS)} {rng.choice(APELLIDOS)}"


def generar_verdad(rng: random.Random, i: int) -> tuple[dict, dict]:
    """Devuelve (factura, extras) donde extras guarda lo que solo sirve para dibujar (dirección, hora, sector…)."""
    sector = rng.choice(list(SECTORES))
    prefijo, productos = SECTORES[sector]
    tipo = rng.choices(["privada", "natural", "publica"], [0.6, 0.35, 0.05])[0]
    if tipo == "natural":
        p = _persona(rng)
        razon = p.upper() if rng.random() < 0.6 else f"{p} - {prefijo} {rng.choice(PALABRAS_EMPRESA)}"
    elif tipo == "publica":
        sector, (prefijo, productos) = "servicios", SECTORES["servicios"]
        razon = f"Empresa Pública Municipal de Servicios {rng.choice(PALABRAS_EMPRESA)} EP"
    else:
        razon = f"{prefijo} {rng.choice(PALABRAS_EMPRESA)} {rng.choice(SUFIJOS)}"
    ruc = sri.generar_ruc(rng, tipo)
    numero = f"{rng.choice(['001', '001', '002', '003'])}-{rng.choice(['001', '001', '002', '100'])}-" \
             f"{rng.randint(1, 250000):09d}"
    fecha = date(2023, 1, 1) + timedelta(days=rng.randint(0, 3 * 365 - 1))
    clave = sri.armar_clave(fecha, ruc, numero, f"{rng.randint(0, 99999999):08d}")

    r = rng.random()
    if r < 0.25:
        cliente, ident = "CONSUMIDOR FINAL", sri.CONSUMIDOR_FINAL
    elif r < 0.7:
        cliente, ident = _persona(rng), sri.generar_cedula(rng)
    else:
        pre, _ = SECTORES[rng.choice(list(SECTORES))]
        cliente = f"{rng.choice(['Comercial', 'Inversiones', 'Constructora', 'Importadora', 'Hacienda'])} " \
                  f"{rng.choice(PALABRAS_EMPRESA)} {rng.choice(SUFIJOS)}"
        ident = sri.generar_ruc(rng, "privada")

    n = rng.choices(range(1, 13), [8, 10, 12, 12, 10, 9, 8, 7, 6, 5, 4, 3])[0]
    items = []
    for desc, pmin, pmax, decimal in rng.sample(productos, min(n, len(productos))):
        cant = round(rng.uniform(0.5, 30), 1) if decimal else rng.choices(
            [1, 2, 3, 4, 5, 6, 10, 12, 20, 25, 50], [30, 18, 10, 8, 8, 5, 6, 4, 4, 3, 2])[0]
        pu = round(rng.uniform(pmin, pmax), 2)
        items.append({"descripcion": desc, "cantidad": cant, "precio_unitario": pu,
                      "total": round(cant * pu + 1e-9, 2)})
    subtotal = round(sum(x["total"] for x in items), 2)
    descuento = round(subtotal * rng.choice([0.05, 0.10, 0.03]), 2) if rng.random() < 0.25 else 0.0
    tasa = sri.tasa_iva(fecha)
    iva = round((subtotal - descuento) * tasa + 1e-9, 2)
    total = round(subtotal - descuento + iva, 2)
    factura = {"ruc_emisor": ruc, "razon_social_emisor": razon, "numero_factura": numero,
               "numero_autorizacion": clave, "fecha_emision": fecha.isoformat(), "cliente_nombre": cliente,
               "cliente_identificacion": ident, "items": items, "subtotal": subtotal, "descuento": descuento,
               "iva": iva, "total": total}
    Factura.model_validate(factura)
    extras = {"sector": sector, "tasa": tasa, "direccion": f"{rng.choice(CALLES)} N{rng.randint(10, 99)}-"
              f"{rng.randint(10, 300)} y {rng.choice(CALLES).split(' ', 1)[1]}", "ciudad": rng.choice(CIUDADES),
              "hora": f"{rng.randint(8, 20):02d}:{rng.randint(0, 59):02d}",
              "telefono": f"0{rng.randint(2, 7)}{rng.randint(2000000, 2999999)}",
              "email": f"cliente{i:03d}@correo.ec", "codigos": [f"P{rng.randint(1000, 9999)}" for _ in items]}
    return factura, extras


# ======================================================================== formatos
def num(x: float, estilo: str = "us", dec: int = 2) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if estilo == "ec" else s


def cant(x: float, estilo: str = "us") -> str:
    if float(x).is_integer():
        return str(int(x))
    return num(x, estilo, 1 if round(x, 1) == x else 2)


def fecha_txt(iso: str, estilo: str) -> str:
    d = date.fromisoformat(iso)
    if estilo == "dmy":
        return f"{d:%d/%m/%Y}"
    if estilo == "iso":
        return iso
    if estilo == "largo":
        return f"{d.day} de {MESES[d.month - 1]} de {d.year}"
    if estilo == "corto":
        return f"{d.day:02d}-{MESES[d.month - 1][:3]}-{d.year}"
    raise ValueError(estilo)


def _cortar(texto: str, ancho: int) -> list[str]:
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > ancho and actual:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}".strip()
    return lineas + ([actual] if actual else [])


# ======================================================================== plantillas
def _a_ride(c: canvas.Canvas, f: dict, x: dict):
    """A · RIDE clásico del SRI: dos recuadros arriba, tabla de 6 columnas, cuadro de totales."""
    W, H = A4
    c.setLineWidth(0.8)
    c.roundRect(15 * mm, H - 105 * mm, 88 * mm, 60 * mm, 4)
    c.roundRect(107 * mm, H - 105 * mm, 88 * mm, 90 * mm, 4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, H - 30 * mm, "LOGO")
    y = H - 55 * mm
    if f["razon_social_emisor"]:
        for ln in _cortar(f["razon_social_emisor"].upper(), 38):
            c.setFont("Helvetica-Bold", 9)
            c.drawString(18 * mm, y, ln)
            y -= 4.5 * mm
    c.setFont("Helvetica", 7.5)
    for ln in [f"Dirección Matriz: {x['direccion']}, {x['ciudad']}", "Obligado a llevar contabilidad: SI",
               f"Teléfono: {x['telefono']}"]:
        y -= 1 * mm
        c.drawString(18 * mm, y, ln)
        y -= 4 * mm
    y = H - 23 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(110 * mm, y, f"R.U.C.: {f['ruc_emisor']}")
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(110 * mm, y, "FACTURA")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    c.drawString(110 * mm, y, f"No. {f['numero_factura']}")
    if f["numero_autorizacion"]:
        y -= 7 * mm
        c.setFont("Helvetica-Bold", 8)
        c.drawString(110 * mm, y, "NÚMERO DE AUTORIZACIÓN")
        y -= 4.5 * mm
        c.setFont("Helvetica", 6.6)
        c.drawString(110 * mm, y, f["numero_autorizacion"])
    y -= 7 * mm
    c.setFont("Helvetica", 7.5)
    for ln in [f"FECHA Y HORA DE AUTORIZACIÓN: {fecha_txt(f['fecha_emision'], 'dmy')} {x['hora']}:12",
               "AMBIENTE: PRODUCCIÓN", "EMISIÓN: NORMAL"]:
        c.drawString(110 * mm, y, ln)
        y -= 5 * mm
    if f["numero_autorizacion"]:
        c.setFont("Helvetica-Bold", 8)
        c.drawString(110 * mm, y, "CLAVE DE ACCESO")
        y -= 4 * mm
        c.setFillColor(colors.black)
        for k, d in enumerate(f["numero_autorizacion"]):        # "código de barras" decorativo
            c.rect(110 * mm + k * 1.6 * mm, y - 9 * mm, (0.4 + int(d) % 3 * 0.35) * mm, 9 * mm, stroke=0, fill=1)
        y -= 13 * mm
        c.setFont("Helvetica", 6.6)
        c.drawString(110 * mm, y, f["numero_autorizacion"])
    # cliente
    y = H - 112 * mm
    c.rect(15 * mm, y - 20 * mm, 180 * mm, 22 * mm)
    c.setFont("Helvetica", 8)
    if f["cliente_nombre"]:
        c.drawString(18 * mm, y - 4 * mm, f"Razón Social / Nombres y Apellidos: {f['cliente_nombre']}")
    if f["cliente_identificacion"]:
        c.drawString(18 * mm, y - 10 * mm, f"Identificación: {f['cliente_identificacion']}")
    c.drawString(18 * mm, y - 16 * mm, f"Fecha Emisión: {fecha_txt(f['fecha_emision'], 'dmy')}")
    c.drawString(120 * mm, y - 16 * mm, "Guía de Remisión:")
    # tabla
    y -= 28 * mm
    cols = [15, 33, 48, 120, 145, 168, 195]
    enc = ["Código", "Cant.", "Descripción", "Precio Unitario", "Descuento", "Precio Total"]
    c.setFont("Helvetica-Bold", 7.5)
    c.rect(15 * mm, y - 2 * mm, 180 * mm, 7 * mm)
    for k, e in enumerate(enc):
        c.drawString((cols[k] + 1.5) * mm, y, e)
    c.setFont("Helvetica", 7.5)
    for it, cod in zip(f["items"], x["codigos"]):
        y -= 6.5 * mm
        c.drawString(16.5 * mm, y, cod)
        c.drawRightString(45 * mm, y, cant(it["cantidad"]))
        c.drawString(49.5 * mm, y, it["descripcion"][:44])
        c.drawRightString(143 * mm, y, num(it["precio_unitario"]))
        c.drawRightString(166 * mm, y, "0.00")
        c.drawRightString(193 * mm, y, num(it["total"]))
        c.line(15 * mm, y - 2 * mm, 195 * mm, y - 2 * mm)
    # información adicional + totales
    y -= 12 * mm
    c.rect(15 * mm, y - 30 * mm, 95 * mm, 34 * mm)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(18 * mm, y, "Información Adicional")
    c.setFont("Helvetica", 7.5)
    c.drawString(18 * mm, y - 6 * mm, f"Email: {x['email']}")
    c.drawString(18 * mm, y - 11 * mm, f"Dirección: {x['ciudad']}")
    c.drawString(18 * mm, y - 18 * mm, "Forma de pago: 20 - OTROS CON UTILIZACIÓN DEL SISTEMA FINANCIERO")
    pct = int(round(x["tasa"] * 100))
    filas = [("SUBTOTAL", f["subtotal"]), ("TOTAL DESCUENTO", f["descuento"]),
             (f"SUBTOTAL {pct}%", round(f["subtotal"] - f["descuento"], 2)), ("SUBTOTAL 0%", 0.0),
             ("SUBTOTAL NO OBJETO DE IVA", 0.0), ("ICE", 0.0), (f"IVA {pct}%", f["iva"]), ("PROPINA", 0.0),
             ("VALOR TOTAL", f["total"])]
    for etq, v in filas:
        c.setFont("Helvetica-Bold" if etq == "VALOR TOTAL" else "Helvetica", 7.5)
        c.rect(118 * mm, y - 2 * mm, 77 * mm, 5.5 * mm)
        c.drawString(120 * mm, y, etq)
        c.drawRightString(193 * mm, y, num(v))
        y -= 5.5 * mm


def _b_moderna(c: canvas.Canvas, f: dict, x: dict):
    """B · Moderna: banda de color, 'Facturar a', tabla de 4 columnas, autorización al pie."""
    W, H = A4
    azul = colors.HexColor("#1f4e79")
    c.setFillColor(azul)
    c.rect(0, H - 38 * mm, W, 38 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    razon = f["razon_social_emisor"] or ""
    c.setFont("Helvetica-Bold", min(17, 17 * 44 / max(1, len(razon))))     # nombres largos: letra más pequeña
    c.drawString(18 * mm, H - 18 * mm, razon)
    c.setFont("Helvetica", 9)
    c.drawString(18 * mm, H - 26 * mm, f"RUC {f['ruc_emisor']}  ·  {x['direccion']}, {x['ciudad']}")
    c.drawString(18 * mm, H - 31 * mm, f"Tel. {x['telefono']}")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(192 * mm, H - 50 * mm, f"FACTURA N° {f['numero_factura']}")
    c.setFont("Helvetica", 9.5)
    c.drawRightString(192 * mm, H - 57 * mm, f"Fecha: {fecha_txt(f['fecha_emision'], 'iso')}")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(18 * mm, H - 50 * mm, "Facturar a:")
    c.setFont("Helvetica", 9.5)
    if f["cliente_nombre"]:
        c.drawString(18 * mm, H - 56 * mm, f["cliente_nombre"])
    if f["cliente_identificacion"]:
        c.drawString(18 * mm, H - 62 * mm, f"C.I./RUC: {f['cliente_identificacion']}")
    y = H - 78 * mm
    c.setFillColor(colors.HexColor("#dce6f1"))
    c.rect(15 * mm, y - 2.5 * mm, 180 * mm, 8 * mm, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(18 * mm, y, "Descripción")
    c.drawRightString(130 * mm, y, "Cant.")
    c.drawRightString(160 * mm, y, "Precio")
    c.drawRightString(192 * mm, y, "Importe")
    c.setFont("Helvetica", 9)
    for it in f["items"]:
        y -= 7.5 * mm
        c.drawString(18 * mm, y, it["descripcion"])
        c.drawRightString(130 * mm, y, cant(it["cantidad"]))
        c.drawRightString(160 * mm, y, f"${num(it['precio_unitario'])}")
        c.drawRightString(192 * mm, y, f"${num(it['total'])}")
    y -= 6 * mm
    c.setStrokeColor(azul)
    c.line(110 * mm, y, 195 * mm, y)
    pct = int(round(x["tasa"] * 100))
    filas = [("Subtotal", f["subtotal"])] + ([("Descuento", -f["descuento"])] if f["descuento"] else []) + \
            [(f"IVA {pct} %", f["iva"]), ("Total a pagar", f["total"])]
    for etq, v in filas:
        y -= 7 * mm
        c.setFont("Helvetica-Bold" if etq == "Total a pagar" else "Helvetica", 10 if etq == "Total a pagar" else 9)
        c.drawString(125 * mm, y, etq)
        c.drawRightString(192 * mm, y, f"{'-' if v < 0 else ''}${num(abs(v))}")
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#555555"))
    if f["numero_autorizacion"]:
        c.drawString(18 * mm, 22 * mm, f"Autorización SRI: {f['numero_autorizacion']}")
    c.drawString(18 * mm, 17 * mm, "Documento generado electrónicamente · Gracias por su compra")


def _c_ticket(f: dict, x: dict, ruta) -> None:
    """C · Ticket de punto de venta (80 mm, Courier); la clave de acceso se parte en dos líneas."""
    lineas: list[tuple[str, str]] = []       # (alineación, texto)
    a = lineas.append
    if f["razon_social_emisor"]:
        for ln in _cortar(f["razon_social_emisor"].upper(), 34):
            a(("c", ln))
    a(("c", f"RUC: {f['ruc_emisor']}"))
    a(("c", f"Matriz: {x['ciudad']}"))
    a(("c", "-" * 38))
    a(("l", f"FACTURA: {f['numero_factura']}"))
    a(("l", f"FECHA: {fecha_txt(f['fecha_emision'], 'dmy')} {x['hora']}"))
    if f["cliente_nombre"]:
        for k, ln in enumerate(_cortar(f["cliente_nombre"].upper(), 29)):
            a(("l", ("CLIENTE: " if k == 0 else "         ") + ln))
    if f["cliente_identificacion"]:
        a(("l", f"CI/RUC: {f['cliente_identificacion']}"))
    a(("c", "-" * 38))
    a(("l", "CANT  DESCRIPCION              TOTAL"))
    for it in f["items"]:
        for k, ln in enumerate(_cortar(it["descripcion"].upper(), 30)):
            a(("l", ln if k == 0 else "  " + ln))
        izq = f"  {cant(it['cantidad'])} x {num(it['precio_unitario'])}"
        a(("l", izq + num(it["total"]).rjust(38 - len(izq))))
    a(("c", "-" * 38))
    pct = int(round(x["tasa"] * 100))
    tot = [("SUBTOTAL", f["subtotal"])] + ([("DESC.", f["descuento"])] if f["descuento"] else []) + \
          [(f"IVA {pct}%", f["iva"]), ("TOTAL USD", f["total"])]
    for etq, v in tot:
        a(("l", etq + num(v).rjust(38 - len(etq))))
    a(("c", "-" * 38))
    if f["numero_autorizacion"]:
        a(("l", "CLAVE DE ACCESO:"))
        a(("l", f["numero_autorizacion"][:25]))
        a(("l", f["numero_autorizacion"][25:]))
    a(("c", "GRACIAS POR SU COMPRA"))
    ancho, alto = 80 * mm, (len(lineas) * 4.2 + 20) * mm
    c = canvas.Canvas(str(ruta), pagesize=(ancho, alto))
    y = alto - 10 * mm
    for al, t in lineas:
        c.setFont("Courier-Bold" if t.startswith("TOTAL USD") else "Courier", 8)
        if al == "c":
            c.drawCentredString(ancho / 2, y, t)
        else:
            c.drawString(4 * mm, y, t)
        y -= 4.2 * mm
    c.save()


def _d_servicios(c: canvas.Canvas, f: dict, x: dict):
    """D · Membrete de profesional/servicios: sin tabla, detalle numerado, fecha en letras."""
    W, H = A4
    c.setFont("Times-Bold", 16)
    c.drawCentredString(W / 2, H - 25 * mm, f["razon_social_emisor"] or "")
    c.setFont("Times-Roman", 10)
    c.drawCentredString(W / 2, H - 31 * mm, f"RUC {f['ruc_emisor']} · {x['direccion']} · {x['telefono']}")
    c.line(20 * mm, H - 35 * mm, W - 20 * mm, H - 35 * mm)
    c.setFont("Times-Bold", 12)
    c.drawString(20 * mm, H - 46 * mm, f"Factura No. {f['numero_factura']}")
    c.setFont("Times-Roman", 11)
    c.drawRightString(W - 20 * mm, H - 46 * mm, f"{x['ciudad']}, {fecha_txt(f['fecha_emision'], 'largo')}")
    y = H - 58 * mm
    if f["cliente_nombre"]:
        c.drawString(20 * mm, y, f"Señor(es): {f['cliente_nombre']}")
        y -= 6 * mm
    if f["cliente_identificacion"]:
        c.drawString(20 * mm, y, f"RUC/CI: {f['cliente_identificacion']}")
        y -= 6 * mm
    y -= 6 * mm
    c.setFont("Times-Bold", 11)
    c.drawString(20 * mm, y, "Por los siguientes conceptos:")
    c.setFont("Times-Roman", 11)
    for k, it in enumerate(f["items"], 1):
        y -= 7 * mm
        c.drawString(24 * mm, y, f"{k}. {it['descripcion']} — {cant(it['cantidad'])} × ${num(it['precio_unitario'])}"
                                 f" = ${num(it['total'])}")
    y -= 12 * mm
    pct = int(round(x["tasa"] * 100))
    filas = [("Subtotal:", f["subtotal"])] + ([("Descuento:", f["descuento"])] if f["descuento"] else []) + \
            [(f"IVA ({pct}%):", f["iva"]), ("TOTAL:", f["total"])]
    for etq, v in filas:
        c.setFont("Times-Bold" if etq == "TOTAL:" else "Times-Roman", 11)
        c.drawString(130 * mm, y, etq)
        c.drawRightString(W - 20 * mm, y, f"${num(v)}")
        y -= 6 * mm
    c.setFont("Times-Italic", 9)
    if f["numero_autorizacion"]:
        c.drawString(20 * mm, 30 * mm, f"Autorización: {f['numero_autorizacion']}")
    c.drawString(20 * mm, 25 * mm, "Forma de pago: transferencia bancaria")


def _e_compacta(c: canvas.Canvas, f: dict, x: dict):
    """E · NUNCA VISTA por la regex ni el prompt: coma decimal (1.234,56), fecha '17-may-2024', otras etiquetas."""
    W, H = A4
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, H - 18 * mm, "COMPROBANTE DE VENTA ELECTRÓNICO")
    c.setFont("Helvetica", 8.5)
    datos = [("Contribuyente", f["razon_social_emisor"]), ("Identificación tributaria", f["ruc_emisor"]),
             ("Comprobante de venta N.º", f["numero_factura"]),
             ("Fecha de emisión", fecha_txt(f["fecha_emision"], "corto")),
             ("Clave de acceso", f["numero_autorizacion"]), ("Adquirente", f["cliente_nombre"]),
             ("Doc. adquirente", f["cliente_identificacion"])]
    y = H - 28 * mm
    for etq, v in datos:
        if v:
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(15 * mm, y, f"{etq}:")
            c.setFont("Helvetica", 8.5)
            c.drawString(62 * mm, y, str(v))
            y -= 5.2 * mm
    y -= 6 * mm
    c.setFont("Helvetica-Bold", 8.5)
    for etq, xx, der in [("Detalle", 15, False), ("Cantidad", 125, True), ("Valor unit.", 155, True),
                         ("Valor total", 195, True)]:
        (c.drawRightString if der else c.drawString)(xx * mm, y, etq)
    c.line(15 * mm, y - 1.5 * mm, 195 * mm, y - 1.5 * mm)
    c.setFont("Helvetica", 8.5)
    for it in f["items"]:
        y -= 5.5 * mm
        c.drawString(15 * mm, y, it["descripcion"])
        c.drawRightString(125 * mm, y, cant(it["cantidad"], "ec"))
        c.drawRightString(155 * mm, y, num(it["precio_unitario"], "ec"))
        c.drawRightString(195 * mm, y, num(it["total"], "ec"))
    c.line(15 * mm, y - 2 * mm, 195 * mm, y - 2 * mm)
    pct = int(round(x["tasa"] * 100))
    filas = [("Suma", f["subtotal"]), ("(-) Descuento", f["descuento"]),
             (f"Base imponible {pct}%", round(f["subtotal"] - f["descuento"], 2)),
             (f"Impuesto (IVA {pct}%)", f["iva"]), ("Importe total USD", f["total"])]
    y -= 8 * mm
    for etq, v in filas:
        c.setFont("Helvetica-Bold" if etq.startswith("Importe") else "Helvetica", 8.5)
        c.drawString(130 * mm, y, etq)
        c.drawRightString(195 * mm, y, num(v, "ec"))
        y -= 5 * mm


def dibujar(f: dict, x: dict, plantilla: str, ruta) -> None:
    if plantilla == "C_ticket":
        return _c_ticket(f, x, ruta)
    c = canvas.Canvas(str(ruta), pagesize=A4)
    {"A_ride_sri": _a_ride, "B_moderna": _b_moderna, "D_servicios": _d_servicios,
     "E_compacta": _e_compacta}[plantilla](c, f, x)
    c.showPage()
    c.save()


# ======================================================================== ruido de escáner
def escanear(ruta_pdf, rng: random.Random, rotacion: int = 0, dpi: int = 200) -> None:
    """Reemplaza el PDF por una 'foto' de la página: sin capa de texto, inclinada, con ruido, borrosa y en JPEG."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(ruta_pdf))
    img = pdf[0].render(scale=dpi / 72).to_pil().convert("L")
    pdf.close()
    angulo = rng.uniform(-2.0, 2.0) + rotacion
    img = img.rotate(angulo, expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    nprng = np.random.default_rng(rng.randint(0, 2**31))
    arr = arr * rng.uniform(0.88, 0.97) + nprng.normal(0, rng.uniform(6, 14), arr.shape) + rng.uniform(5, 15)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(45, 70))
    Image.open(buf).save(str(ruta_pdf), "PDF", resolution=dpi)


# ======================================================================== principal
CAMPOS_OMITIBLES = ["numero_autorizacion", "cliente_identificacion", "cliente_nombre"]


def generar(n: int = config.N_FACTURAS, semilla: int = config.SEMILLA) -> list[dict]:
    rng = random.Random(semilla)
    for d in (config.DIR_FACTURAS, config.DIR_VERDAD):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    filas = []
    for i in range(1, n + 1):
        fid = f"FAC_{i:03d}"
        plantilla = PLANTILLAS[(i - 1) % len(PLANTILLAS)]
        f, x = generar_verdad(rng, i)
        faltantes = []
        if rng.random() < 0.12:
            faltantes = rng.sample(CAMPOS_OMITIBLES, rng.choice([1, 1, 2]))
            for campo in faltantes:
                f[campo] = None
        ruta = config.DIR_FACTURAS / f"{fid}.pdf"
        dibujar(f, x, plantilla, ruta)
        escaneada = rng.random() < 0.15
        rotacion = rng.choice([90, 180, 270]) if escaneada and rng.random() < 0.3 else 0
        if escaneada:
            escanear(ruta, rng, rotacion)
        condicion = ("rotada" if rotacion else "escaneada") if escaneada else "digital"
        (config.DIR_VERDAD / f"{fid}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
        filas.append({"id": fid, "plantilla": plantilla, "condicion": condicion, "rotacion": rotacion,
                      "faltantes": "|".join(faltantes), "n_items": len(f["items"]), "sector": x["sector"],
                      "kb": round(ruta.stat().st_size / 1024, 1)})
    with open(config.CATALOGO, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    copiar_muestras(filas)
    return filas


def copiar_muestras(filas: list[dict]) -> None:
    """10 facturas versionadas (2 por plantilla: una limpia y una escaneada o con faltantes) para demo y pruebas."""
    shutil.rmtree(config.DIR_MUESTRAS, ignore_errors=True)
    config.DIR_MUESTRAS.mkdir(parents=True)
    for p in PLANTILLAS:
        de_p = [r for r in filas if r["plantilla"] == p]
        limpia = next(r for r in de_p if r["condicion"] == "digital" and not r["faltantes"])
        ruidosa = next((r for r in de_p if r["condicion"] != "digital"), None) or \
            next(r for r in de_p if r["faltantes"])
        for r in (limpia, ruidosa):
            shutil.copy(config.DIR_FACTURAS / f"{r['id']}.pdf", config.DIR_MUESTRAS)
            shutil.copy(config.DIR_VERDAD / f"{r['id']}.json", config.DIR_MUESTRAS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=config.N_FACTURAS)
    a = ap.parse_args()
    filas = generar(a.n)
    from collections import Counter
    print(f"{len(filas)} facturas en {config.DIR_FACTURAS}")
    print("plantillas:", dict(Counter(r["plantilla"] for r in filas)))
    print("condición:", dict(Counter(r["condicion"] for r in filas)))
    print("con faltantes:", sum(bool(r["faltantes"]) for r in filas))
