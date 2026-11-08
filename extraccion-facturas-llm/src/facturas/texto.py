"""
PDF → texto.

1. **pypdf** (capa de texto del PDF digital): instantáneo y exacto.
2. Si la página no tiene texto (factura escaneada o foto), **OCR nativo de Windows** (`Windows.Media.Ocr`, el mismo
   motor de la app Fotos/Recortes) vía `winrt`: no hay que instalar Tesseract ni modelos; solo el paquete de idioma
   español de Windows (viene con Windows en español). En Linux/Mac no existe → se informa y el texto queda vacío.
3. Orientación: el OCR de Windows corrige inclinaciones pequeñas pero no páginas giradas 90/180/270°. Si el primer
   intento no encuentra las palabras típicas de una factura, se prueba en las otras 3 orientaciones y se queda la mejor.

El OCR devuelve palabras con su caja; se rearman las filas por coordenada vertical y se respetan los huecos
horizontales, para que una fila de la tabla siga siendo una línea (igual que en el texto de pypdf).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

PALABRAS_CLAVE = re.compile(r"RUC|R\.U\.C|FACTURA|TOTAL|IVA|SUBTOTAL|FECHA|CLIENTE|COMPROBANTE", re.I)


@dataclass
class TextoFactura:
    texto: str
    metodo: str                     # "pypdf" | "ocr" | "sin_texto"
    paginas: int = 1
    rotacion: int = 0               # grados que hubo que girar la imagen para leerla
    avisos: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------- pypdf
def texto_pypdf(ruta) -> tuple[str, int]:
    """Modo 'layout': respeta la posición, así cada fila de la tabla queda en una línea (el modo normal pone cada
    celda en su propia línea y el modelo ya no sabe qué cantidad va con qué precio)."""
    lector = PdfReader(str(ruta))
    partes = [(p.extract_text(extraction_mode="layout") or "") for p in lector.pages]
    return normalizar("\n".join(partes)), len(lector.pages)


def normalizar(texto: str) -> str:
    """Quita líneas vacías y comprime los huecos largos a 4 espacios (menos tokens, se conserva la separación)."""
    lineas = (re.sub(r" {3,}", "    ", ln).strip() for ln in texto.splitlines())
    return "\n".join(ln for ln in lineas if ln)


def imagenes_de_paginas(ruta, dpi: int = 200) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(ruta))
    try:
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()


# ------------------------------------------------------------------------- OCR de Windows
def ocr_disponible() -> bool:
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine
        return OcrEngine.try_create_from_language(Language("es-ES")) is not None or \
            OcrEngine.try_create_from_user_profile_languages() is not None
    except Exception:  # noqa: BLE001  (no es Windows o falta winrt)
        return False


async def _ocr_async(img: Image.Image):
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter

    motor = OcrEngine.try_create_from_language(Language("es-ES")) or OcrEngine.try_create_from_user_profile_languages()
    lim = OcrEngine.max_image_dimension
    if max(img.size) > lim:
        img = img.copy()
        img.thumbnail((lim, lim))
    rgba = img.convert("RGBA")
    escritor = DataWriter()
    escritor.write_bytes(rgba.tobytes())
    bmp = SoftwareBitmap.create_copy_from_buffer(escritor.detach_buffer(), BitmapPixelFormat.RGBA8,
                                                 rgba.width, rgba.height)
    res = await motor.recognize_async(bmp)
    palabras = []
    for linea in res.lines:
        for w in linea.words:
            r = w.bounding_rect
            palabras.append((r.x, r.y, r.width, r.height, w.text))
    return palabras


def _rearmar_lineas(palabras: list[tuple]) -> str:
    """Agrupa palabras por fila (centro vertical) y las ordena por x; un hueco grande = varios espacios."""
    if not palabras:
        return ""
    alto = sorted(p[3] for p in palabras)[len(palabras) // 2] or 10
    filas: list[list[tuple]] = []
    for p in sorted(palabras, key=lambda p: p[1] + p[3] / 2):
        cy = p[1] + p[3] / 2
        if filas and abs(cy - filas[-1][-1][0]) < alto * 0.55:
            filas[-1].append((cy, p))
        else:
            filas.append([(cy, p)])
    lineas = []
    ancho_car = alto * 0.5
    for fila in filas:
        ps = sorted((p for _, p in fila), key=lambda p: p[0])
        s, fin = "", None
        for x, _, w, _, t in ps:
            if fin is not None:
                hueco = x - fin
                s += " " * max(1, min(12, int(hueco / ancho_car))) if hueco > ancho_car * 1.5 else " "
            s += t
            fin = x + w
        lineas.append(s)
    return "\n".join(lineas)


def ocr_imagen(img: Image.Image) -> str:
    return _rearmar_lineas(asyncio.run(_ocr_async(img)))


def _puntaje(texto: str) -> int:
    return 10 * len(PALABRAS_CLAVE.findall(texto)) + len(re.findall(r"\d", texto)) // 10


def ocr_con_orientacion(img: Image.Image) -> tuple[str, int]:
    """OCR en la orientación normal; si no aparecen palabras de factura, prueba 90/180/270 y elige la mejor."""
    texto = ocr_imagen(img)
    if len(PALABRAS_CLAVE.findall(texto)) >= 4:
        return texto, 0
    mejor = (_puntaje(texto), texto, 0)
    for g in (90, 180, 270):
        t = ocr_imagen(img.rotate(g, expand=True, fillcolor="white"))
        mejor = max(mejor, (_puntaje(t), t, g), key=lambda m: m[0])
    return mejor[1], mejor[2]


# ------------------------------------------------------------------------- principal
def extraer_texto(ruta, minimo_caracteres: int = 40) -> TextoFactura:
    ruta = Path(ruta)
    texto, paginas = texto_pypdf(ruta)
    if len(texto) >= minimo_caracteres:
        return TextoFactura(texto, "pypdf", paginas)
    if not ocr_disponible():
        return TextoFactura("", "sin_texto", paginas,
                            avisos=["El PDF no tiene capa de texto y el OCR de Windows no está disponible "
                                    "(requiere Windows 10/11 con el idioma español)."])
    partes, rot = [], 0
    for img in imagenes_de_paginas(ruta):
        t, rot = ocr_con_orientacion(img)
        partes.append(t)
    avisos = [f"Página girada {rot}°: se corrigió antes del OCR."] if rot else []
    return TextoFactura(normalizar("\n".join(partes)), "ocr", paginas, rot, avisos)


if __name__ == "__main__":
    import sys

    r = extraer_texto(sys.argv[1])
    print(f"[{r.metodo} · rot {r.rotacion}] {r.avisos}\n{r.texto}")
