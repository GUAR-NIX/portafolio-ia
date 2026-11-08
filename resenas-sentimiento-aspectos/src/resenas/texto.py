"""Segmentación de reseñas en cláusulas: la unidad que usan los enfoques clásico y de embeddings."""
from __future__ import annotations

import re

_FIN_ORACION = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[.!?])(?=[A-Z])")
# "but", "however", "although"… separan opiniones opuestas: "love the color but it broke in a week"
_CONTRASTE = re.compile(r"\s*(?:,\s*|\s)(?=\b(?:but|however|although|though|except|yet|pero|aunque|sin embargo)\b)",
                        re.I)


def texto_completo(titulo: str, texto: str) -> str:
    """Título y cuerpo en un solo texto; el título suele resumir la opinión ("Broke after 2 weeks")."""
    titulo, texto = (titulo or "").strip(), (texto or "").strip()
    if titulo and not re.search(r"[.!?]$", titulo):
        titulo += "."
    return f"{titulo}\n{texto}".strip()


def clausulas(texto: str, min_car: int = 3) -> list[str]:
    """Divide en oraciones y luego en cláusulas de contraste. Descarta fragmentos sin letras."""
    salida = []
    for oracion in _FIN_ORACION.split(texto or ""):
        for c in _CONTRASTE.split(oracion):
            c = c.strip(" ,;-—")
            if len(c) >= min_car and re.search(r"[A-Za-zÁÉÍÓÚáéíóúñ]", c):
                salida.append(c)
    return salida


def normalizar(t: str) -> str:
    """Minúsculas y espacios simples: para comparar citas con el texto."""
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def cita_en_texto(cita: str, texto: str) -> bool:
    """¿La cita aparece literal en la reseña? (tolerante a mayúsculas, espacios y comillas tipográficas)."""
    def n(s: str) -> str:
        s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return normalizar(s).strip(" .!?,\"'")
    c = n(cita)
    return bool(c) and c in n(texto)


_ES = {"el", "la", "los", "las", "que", "de", "muy", "es", "y", "producto", "con", "para", "por", "pero", "una",
       "excelente", "bueno", "buena", "lo", "se", "su", "llegó", "funciona", "calidad", "precio", "más"}
_EN = {"the", "and", "it", "is", "to", "of", "my", "this", "for", "with", "was", "but", "not", "great", "phone"}


def es_espanol(texto: str) -> bool:
    """Heurística por palabras frecuentes: suficiente para separar las reseñas en español que aparecen en
    Amazon EE. UU. (no se instala un detector de idioma solo para esto)."""
    palabras = re.findall(r"[a-záéíóúñü]+", (texto or "").lower())
    es = sum(p in _ES for p in palabras)
    en = sum(p in _EN for p in palabras)
    return es >= 2 and es > en
