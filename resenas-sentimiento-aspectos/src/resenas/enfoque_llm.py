"""
Enfoque 3 · LLM local (qwen2.5:7b en Ollama) con salida estructurada.

El JSON Schema obliga al modelo a devolver una lista de {cita, aspecto, polaridad} con aspectos y polaridades de un
catálogo cerrado y el sentimiento global. pydantic valida la respuesta; las citas que no aparecen literalmente en la
reseña se cuentan (alucinación de cita) pero no se descartan: la métrica de aspectos no depende de la cita.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from . import config
from .esquema import AnalisisResena, esquema_ollama
from .llm import LLM, ErrorLLM, extraer_json
from .texto import cita_en_texto, texto_completo

_CATALOGO = "\n".join(f"- {a}: {d}" for a, d in config.ASPECTOS.items())

SISTEMA = f"""Eres un analista de reseñas de productos (teléfonos y accesorios). Lee la reseña (en inglés) y
devuelve los aspectos del producto sobre los que el autor opina, con su polaridad y la cita literal que lo justifica.

Catálogo cerrado de aspectos:
{_CATALOGO}

Reglas:
1. Solo opiniones del autor sobre el producto comprado. Cada aspecto aparece una sola vez.
2. Elogios genéricos ("great product", "works as described", "love it") NO son aspectos: solo cuentan para el
   sentimiento global. "calidad" exige hablar de fabricación, materiales o durabilidad ("sturdy", "broke in a week").
3. "Dejó de funcionar" → calidad negativo. "Llegó roto/usado/incompleto" → envio negativo.
4. "I returned it" sin hablar del proceso no es atencion; reembolso, garantía o trato del vendedor sí.
5. Protector de pantalla: no cubre los bordes → ajuste; burbujas o claridad → pantalla; difícil de aplicar →
   facilidad_uso. Protegió de una caída → proteccion.
6. polaridad: positivo, negativo o neutro (mencionado sin valorar o con valoración que se anula).
7. cita: copia literal y corta de la reseña, en inglés, sin traducir.
8. sentimiento_global: la opinión general del autor (no adivines las estrellas). neutro = mixto sin lado claro.
Si la reseña no opina sobre ningún aspecto del catálogo, devuelve una lista vacía."""


def mensaje_usuario(titulo: str, texto: str) -> str:
    return f"Reseña:\n{texto_completo(titulo, texto)}"


def analizar(titulo: str, texto: str, llm: LLM | None = None) -> dict:
    """Devuelve {"sentimiento_global", "aspectos": [...], "citas_no_literales", "segundos", "tokens_salida", "error"}."""
    llm = llm or LLM()
    antes = (llm.uso.segundos, llm.uso.tokens_salida)
    try:
        crudo = llm.chat(SISTEMA, [{"role": "user", "content": mensaje_usuario(titulo, texto)}],
                         json_schema=esquema_ollama(), max_tokens=500)
        analisis = AnalisisResena.model_validate(extraer_json(crudo))
        error = None
    except (ErrorLLM, ValidationError, json.JSONDecodeError) as e:
        if isinstance(e, ErrorLLM) and ("conexión" in str(e) or "Modo demo" in str(e)):
            raise   # sin Ollama no tiene sentido seguir: no es un error del modelo
        analisis, error = AnalisisResena(sentimiento_global="neutro"), str(e)[:300]
    completo = texto_completo(titulo, texto)
    aspectos = [m.model_dump() for m in analisis.aspectos]
    return {
        "sentimiento_global": analisis.sentimiento_global,
        "aspectos": aspectos,
        "citas_no_literales": sum(not cita_en_texto(m["cita"], completo) for m in aspectos),
        "segundos": round(llm.uso.segundos - antes[0], 2),
        "tokens_salida": llm.uso.tokens_salida - antes[1],
        "error": error,
    }
