"""
Generación con qwen2.5:7b y salida estructurada (pydantic → JSON Schema → `format` de Ollama).

Reglas que se hacen cumplir en código, no solo en el prompt:
* El modelo solo ve los 4–6 artículos recuperados y solo puede citar sus identificadores.
* Cada cita se verifica contra el índice: una cita a un artículo que no se le mostró se elimina y se cuenta como inventada.
* Rechazo en dos capas: (1) si el mejor puntaje del re-ranker está bajo el umbral, no se llama al LLM;
  (2) el LLM puede responder `cubierto_por_la_ley = false`.
"""
from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from . import config
from .llm import LLM, ErrorLLM, extraer_json
from .recuperar import Buscador
from .texto import ventana_relevante

NO_CUBIERTO = ("La normativa cargada (Código del Trabajo y artículos laborales de la Constitución) no responde esa "
               "pregunta. Consulte al Ministerio del Trabajo o a un abogado laboralista.")


class SalidaLLM(BaseModel):
    """Orden de los campos = orden en que el modelo los genera: primero elige la evidencia, luego decide y redacta."""
    articulos_citados: list[str] = Field(description="Identificadores de los fragmentos usados, p. ej. ['CT-69']")
    cubierto_por_la_ley: bool = Field(description="false si los fragmentos no contienen la respuesta")
    # max_length entra al JSON Schema y Ollama lo impone por gramática: un 7B tiende a copiar el artículo entero
    respuesta: str = Field(max_length=600, description="Respuesta en español, breve, solo con lo que dicen los fragmentos")
    confianza: Literal["alta", "media", "baja"]


class Respuesta(BaseModel):
    pregunta: str
    respuesta: str
    articulos_citados: list[str]
    citas_eliminadas: list[str] = []      # citas del modelo que no estaban en el contexto (inventadas)
    cubierto_por_la_ley: bool
    confianza: Literal["alta", "media", "baja"]
    rechazo_por_umbral: bool = False
    contexto: list[str]                   # ids de los artículos que leyó el modelo
    puntaje_max_rerank: float | None = None
    segundos_recuperacion: float = 0.0
    segundos_llm: float = 0.0
    desde_grabacion: bool = False


SISTEMA = """Eres un asistente de la ley laboral del Ecuador. Respondes SOLO con los fragmentos de ley que se te dan.
Reglas:
1. Usa únicamente la información de los fragmentos. No uses conocimiento propio, no supongas montos ni plazos.
2. Cita los identificadores exactos de los fragmentos que usaste (por ejemplo "CT-69"). Nunca cites otro.
3. Si los fragmentos no contienen la respuesta, pon cubierto_por_la_ley = false, articulos_citados = [] y responde:
   "La ley cargada no cubre esa pregunta."
4. Responde en español, en 1 a 3 oraciones (máximo 70 palabras), mencionando el artículo ("según el art. 69 del
   Código del Trabajo..."). No copies el artículo entero: responde solo lo que se pregunta.
5. confianza: "alta" si un fragmento lo dice literalmente; "media" si hay que combinar fragmentos; "baja" si es dudoso."""


def formatear_contexto(articulos: list[dict], pregunta: str, max_chars: int = 1800) -> str:
    bloques = []
    for a in articulos:
        texto = ventana_relevante(a["texto"], pregunta, max_chars)
        cab = f"[{a['id']}] {a['fuente']}, art. {a['numero']}"
        if a.get("epigrafe"):
            cab += f" ({a['epigrafe']})"
        bloques.append(f"{cab}\n{texto}")
    return "\n\n".join(bloques)


class Asistente:
    def __init__(self, buscador: Buscador | None = None, llm: LLM | None = None,
                 umbral_rerank: float | None = None, metodo: str = "rerank", k: int = config.K_CONTEXTO,
                 sinonimos: bool = True):
        self.buscador = buscador or Buscador()
        self.llm = llm or LLM()
        self.umbral = umbral_rerank
        self.metodo = metodo
        self.k = k
        self.sinonimos = sinonimos

    def responder(self, pregunta: str) -> Respuesta:
        t0 = time.perf_counter()
        hits = self.buscador.buscar(pregunta, self.metodo, self.k, self.sinonimos)
        t_rec = time.perf_counter() - t0
        ids = [h.id for h in hits]
        pmax = hits[0].puntaje if hits and self.metodo == "rerank" else None
        if self.umbral is not None and pmax is not None and pmax < self.umbral:
            return Respuesta(pregunta=pregunta, respuesta=NO_CUBIERTO, articulos_citados=[], cubierto_por_la_ley=False,
                             confianza="alta", rechazo_por_umbral=True, contexto=ids, puntaje_max_rerank=pmax,
                             segundos_recuperacion=round(t_rec, 3))
        articulos = [self.buscador.por_id[i] for i in ids]
        mensaje = f"Fragmentos:\n\n{formatear_contexto(articulos, pregunta)}\n\nPregunta: {pregunta}"
        salida = self.llm.chat(SISTEMA, [{"role": "user", "content": mensaje}],
                               json_schema=SalidaLLM.model_json_schema(), max_tokens=500)
        try:
            datos = SalidaLLM.model_validate(extraer_json(salida["texto"]))
        except Exception as e:  # noqa: BLE001
            raise ErrorLLM(f"Salida inválida del modelo: {salida['texto'][:200]}") from e
        validas = [c for c in dict.fromkeys(datos.articulos_citados) if c in ids]
        inventadas = [c for c in datos.articulos_citados if c not in ids]
        cubierto = datos.cubierto_por_la_ley and bool(validas)
        return Respuesta(
            pregunta=pregunta,
            respuesta=datos.respuesta if cubierto else NO_CUBIERTO,
            articulos_citados=validas if cubierto else [],
            citas_eliminadas=inventadas,
            cubierto_por_la_ley=cubierto,
            confianza=datos.confianza,
            contexto=ids,
            puntaje_max_rerank=pmax,
            segundos_recuperacion=round(t_rec, 3),
            segundos_llm=salida.get("segundos", 0.0),
            desde_grabacion=self.llm.uso.desde_grabacion > 0,
        )
