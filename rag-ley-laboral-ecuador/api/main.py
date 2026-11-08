"""
API del asistente laboral.

    uvicorn api.main:app --port 8000
    RAG_PROVEEDOR=grabado uvicorn api.main:app     # demo sin Ollama (solo preguntas grabadas)

GET  /salud                 estado del índice y del LLM
POST /preguntar             {"pregunta": "..."} → respuesta, artículos citados (con su texto), confianza
GET  /articulo/{id}         texto completo de un artículo (las citas de la app enlazan aquí)
GET  /buscar?q=...&metodo=  solo recuperación (bm25 | vectorial | hibrido | rerank)
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_laboral import config
from rag_laboral.generar import Asistente
from rag_laboral.llm import ErrorLLM

app = FastAPI(title="Asistente de la ley laboral del Ecuador (RAG híbrido)", version="1.0.0",
              description="Responde solo con artículos del Código del Trabajo y la Constitución, citándolos. "
                          "No reemplaza la asesoría de un abogado.")


@lru_cache(maxsize=1)
def asistente() -> Asistente:
    return Asistente(umbral_rerank=umbral_configurado())


def umbral_configurado() -> float | None:
    import json
    ruta = config.REPORTS / "generacion.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8")).get("umbral", {}).get("valor")
    return None


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=3, max_length=500)


def articulo_publico(a: dict) -> dict:
    return {k: a.get(k, "") for k in ("id", "fuente", "numero", "epigrafe", "titulo", "capitulo", "seccion", "texto")}


@app.get("/salud")
def salud():
    a = asistente()
    return {"articulos": len(a.buscador.articulos), "llm": a.llm.disponible(), "umbral_rerank": a.umbral}


@app.post("/preguntar")
def preguntar(p: Pregunta):
    a = asistente()
    try:
        r = a.responder(p.pregunta)
    except (ErrorLLM, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    citas = [articulo_publico(a.buscador.por_id[c]) for c in r.articulos_citados]
    return {**r.model_dump(), "citas": citas,
            "aviso": "Información general basada en el texto de la ley; no reemplaza a un abogado."}


@app.get("/articulo/{id_articulo}")
def articulo(id_articulo: str):
    a = asistente().buscador.por_id.get(id_articulo)
    if a is None:
        raise HTTPException(status_code=404, detail="Artículo no encontrado en el índice")
    return articulo_publico(a)


@app.get("/buscar")
def buscar(q: str, metodo: str = "rerank", k: int = 5):
    if metodo not in ("bm25", "vectorial", "hibrido", "rerank"):
        raise HTTPException(status_code=422, detail="metodo inválido")
    b = asistente().buscador
    try:
        hits = b.buscar(q, metodo, min(k, 20))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return [{"id": h.id, "puntaje": round(h.puntaje, 4), "epigrafe": b.por_id[h.id].get("epigrafe", "")} for h in hits]
