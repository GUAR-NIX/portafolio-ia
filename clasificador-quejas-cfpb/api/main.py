"""API FastAPI para enrutar reclamos financieros con abstención selectiva a triaje humano."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from quejas import predecir

app = FastAPI(
    title="API de Clasificación y Enrutamiento de Quejas Financieras (CFPB)",
    description=(
        "Clasifica reclamos financieros en 6 mesas especialistas (reporte de crédito, cobranza, "
        "hipotecas, tarjetas, cuentas bancarias, préstamos de consumo) y deriva los casos "
        "ambiguos a triaje humano según un umbral económico de abstención. /enrutar aplica la cascada "
        "TF-IDF → LLM y marca la urgencia (fraude, ejecución hipotecaria, reposesión, amenazas legales)."
    ),
    version="2.0.0",
)


class SolicitudQueja(BaseModel):
    id_queja: str = Field(default="CFPB_001", description="Identificador del ticket o reclamo")
    texto_queja: str = Field(
        default=(
            "I checked my credit report with Equifax and found an inaccurate late payment "
            "and an unauthorized hard inquiry that lowered my credit score by 60 points."
        ),
        min_length=10,
        max_length=10000,
        description="Narrativa en texto libre del cliente",
    )
    enfoque: Literal["auto", "tfidf", "bge_m3"] = Field(
        default="auto",
        description="Motor de clasificación: 'auto' (bge-m3 si está en caché, si no tfidf), 'tfidf' o 'bge_m3'",
    )
    umbral_abstencion: float | None = Field(
        default=None,
        ge=0.20,
        le=0.95,
        description="Umbral mínimo de confianza para enrutar sin triaje humano (por defecto el óptimo económico)",
    )


class SolicitudCascada(BaseModel):
    texto_queja: str = Field(..., min_length=10, max_length=10000, description="Narrativa en texto libre del cliente")
    consultar_ollama: bool = Field(default=False, description="Si la queja va al LLM y no hay respuesta grabada, consultar Ollama en vivo")


class ItemLoteQueja(BaseModel):
    id_queja: str = Field(..., description="Identificador de la queja")
    texto_queja: str = Field(..., min_length=10, max_length=10000)


class SolicitudLoteQuejas(BaseModel):
    quejas: list[ItemLoteQueja] = Field(..., min_length=1, max_length=1000)


@app.get("/salud")
def salud():
    return {"estado": "ok", "servicio": "clasificador_quejas_cfpb", "version": "2.0.0"}


@app.get("/modelo")
def info_modelo():
    return predecir.cargar_metadata()


@app.post("/clasificar")
def clasificar(req: SolicitudQueja):
    try:
        res = predecir.clasificar_queja(
            texto=req.texto_queja,
            enfoque=req.enfoque,
            umbral_abstencion=req.umbral_abstencion,
            permitir_ollama=False,
        )
        res["id_queja"] = req.id_queja
        return res
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/enrutar")
def enrutar(req: SolicitudCascada):
    """Cascada completa: TF-IDF decide si está seguro; si no, el LLM (o triaje humano si no hay LLM). Incluye urgencia."""
    return predecir.enrutar_con_cascada(req.texto_queja, permitir_ollama=req.consultar_ollama)


@app.post("/enrutar_lote")
def enrutar_lote(
    req: SolicitudLoteQuejas,
    enfoque: Literal["auto", "tfidf", "bge_m3"] = Query(default="auto"),
    umbral_abstencion: float | None = Query(default=None, ge=0.20, le=0.95),
):
    lista = [q.model_dump() for q in req.quejas]
    return predecir.enrutar_lote(
        quejas=lista,
        enfoque=enfoque,
        umbral_abstencion=umbral_abstencion,
    )
