"""API REST FastAPI para el Agente de Datos Abiertos (Tool-Calling sobre Banco Mundial y Open-Meteo)."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agente import config, datos, evaluacion, herramientas, motor_agente

app = FastAPI(
    title="API de Agente de Datos Abiertos (Banco Mundial + Open-Meteo)",
    description=(
        "Agente con llamadas nativas a herramientas (qwen2.5:7b vía Ollama): el LLM elige herramientas "
        "y argumentos en un bucle de hasta 3 turnos; el código valida con Pydantic, verifica el anclaje "
        "de cada argumento en la pregunta y ejecuta sobre la instantánea versionada o las APIs en vivo."
    ),
    version="2.0.0",
)


class SolicitudConsulta(BaseModel):
    id_consulta: str = Field(default="REQ_001", description="Identificador de trazabilidad de la consulta")
    pregunta: str = Field(
        default="¿Cuál fue el PIB per cápita en USD de Ecuador (ECU) entre 2020 y 2023 según el Banco Mundial?",
        min_length=5,
        max_length=2000,
        description="Pregunta en lenguaje natural sobre economía o clima",
    )
    modo_datos: Literal["instantanea", "vivo"] = Field(
        default="instantanea",
        description="instantanea: archivo versionado (BM 1990-2024, clima 2020-2024); vivo: APIs HTTP con caché",
    )
    enfoque: Literal[
        "tool_calling_nativo",
        "react_texto",
        "enrutador_reglas",
        "zeroshot_sin_tools",
    ] = Field(
        default="tool_calling_nativo",
        description="Arquitectura del agente a ejecutar",
    )


class LlamadaTrazaItem(BaseModel):
    herramienta: str = Field(..., description="Nombre de la herramienta invocada")
    argumentos: dict[str, Any] = Field(default_factory=dict, description="Argumentos pasados a la herramienta")


class SolicitudEvaluarTraza(BaseModel):
    pregunta: str = Field(..., min_length=5, description="Pregunta original del usuario")
    llamadas_herramientas: list[LlamadaTrazaItem] = Field(
        default_factory=list,
        description="Secuencia de llamadas a herramientas ejecutadas por el agente",
    )
    respuesta_final: str = Field(..., min_length=1, description="Respuesta final generada por el agente")


@app.get("/health")
@app.get("/salud")
def health() -> dict[str, Any]:
    return {
        "estado": "ok",
        "servicio": "agente_datos_abiertos_tool_calling",
        "version": "2.0.0",
        "modo_datos": config.MODO_DATOS,
        "version_prompt": config.VERSION_PROMPT_AGENTE,
        "modelo_llm": config.MODELO_LLM,
        "n_herramientas": len(herramientas.REGISTRO_HERRAMIENTAS),
    }


@app.get("/herramientas")
def listar_herramientas() -> dict[str, Any]:
    return {
        "total_herramientas": len(herramientas.REGISTRO_HERRAMIENTAS),
        "esquemas_ollama": herramientas.obtener_esquemas_ollama(config.VERSION_PROMPT_AGENTE),
        "paises_soportados": config.PAISES,
        "indicadores_banco_mundial": config.INDICADORES_BM,
        "ciudades_open_meteo": config.CIUDADES_CLIMA,
        "variables_climaticas": config.VARIABLES_CLIMA,
    }


@app.post("/consultar")
def consultar(req: SolicitudConsulta) -> dict[str, Any]:
    try:
        datos.establecer_modo(req.modo_datos)
        salida = motor_agente.consultar_agente(pregunta=req.pregunta, enfoque=req.enfoque)
        auditoria = evaluacion.evaluar_traza_externa(
            pregunta=req.pregunta,
            llamadas_herramientas=salida["llamadas_herramientas"],
            respuesta_final=salida["respuesta_final"],
        )
        return {
            "id_consulta": req.id_consulta,
            "pregunta": req.pregunta,
            "enfoque": req.enfoque,
            "se_abstuvo": salida["se_abstuvo"],
            "llamadas_herramientas": salida["llamadas_herramientas"],
            "respuesta_final": salida["respuesta_final"],
            "texto_llm": salida.get("texto_llm", ""),
            "rastro_pasos_llm": salida.get("pasos", []),
            "n_pasos_llm": salida.get("n_pasos_llm"),
            "modo_datos": req.modo_datos,
            "reintentos_validacion": salida["reintentos_validacion"],
            "fidelidad_numerica_groundedness": auditoria["fidelidad_numerica_groundedness"],
            "validez_esquema_pydantic": auditoria["validez_esquema_pydantic"],
            "latencia_ms": salida["latencia_ms"],
            "tokens_prompt": salida["tokens_prompt"],
            "tokens_salida": salida["tokens_salida"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # Ollama apagado o API pública caída
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        datos.establecer_modo("instantanea")


@app.post("/evaluar_traza")
def evaluar_traza(req: SolicitudEvaluarTraza) -> dict[str, Any]:
    lista_llamadas = [ll.model_dump() for ll in req.llamadas_herramientas]
    return evaluacion.evaluar_traza_externa(
        pregunta=req.pregunta,
        llamadas_herramientas=lista_llamadas,
        respuesta_final=req.respuesta_final,
    )
