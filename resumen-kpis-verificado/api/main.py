"""API opcional (FastAPI): capa de KPIs, verificador e informes.

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpis import capa_kpis, config, generadores, llm, verificador  # noqa: E402

app = FastAPI(title="Resúmenes ejecutivos de KPIs verificados",
              description="Banco Mundial (WDI, CC BY 4.0) · 18 países de América Latina · qwen2.5:7b local")


class PedidoVerificar(BaseModel):
    texto: str
    pais: str = "ECU"
    anio: int = 2024


def _validar(pais: str, anio: int) -> None:
    if pais not in config.PAISES:
        raise HTTPException(404, f"País no soportado: {pais}")
    if not 2001 <= anio <= config.ANIO_FIN:
        raise HTTPException(404, f"Año fuera de rango: {anio}")


@app.get("/salud")
def salud() -> dict:
    return {"estado": "ok", "paises": len(config.PAISES), "indicadores": len(config.INDICADORES)}


@app.get("/kpis/{pais}/{anio}")
def kpis(pais: str, anio: int) -> dict:
    _validar(pais, anio)
    return {"pais": pais, "anio": anio, "hechos": [h.dict() for h in capa_kpis.hechos(pais, anio)]}


@app.post("/verificar")
def verificar(p: PedidoVerificar) -> dict:
    _validar(p.pais, p.anio)
    return verificador.verificar_markdown(p.texto, p.pais, p.anio)


@app.get("/informe/{pais}/{anio}")
def informe(pais: str, anio: int, enfoque: str = "plantilla", version: str = "v2") -> dict:
    _validar(pais, anio)
    if enfoque not in generadores.ENFOQUES:
        raise HTTPException(400, f"Enfoque inválido: {enfoque}")
    try:
        r = generadores.generar(enfoque, pais, anio, version)
    except llm.SinGrabacion as e:
        raise HTTPException(409, str(e)) from e
    return {k: r[k] for k in ("enfoque", "version", "pais", "anio", "texto", "verificacion", "segundos_llm",
                              "llamadas_llm")}
