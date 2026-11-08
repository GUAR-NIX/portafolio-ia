"""Verificación en vivo: (1) la verdad terreno de validación y prueba contra las APIs reales y
(2) el agente en modo "vivo" sobre unas pocas preguntas (incluida una de clima anterior a 2020).

    python -m agente.verificacion_vivo verdad     # solo red, sin LLM
    python -m agente.verificacion_vivo agente     # red + Ollama (o grabación)
"""
from __future__ import annotations

import json
import sys
from typing import Any

import pandas as pd

from . import clientes_api, config, datos, evaluacion, herramientas, motor_agente

PREGUNTAS_VIVO = [
    "¿Cuánta precipitación acumulada en milímetros se registró en Guayaquil entre 2023-01-01 y 2023-12-31?",
    "Compara el PIB per cápita de México y Costa Rica en 2023.",
    "¿En qué porcentaje cambió el PIB per cápita de Perú entre 2012 y 2022?",
    # Fuera de la instantánea (2020-2024) pero dentro de Open-Meteo Archive (desde 1940):
    "¿Cuál fue la precipitación acumulada en milímetros en Quito entre 1995-01-01 y 1995-12-31 según Open-Meteo?",
    "¿Cuánto llovió en Guayaquil en 2010?",
]


def verificar_verdad_terreno() -> pd.DataFrame:
    """Re-ejecuta cada llamada de datos esperada en modo "vivo" y la compara con la instantánea."""
    filas: list[dict[str, Any]] = []
    vistas: set[str] = set()
    for nombre in ("validacion", "prueba"):
        for q in evaluacion.cargar_conjunto(nombre):
            for ll in q["llamadas_esperadas"]:
                if ll["herramienta"] == "calcular_correlacion_o_cambio":
                    continue
                clave = json.dumps(ll, sort_keys=True, ensure_ascii=False)
                if clave in vistas:
                    continue
                vistas.add(clave)
                resultados = {}
                for modo in ("instantanea", "vivo"):
                    datos.establecer_modo(modo)
                    ej = herramientas.ejecutar_herramienta_segura(ll["herramienta"], ll["argumentos"])
                    resultados[modo] = ej
                datos.establecer_modo("instantanea")
                cif_i = _cifras(resultados["instantanea"])
                cif_v = _cifras(resultados["vivo"])
                difs = [abs(a - b) / max(abs(a), 1e-6) for a, b in zip(cif_i, cif_v)]
                filas.append({
                    "id_pregunta": q["id_pregunta"],
                    "herramienta": ll["herramienta"],
                    "argumentos": json.dumps(ll["argumentos"], ensure_ascii=False),
                    "exito_vivo": resultados["vivo"]["exito"],
                    "n_cifras": len(cif_i),
                    "cifras_instantanea": cif_i,
                    "cifras_vivo": cif_v,
                    "max_dif_relativa_pct": round(100 * max(difs), 3) if difs and len(cif_i) == len(cif_v) else None,
                    "coincide_0_5pct": bool(difs) and len(cif_i) == len(cif_v) and max(difs) <= 0.005,
                    "error_vivo": resultados["vivo"].get("mensaje_error"),
                })
    df = pd.DataFrame(filas)
    df.to_csv(config.REPORTES_DIR / "verificacion_verdad_terreno_vivo.csv", index=False, encoding="utf-8")
    return df


def _cifras(ej: dict[str, Any]) -> list[float]:
    if not ej.get("exito"):
        return []
    r = ej["resultado"]
    if ej["herramienta"] == "consultar_indicador_banco_mundial":
        return [float(v) for v in r["serie_anual"].values()]
    if ej["herramienta"] == "comparar_paises_indicador":
        return [float(x["valor"]) for x in sorted(r["ranking"], key=lambda x: x["codigo_pais"])]
    if ej["herramienta"] == "consultar_clima_historico_ciudad":
        return [float(r["valor_periodo"])]
    m = r["metricas_anuales"]
    return [float(m["temp_media_c"]), float(m["precipitacion_mm"])]


def probar_agente_en_vivo() -> list[dict[str, Any]]:
    """Corre el agente (prompt elegido) en ambos modos sobre PREGUNTAS_VIVO."""
    salida = []
    for pregunta in PREGUNTAS_VIVO:
        fila: dict[str, Any] = {"pregunta": pregunta}
        for modo in ("instantanea", "vivo"):
            datos.establecer_modo(modo)
            r = motor_agente.consultar_agente(pregunta, "tool_calling_nativo")
            fila[modo] = {
                "se_abstuvo": r["se_abstuvo"],
                "llamadas": r["llamadas_herramientas"],
                "errores": [c["mensaje_error"] for p in r["pasos"] for c in p["llamadas"] if not c["exito"]],
                "respuesta_final": r["respuesta_final"],
                "n_pasos_llm": r["n_pasos_llm"],
            }
        datos.establecer_modo("instantanea")
        salida.append(fila)
    fila_stats = {"estadisticas_cliente_http": clientes_api.cliente_global().estadisticas}
    (config.REPORTES_DIR / "prueba_modo_vivo.json").write_text(
        json.dumps({"preguntas": salida, **fila_stats}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return salida


if __name__ == "__main__":
    fase = sys.argv[1] if len(sys.argv) > 1 else ""
    if fase == "verdad":
        df = verificar_verdad_terreno()
        print(df[["id_pregunta", "herramienta", "exito_vivo", "max_dif_relativa_pct", "coincide_0_5pct"]].to_string())
        print("Coinciden:", int(df["coincide_0_5pct"].sum()), "de", len(df))
    elif fase == "agente":
        print(json.dumps(probar_agente_en_vivo(), ensure_ascii=False, indent=1)[:4000])
    else:
        raise SystemExit(__doc__)
