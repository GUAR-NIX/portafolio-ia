"""Genera README.md y la sección de resultados de la guía desde reports/resultados.json.

    python -m kpis.informe

Las plantillas están en scripts/plantillas/. Marcadores:
  {{pct:ruta}}  porcentaje con 1 decimal («12,3 %»)     {{ic:ruta}}   IC en % («(4,1–20,5 %)»)
  {{num1:ruta}} número con 1 decimal                    {{num2:ruta}} con 2 decimales
  {{int:ruta}}  entero                                   {{txt:ruta}}  texto tal cual
  {{tabla:nombre}} tabla Markdown armada aquí
La ruta navega el JSON con puntos: prueba.libre.tasa_alucinacion
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import config
from .generadores import ENFOQUES, NOMBRES

PLANTILLAS = config.RAIZ / "scripts" / "plantillas"


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def ic(par) -> str:
    return f"({pct(par[0]).replace(' %', '')}–{pct(par[1])})"


def num(x: float, d: int) -> str:
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def _ruta(res: dict, ruta: str):
    obj = res
    for parte in ruta.split("."):
        obj = obj[int(parte)] if isinstance(obj, list) else obj[parte]
    return obj


def _tabla_prueba(res: dict) -> str:
    filas = ["| Enfoque | Cifras (por informe) | Alucinadas (IC 95 %) | Incorrectas | Sin fuente | Informes sin ningún "
             "error | Cobertura de hallazgos (IC 95 %) | Lecturas engañosas detectadas | Llamadas al LLM | "
             "Tiempo por informe |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in ENFOQUES:
        f = res["prueba"][e]
        nombre = f"**{NOMBRES[e]}**" if e == "verificado" else NOMBRES[e]
        tiempo = "< 1 s" if f["segundos_por_informe"] < 1 else f"{num(f['segundos_por_informe'], 0)} s"
        filas.append(f"| {nombre} | {f['cifras']} ({num(f['cifras_por_informe'], 1)}) | "
                     f"{pct(f['tasa_alucinacion'])} {ic(f['tasa_alucinacion_ic'])} | {f['incorrectas']} | "
                     f"{f['sin_fuente']} | {pct(f['informes_sin_error'])} | {pct(f['cobertura'])} "
                     f"{ic(f['cobertura_ic'])} | {f['comparaciones_mal_n']} de {f['comparaciones']} revisadas | "
                     f"{num(f['llamadas_llm'], 2)} | {tiempo} |")
    return "\n".join(filas)


def _tabla_validacion(res: dict) -> str:
    filas = ["| Enfoque · prompt | Cifras (por informe) | Alucinadas (IC 95 %) | Correctas por informe | Referencias "
             "JSON correctas | Referencias pegadas en el texto (por informe) | Cobertura |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for clave in ["estructurado_v1", "estructurado_v2", "verificado_v1", "verificado_v2", "libre", "plantilla"]:
        f = res["validacion"][clave]
        enf, _, ver = clave.partition("_")
        etiqueta = NOMBRES[enf] + (f" · `{ver}`" if ver else (f" · `{res['version_prompt_elegida']}`"
                                                                 if enf == "libre" else ""))
        refs = "—" if f["referencias_ok"] != f["referencias_ok"] else pct(f["referencias_ok"])
        filas.append(f"| {etiqueta} | {f['cifras']} ({num(f['cifras_por_informe'], 1)}) | "
                     f"{pct(f['tasa_alucinacion'])} {ic(f['tasa_alucinacion_ic'])} | "
                     f"{num(f['correctas_por_informe'], 1)} | {refs} | {num(f['refs_en_linea'], 1)} | "
                     f"{pct(f['cobertura'])} |")
    return "\n".join(filas)


def _tabla_pareadas(res: dict) -> str:
    nombre_m = {"tasa_alucinacion": "Cifras alucinadas", "cobertura": "Cobertura"}
    filas = ["| Comparación (prueba, mismos 12 informes) | Métrica | Diferencia | IC 95 % |", "|---|---|---:|---:|"]
    for p in res["pareadas_prueba"]:
        filas.append(f"| {NOMBRES[p['a']]} − {NOMBRES[p['b']]} | {nombre_m[p['metrica']]} | "
                     f"{num(p['diferencia'] * 100, 1)} pp | {num(p['ic_inf'] * 100, 1)} a "
                     f"{num(p['ic_sup'] * 100, 1)} pp |")
    return "\n".join(filas)


def _tabla_legibilidad(res: dict) -> str:
    leg = res["legibilidad"]
    filas = ["| Enfoque | Claridad | Concisión | Utilidad para decidir | Tono ejecutivo | **Media** |",
             "|---|---:|---:|---:|---:|---:|"]
    for e in ENFOQUES:
        f = leg[e]
        filas.append(f"| {NOMBRES[e]} | " + " | ".join(num(f[k], 2) for k in ("claridad", "concision", "utilidad",
                                                                               "tono")) + f" | **{num(f['total'], 2)}** |")
    return "\n".join(filas)


def _tabla_auditoria(res: dict) -> str:
    a = pd.read_csv(config.DATA / "auditoria_verificador.csv")
    t = pd.crosstab(a["estado_verificador"], a["juicio_humano"])
    cols = [c for c in ("correcta", "incorrecta", "sin_fuente") if c in t.columns]
    filas = ["| Verificador \\ revisión a mano | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for est in ("correcta", "incorrecta", "sin_fuente"):
        if est in t.index:
            filas.append(f"| {est} | " + " | ".join(str(int(t.loc[est].get(c, 0))) for c in cols) + " |")
    return "\n".join(filas)


TABLAS = {"prueba": _tabla_prueba, "validacion": _tabla_validacion, "pareadas": _tabla_pareadas,
          "legibilidad": _tabla_legibilidad, "auditoria": _tabla_auditoria}


def renderizar(texto: str, res: dict) -> str:
    def sustituir(m: re.Match) -> str:
        tipo, arg = m.group(1), m.group(2)
        if tipo == "tabla":
            return TABLAS[arg](res)
        v = _ruta(res, arg)
        return {"pct": lambda: pct(v), "ic": lambda: ic(v), "num1": lambda: num(v, 1), "num2": lambda: num(v, 2),
                "num0": lambda: num(v, 0), "pp": lambda: num(v * 100, 1) + " pp",
                "icpp": lambda: f"{num(v[0] * 100, 1)} a {num(v[1] * 100, 1)} pp",
                "int": lambda: str(int(v)), "txt": lambda: str(v)}[tipo]()
    return re.sub(r"\{\{(\w+):([\w.]+)\}\}", sustituir, texto)


def main() -> None:
    res = json.loads((config.REPORTS / "resultados.json").read_text(encoding="utf-8"))
    readme = renderizar((PLANTILLAS / "README.plantilla.md").read_text(encoding="utf-8"), res)
    (config.RAIZ / "README.md").write_text(readme, encoding="utf-8")
    guia = renderizar((PLANTILLAS / "GUIA.plantilla.md").read_text(encoding="utf-8"), res)
    (config.RAIZ / "docs" / "GUIA_PASO_A_PASO.md").write_text(guia, encoding="utf-8")
    sin = re.findall(r"\{\{[^}]*\}\}", readme + guia)
    if sin:
        raise SystemExit(f"Marcadores sin resolver: {sin}")
    print("README.md y docs/GUIA_PASO_A_PASO.md generados")


if __name__ == "__main__":
    main()
