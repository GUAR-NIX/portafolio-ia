"""Prepara los dos archivos que se completan a mano (una sola vez, tras correr la evaluación):

1. data/auditoria_verificador.csv: todas las cifras que el verificador marcó como incorrectas o sin
   fuente en prueba, más una muestra aleatoria de 40 marcadas como correctas. La columna
   `juicio_humano` se llena leyendo la oración y la tabla de KPIs (correcta / incorrecta / sin_fuente).
2. data/rubrica_legibilidad.csv + data/rubrica_mapa.json: 6 informes de prueba × 4 enfoques, con ids
   barajados para puntuar a ciegas (sin saber el enfoque) con la rúbrica de docs/GUIA_PASO_A_PASO.md.
   Los textos a leer quedan en data/rubrica_textos.md.

    python scripts/preparar_revision_manual.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from kpis import config  # noqa: E402

SEMILLA = 7
PAISES_RUBRICA = ["ECU", "BRA", "COL", "HND", "PAN", "URY"]


def auditoria() -> None:
    destino = config.DATA / "auditoria_verificador.csv"
    if destino.exists():
        print("ya existe", destino.name)
        return
    c = pd.read_csv(config.REPORTS / "detalle_cifras.csv")
    c = c[(c["conjunto"] == "prueba") & (c["enfoque"] != "plantilla")]
    malas = c[c["estado"] != "correcta"]
    buenas = c[c["estado"] == "correcta"].sample(40, random_state=SEMILLA)
    a = pd.concat([malas, buenas])[["enfoque", "pais", "anio", "seccion", "cifra", "estado", "valor_real", "hecho",
                                   "oracion"]]
    a = a.rename(columns={"estado": "estado_verificador"}).assign(juicio_humano="", nota="")
    a.to_csv(destino, index=False)
    print(destino.name, len(a))


def rubrica() -> None:
    if config.RUBRICA.exists():
        print("ya existe", config.RUBRICA.name)
        return
    textos = json.loads((config.REPORTS / "informes_generados.json").read_text(encoding="utf-8"))
    sel = [t for t in textos if t["conjunto"] == "prueba" and t["pais"] in PAISES_RUBRICA]
    random.Random(SEMILLA).shuffle(sel)
    mapa, md, filas = {}, ["# Textos para la rúbrica (a ciegas)\n"], []
    for k, t in enumerate(sel, 1):
        i = f"R{k:02d}"
        mapa[i] = {"enfoque": t["enfoque"], "pais": t["pais"], "anio": t["anio"]}
        md += [f"\n---\n\n## {i} · {config.PAISES[t['pais']]} {t['anio']}\n", t["texto"]]
        filas.append({"id": i, "claridad": "", "concision": "", "utilidad": "", "tono": "", "comentario": ""})
    (config.DATA / "rubrica_mapa.json").write_text(json.dumps(mapa, ensure_ascii=False, indent=1), encoding="utf-8")
    (config.DATA / "rubrica_textos.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(filas).to_csv(config.RUBRICA, index=False)
    print(config.RUBRICA.name, len(filas))


if __name__ == "__main__":
    auditoria()
    rubrica()
