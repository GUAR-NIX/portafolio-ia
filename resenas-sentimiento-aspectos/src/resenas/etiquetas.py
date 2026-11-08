"""
Etiquetas hechas a mano: formato compacto para escribirlas rápido y conversión a JSONL validado.

Formato compacto (una reseña por línea; `#` comenta):

    R000123 | - | calidad-: Cracked after a day ; envio+: shipping was fast

* 2.º campo: sentimiento global `+` positivo, `-` negativo, `0` neutro.
* 3.º campo: menciones separadas por ` ; `, cada una `aspecto` + polaridad (`+`, `-`, `0`) + `:` + cita literal.
  Vacío si la reseña no opina sobre ningún aspecto del catálogo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config
from .esquema import AnalisisResena
from .texto import cita_en_texto, texto_completo

SIMBOLO = {"+": "positivo", "-": "negativo", "0": "neutro"}


def parsear_linea(linea: str) -> dict:
    partes = [p.strip() for p in linea.split("|", 2)]
    if len(partes) < 2:
        raise ValueError(f"línea mal formada: {linea!r}")
    id_, glob = partes[0], partes[1]
    menciones = []
    if len(partes) == 3 and partes[2]:
        for m in partes[2].split(" ; "):
            cabeza, _, cita = m.partition(":")
            cabeza = cabeza.strip()
            aspecto, pol = cabeza[:-1], cabeza[-1]
            if aspecto not in config.ASPECTOS or pol not in SIMBOLO:
                raise ValueError(f"{id_}: mención inválida {m!r}")
            menciones.append({"aspecto": aspecto, "polaridad": SIMBOLO[pol], "cita": cita.strip()})
    d = {"id": id_, "sentimiento_global": SIMBOLO[glob], "aspectos": menciones}
    AnalisisResena.model_validate(d)   # tipos y aspectos del catálogo
    if len({m["aspecto"] for m in menciones}) != len(menciones):
        raise ValueError(f"{id_}: aspecto repetido")
    return d


def compacto_a_jsonl(origen: Path, destino: Path, muestra: dict[str, dict]) -> list[dict]:
    """Convierte el archivo compacto, comprueba que cada cita sea literal y escribe el JSONL."""
    salida, problemas = [], []
    for ln in origen.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        d = parsear_linea(ln)
        r = muestra[d["id"]]
        completo = texto_completo(r["titulo_resena"], r["texto"])
        for m in d["aspectos"]:
            if not cita_en_texto(m["cita"], completo):
                problemas.append(f"{d['id']} {m['aspecto']}: cita no literal {m['cita']!r}")
        salida.append(d)
    ids = [d["id"] for d in salida]
    if len(ids) != len(set(ids)):
        problemas.append("ids repetidos")
    with open(destino, "w", encoding="utf-8") as fh:
        for d in salida:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    for p in problemas:
        print("  ⚠", p)
    return salida


def leer_jsonl(ruta: Path) -> list[dict]:
    return [json.loads(x) for x in Path(ruta).read_text(encoding="utf-8").splitlines() if x.strip()]


def cargar_muestra() -> dict[str, dict]:
    """Reseñas sorteadas para etiquetar: dev + test (inglés y algo de español) y el conjunto en español.

    El texto no se versiona (licencia de los datos): si aún no está en data/raw/, se reconstruye desde la descarga
    y se verifica contra las huellas del índice (FileNotFoundError con instrucciones si falta la descarga)."""
    if not config.MUESTRA_ETIQUETADO.exists():
        from .muestra import reconstruir
        return reconstruir()
    filas = leer_jsonl(config.MUESTRA_ETIQUETADO)
    if config.MUESTRA_ES.exists():
        filas += leer_jsonl(config.MUESTRA_ES)
    return {d["id"]: d for d in filas}


def como_dict(d: dict) -> dict:
    """{"global": pol, "aspectos": {aspecto: pol}}: la forma que usan las métricas."""
    return {"global": d["sentimiento_global"], "aspectos": {m["aspecto"]: m["polaridad"] for m in d["aspectos"]}}


if __name__ == "__main__":
    # python -m resenas.etiquetas data/etiquetas/dev.txt data/etiquetas/dev.jsonl
    sys.stdout.reconfigure(encoding="utf-8")
    res = compacto_a_jsonl(Path(sys.argv[1]), Path(sys.argv[2]), cargar_muestra())
    print(f"{len(res)} reseñas etiquetadas → {sys.argv[2]}")
