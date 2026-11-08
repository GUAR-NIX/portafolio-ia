"""
PDF → texto limpio → un fragmento por ARTÍCULO (no por número de caracteres), con su ubicación como metadatos.

    python -m rag_laboral.extraer          → data/articulos.jsonl y reports/extraccion.json

Limpieza:
* encabezados y pies de página («Página 4 de 159», «LEXIS FINDER - www.lexis.com.ec»);
* notas editoriales, historial de reformas («Artículo sustituido por…»), CONCORDANCIAS y JURISPRUDENCIA: NO son texto
  de la ley → se guardan aparte en `notas` y no se indexan ni se pueden citar;
* palabras partidas por pypdf («traba jador», «e l artículo»): se unen si la palabra unida aparece en el corpus;
* guiones de corte de línea y saltos dentro de un párrafo.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from pypdf import PdfReader

from . import config

# Artículos de la Constitución que se incluyen (trabajo, grupos de atención prioritaria en lo laboral, seguridad social)
ARTS_CONSTITUCION = {33, 34, 43, 47, 48, 66, 325, 326, 327, 328, 329, 330, 331, 332, 333, 367, 368, 369, 370, 371}

RE_ART = re.compile(r"^Art\.\s*(\d+(?:\.\d+)?|\(\.\.\.\)|\.\.\.)\s*(?:\.?\s*-+\s*(.*))?$")
# Solo «palabra + ordinal» en su propio renglón («TÍTULO I», «Capítulo sexto», «Parágrafo 3ro.»): el nombre viene después
RE_ENCABEZADO = re.compile(r"^-?\s*(T[ÍI]TULO|CAP[ÍI]TULO|Cap[íi]tulo|Par[áa]grafo|PAR[ÁA]GRAFO|Secci[óo]n|SECCI[ÓO]N)\s+[\wº]+\.?\s*$")
RE_NOTA = re.compile(
    r"^(Notas?\s*:|NOTAS?\s*:|Concordancias\s*:|CONCORDANCIAS\s*:|Jurisprudencia\s*:|JURISPRUDENCIA\s*:"
    r"|Art[íi]culo (sustituido|agregado|reformado|derogado|renumerado|interpretado|declarado|suprimido)"
    r"|(Inciso|Numeral|Literal|P[áa]rrafo)s? .{0,60}(sustituido|agregado|reformado|derogado|suprimido))")
RE_FIN_ARTICULADO = re.compile(
    r"^(FUENTES DE LA|DISPOSICI[OÓ]N(ES)?\s+(GENERAL|TRANSITORIA|FINAL|DEROGATORIA|REFORMATORIA)|"
    r"Disposici[oó]n(es)?\s+(General|Transitoria|Final|Derogatoria|Reformatoria)[a-z]*\s*$)")
RE_PAGINA = re.compile(
    r"^(P[áa]gina\s*\d+\s*de\s*\d+|.*LEXIS FINDER.*|.*www\.lexis\.com\.ec.*|CONSTITUCI[ÓO]N DE LA REP[ÚU]BLICA DEL ECUADOR - P[áa]gina \d+)$",
    re.I)


def leer_paginas(ruta: Path) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(str(ruta)).pages]


def limpiar_lineas(paginas: list[str]) -> list[str]:
    salida = []
    for p in paginas:
        for ln in p.splitlines():
            s = ln.strip()
            if s and not RE_PAGINA.match(s):
                salida.append(s)
    return salida


def frecuencias(lineas: list[str]) -> collections.Counter:
    return collections.Counter(w.lower() for ln in lineas for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", ln))


def reparar_espacios(texto: str, frec: collections.Counter) -> str:
    """«traba jador» → «trabajador» si «trabajador» aparece en el corpus y una de las partes es rara por sí sola."""
    partes = re.split(r"( )", texto)          # se recorre palabra a palabra: «mediante l a presentación»
    salida: list[str] = []
    i = 0
    while i < len(partes):
        a = partes[i]
        if salida and a and salida[-1] == " " and len(salida) >= 2:
            previa = salida[-2]
            if re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", previa) and re.fullmatch(r"[a-záéíóúüñ]+[.,;:]?", a):
                b, cola = re.match(r"([a-záéíóúüñ]+)(.*)", a).groups()
                fj = frec.get((previa + b).lower(), 0)
                if fj >= 1 and min(frec.get(previa.lower(), 0), frec.get(b, 0)) <= max(2, fj // 3):
                    salida[-2:] = [previa + b + cola]
                    i += 1
                    continue
        salida.append(a)
        i += 1
    return "".join(salida)


def unir_parrafo(lineas: list[str]) -> str:
    texto = ""
    for ln in lineas:
        if not texto:
            texto = ln
        elif texto.endswith("-") and not texto.endswith((" -", ".-")):
            texto = texto[:-1] + ln        # palabra cortada por guion
        elif re.match(r"^(\d+[\.\)]|[a-z]\)|-\s)", ln) or texto.endswith((".", ":", ";")) and ln[:1].isupper():
            texto += "\n" + ln             # numeral, literal o párrafo nuevo
        else:
            texto += " " + ln
    return re.sub(r"[ \t]+", " ", texto).strip()


def trocear(lineas: list[str], fuente: str, prefijo: str) -> list[dict]:
    frec = frecuencias(lineas)
    articulos: list[dict] = []
    ubic = {"titulo": "", "capitulo": "", "seccion": ""}
    pendiente: str | None = None         # el renglón siguiente a «Capítulo IV» es su nombre
    actual: dict | None = None
    ley: list[str] = []
    notas: list[str] = []
    en_nota = False
    previo = "0"
    innumerados: dict[str, int] = collections.defaultdict(int)

    def cerrar():
        if actual is None:
            return
        texto = reparar_espacios(unir_parrafo(ley), frec)
        m = re.match(r"^(.{3,160}?)\s*\.\s*-\s*(.*)$", texto, re.S)  # «Vacaciones anuales. - Todo trabajador…»
        epigrafe, cuerpo = (m.group(1).strip(), m.group(2).strip()) if m and "\n" not in m.group(1) else ("", texto)
        if not epigrafe and prefijo == "CT":   # a veces falta el guion: «Derecho al fondo de reserva. Todo trabajador…»
            m2 = re.match(r"^([^.\n]{3,70})\.\s+([A-ZÁÉÍÓÚ].*)$", texto, re.S)
            if m2 and len(m2.group(1).split()) <= 8:
                epigrafe, cuerpo = m2.group(1).strip(), m2.group(2).strip()
        actual.update(epigrafe=epigrafe, texto=cuerpo, notas=unir_parrafo(notas)[:1500])
        articulos.append(actual)

    for ln in lineas:
        if RE_FIN_ARTICULADO.match(ln):    # disposiciones y fuentes de la codificación: no son artículos
            cerrar()
            actual, pendiente = None, None
            continue
        m = RE_ART.match(ln)
        if m:
            cerrar()
            bruto = m.group(1)
            if bruto[0] in "(.":           # «Art. (...)»: artículo innumerado agregado por una reforma
                base = previo.split(".inn")[0]
                innumerados[base] += 1
                num = f"{base}.inn{innumerados[base]}"
            else:
                num = bruto
            previo = num
            actual = {"id": f"{prefijo}-{num}", "fuente": fuente, "numero": num, **ubic}
            ley, notas, en_nota, pendiente = ([m.group(2)] if m.group(2) else []), [], False, None
            continue
        e = RE_ENCABEZADO.match(ln)
        if e and len(ln) < 120:
            tipo = e.group(1).lower()
            clave = "titulo" if tipo.startswith("t") else "capitulo" if tipo.startswith("c") else "seccion"
            ubic[clave] = ln.lstrip("- ").strip()
            if clave == "titulo":
                ubic["capitulo"] = ubic["seccion"] = ""
            elif clave == "capitulo":
                ubic["seccion"] = ""
            pendiente = clave
            continue
        if pendiente:                      # nombre del encabezado
            if len(ln) < 140 and not RE_NOTA.match(ln):
                ubic[pendiente] = f"{ubic[pendiente]}: {ln}"
                pendiente = None
                continue
            pendiente = None
        if actual is None:
            continue
        if RE_NOTA.match(ln):
            # «Nota:» puede ir en medio de una lista (art. 42); CONCORDANCIAS y JURISPRUDENCIA van hasta el final
            en_nota = "bloque" if re.match(r"^(CONCORDANCIAS|JURISPRUDENCIA|Concordancias|Jurisprudencia)", ln) else "nota"
        elif (en_nota == "nota" and notas and notas[-1].endswith(".") and not ln.startswith("Ver ")
              and re.match(r"^(\d+\.\s|[a-zñ]\)\s)", ln)
              and not re.search(r"Registro Oficial|Para leer|de la OIT|publicad[oa] en", ln)):
            en_nota = False                # terminó la nota y sigue el numeral siguiente de la ley
        (notas if en_nota else ley).append(ln)
    cerrar()
    return articulos


def deduplicar(articulos: list[dict]) -> list[dict]:
    """Si un número sale dos veces (p. ej. un artículo reenumerado), el id lleva sufijo para que sea único."""
    vistos: collections.Counter = collections.Counter()
    for a in articulos:
        vistos[a["id"]] += 1
        if vistos[a["id"]] > 1:
            a["id"] = f"{a['id']}#{vistos[a['id']]}"
    return articulos


def procesar() -> dict:
    resumen = {}
    ct = trocear(limpiar_lineas(leer_paginas(config.RAW / "codigo_trabajo.pdf")), "Código del Trabajo", "CT")
    derogados = [a for a in ct if len(a["texto"]) < 40 and re.search(r"derogad", a["texto"] + a["epigrafe"] + a["notas"], re.I)]
    ct = [a for a in ct if a not in derogados and (a["texto"] or a["epigrafe"])]
    resumen["codigo_trabajo"] = {"articulos": len(ct), "derogados_excluidos": len(derogados),
                                 "innumerados": sum(".inn" in a["numero"] for a in ct)}
    todos = list(ct)
    ruta_c = config.RAW / "constitucion.pdf"
    if ruta_c.exists():
        cons = trocear(limpiar_lineas(leer_paginas(ruta_c)), "Constitución de la República", "CRE")
        cons = [a for a in cons if a["numero"].isdigit() and int(a["numero"]) in ARTS_CONSTITUCION]
        resumen["constitucion"] = {"articulos": len(cons)}
        todos += cons
    todos = deduplicar(todos)
    config.ARTICULOS.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in todos) + "\n", encoding="utf-8")
    largos = sorted(len(a["texto"]) for a in todos)
    resumen.update(total=len(todos), caracteres_mediana=largos[len(largos) // 2], caracteres_max=largos[-1],
                   con_notas_editoriales=sum(bool(a["notas"]) for a in todos),
                   duplicados_con_sufijo=sum("#" in a["id"] for a in todos),
                   sin_epigrafe=sum(not a["epigrafe"] for a in todos if a["fuente"].startswith("Cód")))
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "extraccion.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    print(json.dumps(procesar(), ensure_ascii=False, indent=2))
