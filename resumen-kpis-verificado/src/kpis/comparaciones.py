"""Chequeo de la *dirección* de las comparaciones, complementario al verificador de cifras.

Una frase puede citar cifras correctas y aun así engañar: «la inflación (4,6 %) se mantuvo por debajo de la
mediana regional (4,3 %)». Este módulo revisa dos tipos de afirmación cualitativa por cláusula:

- **frente a la región**: «por encima / supera / superior» o «por debajo / inferior» de la mediana, contra el
  signo de la brecha con la región;
- **frente al año anterior**: «aumentó / subió / se amplió» o «disminuyó / bajó / se redujo / cayó», contra el
  signo de la variación interanual (no se aplica al crecimiento del PIB, donde «creció» describe el nivel).

Es deliberadamente conservador: si no identifica un solo indicador en la cláusula, no opina. No participa en
los reintentos de (c); se reporta como métrica aparte.
"""
from __future__ import annotations

import re

from . import capa_kpis
from .numeros import normalizar
from .verificador import _indicadores_contexto, verificar_texto

_ARRIBA_REG = re.compile(r"por encima|superior a|supera|superando|super[oó]|m[aá]s alt[oa] que|mayor que|"
                         r"sobre la mediana", re.I)
_ABAJO_REG = re.compile(r"por debajo|inferior a|m[aá]s baj[oa] que|menor que|bajo la mediana|debajo de", re.I)
_REGION = re.compile(r"mediana|regi[oó]n|regional|promedio", re.I)
_SUBE = re.compile(r"\b(?:aument[oóa]|subi[oó]|sube|increment[oóa]|se ampli[oóa]|ampli[oó]|se elev[oó]|"
                   r"repunt[oó]|se aceler[oó])", re.I)
_BAJA = re.compile(r"\b(?:disminu[yí]|baj[oóa]\b|se redujo|redujo|se reduj|cay[oó]|descendi[oó]|se contrajo|"
                   r"retroced|se moder[oó]|se desaceler[oó])", re.I)
_DEFICIT = re.compile(r"d[eé]ficit", re.I)
_CLAUSULAS = re.compile(r"(?<=[.;])\s+|,\s+(?=pero|aunque|mientras|lo que|y\s|manteni|super|situ|ubic|coloc|"
                        r"qued|frente)|\s+y\s+(?=(?:la|el|las|los)\s)", re.I)


def _indicador_clausula(oracion: str, ini: int, fin: int, cifras: list, pais: str, anio: int) -> str | None:
    """Indicador de la cláusula [ini, fin) dentro de su oración: el de la primera cifra de la cláusula que el
    verificador atribuyó con contexto; si no hay, la palabra clave inequívoca más cercana antes del final de
    la cláusula (puede estar antes, en la misma oración: «Las remesas …, superando la mediana …»)."""
    validas = [c for c in cifras if ini <= c.inicio < fin and c.estado == "correcta" and c.hecho_id
               and "atribucion_debil" not in c.marcas]
    if validas:
        m = (_ARRIBA_REG.search(oracion, ini, fin) or _ABAJO_REG.search(oracion, ini, fin)
             or _SUBE.search(oracion, ini, fin) or _BAJA.search(oracion, ini, fin))
        antes = [c for c in validas if m and c.inicio < m.start()]
        return (antes[-1] if antes else validas[0]).hecho_id.split("|")[0]
    grupos = _indicadores_contexto(oracion[:fin], fin)
    if grupos and len(grupos[0]) == 1:
        return grupos[0][0]
    return None


def _hecho_reciente(ind: str, pais: str, anio: int, metrica: str):
    for a in range(anio, anio - capa_kpis.VENTANA_DATO_ESCASO - 1, -1):
        h = capa_kpis.buscar_hecho(ind, pais, a, metrica)
        if h is not None:
            return h
    return None


_CRECE = re.compile(r"crec|creci|expansi", re.I)
_NOMINAL = re.compile(r"nominal|corriente|en d[oó]lares|valor en|usd", re.I)


def _pib_nominal_como_crecimiento(texto: str, cifras: list) -> list[dict]:
    """«El PIB creció 11,5 %» cuando 11,5 % es la variación del PIB en dólares corrientes (no el crecimiento real)."""
    out = []
    for c in cifras:
        if c.estado != "correcta" or not c.hecho_id:
            continue
        ind, _, _, metrica = c.hecho_id.split("|")
        if ind in ("pib_usd", "pib_pc") and metrica == "variacion":
            i0 = max(texto.rfind(".", 0, c.inicio), texto.rfind(",", 0, max(c.inicio - 1, 0)), 0)
            tramo = texto[i0:c.fin + 25]
            if _CRECE.search(tramo) and not _NOMINAL.search(tramo):
                out.append({"clausula": tramo.strip(" .,"), "tipo": "pib_nominal_como_crecimiento", "indicador": ind,
                            "dice": "crecimiento", "real": "variación en USD corrientes", "correcta": False})
    return out


def revisar(texto: str, pais: str, anio: int) -> list[dict]:
    texto = normalizar(texto)
    cifras = verificar_texto(texto, pais, anio)
    salida = _pib_nominal_como_crecimiento(texto, cifras)
    cortes = [0] + [m.end() for m in _CLAUSULAS.finditer(texto)] + [len(texto)]
    for ini, fin in zip(cortes, cortes[1:]):
        clausula = texto[ini:fin].strip()
        if not clausula:
            continue
        ind = None
        if _REGION.search(clausula) and (_ARRIBA_REG.search(clausula) or _ABAJO_REG.search(clausula)):
            ind = _indicador_clausula(texto, ini, fin, cifras, pais, anio)
            h = _hecho_reciente(ind, pais, anio, "brecha_region") if ind else None
            if h is not None and abs(h.valor) >= 0.05:
                dice = "arriba" if _ARRIBA_REG.search(clausula) else "abajo"
                real = "arriba" if h.valor > 0 else "abajo"
                salida.append({"clausula": clausula, "tipo": "frente_a_region", "indicador": ind, "dice": dice,
                               "real": real, "brecha": h.valor, "correcta": dice == real})
                continue
        sube, baja = _SUBE.search(clausula), _BAJA.search(clausula)
        if bool(sube) != bool(baja):
            ind = ind or _indicador_clausula(texto, ini, fin, cifras, pais, anio)
            if not ind or ind == "crecimiento":
                continue
            h = _hecho_reciente(ind, pais, anio, "variacion")
            if h is None or abs(h.valor) < 0.05:
                continue
            dice = "sube" if sube else "baja"
            if _DEFICIT.search(clausula) and h.valor <= 0 or _DEFICIT.search(clausula) and h.valor > 0:
                # «el déficit se amplió» = el saldo se hizo más negativo
                dice = {"sube": "baja", "baja": "sube"}[dice] if capa_kpis.buscar_hecho(
                    ind, pais, h.anio, "valor") and capa_kpis.buscar_hecho(ind, pais, h.anio, "valor").valor < 0 \
                    else dice
            real = "sube" if h.valor > 0 else "baja"
            salida.append({"clausula": clausula, "tipo": "frente_al_anio_anterior", "indicador": ind, "dice": dice,
                           "real": real, "variacion": h.valor, "correcta": dice == real})
    return salida


def revisar_resultado(r: dict) -> list[dict]:
    """Aplica `revisar` a cada afirmación (informes estructurados) o línea (texto libre) de un resultado."""
    if not r.get("verificacion"):
        return []
    out = []
    for s in r["verificacion"]["secciones"]:
        if s["seccion"].startswith("recomendacion"):
            continue
        for c in revisar(s["texto"], r["pais"], r["anio"]):
            out.append({"seccion": s["seccion"], **c})
    return out
