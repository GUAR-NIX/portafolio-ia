"""Verificador automático de cifras.

Para cada número del texto decide si es **correcta** (coincide con la capa de KPIs dentro de la
tolerancia de redondeo y con una unidad compatible), **incorrecta** (el texto nombra un indicador
con dato, pero el número no coincide con ninguna de sus cifras) o **sin fuente** (no se puede
atribuir a ninguna cifra de la capa). Si la afirmación trae referencias estructuradas, se usan
primero; si no, el indicador, el país y el año se deducen del contexto de la oración.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import capa_kpis, config
from .capa_kpis import Hecho
from .esquema import Informe, Referencia
from .numeros import Cifra, extraer, normalizar, tolerancia

# ---------------------------------------------------------------- diccionarios de contexto
_PALABRAS_IND: list[tuple[str, tuple[str, ...]]] = []
for _ind in config.INDICADORES:
    for _p in _ind.palabras:
        _PALABRAS_IND.append((_p, (_ind.clave,)))
# «PIB» a secas es ambiguo: tamaño de la economía o su crecimiento
_PALABRAS_IND += [("pib", ("pib_usd", "crecimiento", "pib_pc")), ("economía", ("crecimiento", "pib_usd")),
                  ("actividad económica", ("crecimiento",))]
_PALABRAS_IND.sort(key=lambda x: -len(x[0]))

_REGION = re.compile(r"regi[oó]n|regional|latinoam[eé]rica|am[eé]rica latina|mediana|promedio de la regi|vecinos",
                     re.I)
_PAISES_NOMBRE = {normalizar(n).lower(): iso for iso, n in config.PAISES.items()}
_PAISES_NOMBRE.update({"peru": "PER", "mexico": "MEX", "panama": "PAN", "rep. dominicana": "DOM",
                       "dominicana": "DOM", "brazil": "BRA"})
_ASC = re.compile(r"menor|m[aá]s baj|menos|m[ií]nim|[uú]ltim", re.I)
_RE_ANIO = re.compile(r"\b(19[9]\d|20[0-3]\d)\b")
_ESCASOS = ("pobreza", "gini", "deuda_externa")

ESTADOS = ("correcta", "incorrecta", "sin_fuente")

# Cifras que forman parte de la definición de un indicador, no son afirmaciones: la línea de pobreza
# («USD 3,00 al día», «3 dólares diarios»), el año base de la PPA y la escala del Gini («0–100»).
_DEFINICIONALES = re.compile(
    r"(?:usd|us\$|\$)\s?\d+(?:[.,]\d+)?\s*(?:al|por)\s*d[ií]a|\d+(?:[.,]\d+)?\s*(?:usd|d[oó]lares)\s*(?:al|por)\s*d[ií]a"
    r"|\d+(?:[.,]\d+)?\s*d[oó]lares diarios|ppa\s*(?:de\s*)?\d{4}|\(?\b0\s*[–-]\s*100\b\)?", re.I)


# Referencias técnicas copiadas dentro del texto, p. ej. «(pib_usd,ARG,2023,valor,649500000000)».
# No son cifras redactadas para el lector: se excluyen de la verificación y se cuentan aparte
# (afean el informe y la rúbrica de legibilidad las penaliza).
_REF_EN_LINEA = re.compile(r"\(\s*[a-z_]+\s*,\s*(?:[A-Z]{3}|LAT)\s*,[^()]*\)")


def enmascarar_definicionales(texto: str) -> str:
    """Reemplaza por «#» los dígitos de las cifras definicionales y de las referencias en línea
    (mantiene las posiciones)."""
    texto = _DEFINICIONALES.sub(lambda m: re.sub(r"\d", "#", m.group(0)), texto)
    return _REF_EN_LINEA.sub(lambda m: re.sub(r"\d", "#", m.group(0)), texto)


def contar_referencias_en_linea(texto: str) -> int:
    return len(_REF_EN_LINEA.findall(texto))


@dataclass
class ResultadoCifra:
    texto: str
    inicio: int
    fin: int
    estado: str
    valor_texto: float
    unidad: str | None
    hecho_id: str | None = None
    valor_real: float | None = None
    real_formateado: str | None = None
    motivo: str = ""
    marcas: list[str] = field(default_factory=list)

    def dict(self) -> dict:
        return asdict(self)


def _oracion(texto: str, pos: int) -> tuple[int, int]:
    ini = 0
    for m in re.finditer(r"(?<=[.!?;])\s+|\n", texto):
        if m.end() <= pos:
            ini = m.end()
        elif m.start() >= pos:
            return ini, m.start()
    return ini, len(texto)


def _indicadores_contexto(oracion: str, pos: int) -> list[tuple[str, ...]]:
    """Indicadores mencionados, ordenados por cercanía (prefiere los que van antes del número)."""
    low = oracion.lower()
    ocupado = [False] * len(low)
    hallados: list[tuple[float, tuple[str, ...]]] = []
    for palabra, inds in _PALABRAS_IND:
        for m in re.finditer(r"(?<!\w)" + re.escape(palabra), low):
            if any(ocupado[m.start():m.end()]):
                continue
            if palabra == "pib" and low[max(0, m.start() - 4):m.start()] == "del ":
                continue  # «% del PIB» es la unidad, no el indicador
            for k in range(m.start(), m.end()):
                ocupado[k] = True
            dist = pos - m.end() if m.end() <= pos else (m.start() - pos) * 2 + 10
            hallados.append((dist, inds))
    hallados.sort(key=lambda x: x[0])
    vistos, salida = set(), []
    for _, inds in hallados:
        if inds not in vistos:
            vistos.add(inds)
            salida.append(inds)
    return salida


def _paises_contexto(oracion: str, pais: str) -> list[str]:
    low = normalizar(oracion).lower()
    paises = [iso for nombre, iso in _PAISES_NOMBRE.items() if re.search(r"(?<!\w)" + re.escape(nombre), low)]
    salida = [pais] + [p for p in paises if p != pais]
    if _REGION.search(low):
        salida.append("LAT")
    return salida


def _anios_contexto(oracion: str, anio: int) -> list[int]:
    """Años nombrados en la oración más el del informe; si no nombra ninguno, el del informe y el anterior."""
    mencionados = {int(a) for a in _RE_ANIO.findall(oracion)}
    return sorted(mencionados | {anio}, reverse=True) if mencionados else [anio, anio - 1]


def _unidad_compatible(c: Cifra, h: Hecho) -> tuple[bool, str | None]:
    """(compatible, marca). «%» con un hecho en pp se acepta pero se marca."""
    u = c.unidad
    if h.unidad == "puesto":
        return u == "puesto", None
    if u == "puesto":
        return False, None
    if u is None:
        if c.escala > 1:
            return h.unidad in ("USD", "personas"), None
        return True, None
    if u == h.unidad:
        return True, None
    if u == "%" and h.unidad == "pp":
        return True, "pp_escrito_como_%"
    if u == "pp" and h.unidad == "índice":
        return True, None
    if u == "%" and h.unidad == "índice":
        return True, "indice_escrito_como_%"
    return False, None


def _coincide(c: Cifra, h: Hecho, asc: bool) -> bool:
    reales = [h.valor]
    if h.metrica == "ranking" and asc and h.n:
        reales.append(h.n + 1 - h.valor)
    for k, v in enumerate(c.candidatos):
        tol = tolerancia(c, k)
        for r in reales:
            if abs(v - r) <= tol or (not c.signo_explicito and abs(abs(v) - abs(r)) <= tol):
                return True
    return False


def _candidatos(inds: list[str], paises: list[str], anios: list[int]) -> list[Hecho]:
    idx = capa_kpis.indice_hechos()
    out = []
    for i in inds:
        for p in paises:
            ans = set(anios)
            if i in _ESCASOS:
                ans |= {a - k for a in anios for k in range(1, capa_kpis.VENTANA_DATO_ESCASO + 1)}
            for a in sorted(ans, reverse=True):
                out.extend(idx.get((i, p, a), ()))
    return out


def _referencia_a_hecho(r: Referencia) -> Hecho | None:
    # La mediana regional solo existe para LAT; si el modelo pone el país, la métrica ya es inequívoca
    pais = "LAT" if r.metrica == "mediana_region" else r.pais.upper()
    return capa_kpis.buscar_hecho(r.indicador, pais, int(r.anio), r.metrica)


def _cercano(c: Cifra, hs: list[Hecho]) -> Hecho:
    orden = {"valor": 0, "variacion": 1, "mediana_region": 2, "brecha_region": 3, "ranking": 4}
    return min(hs, key=lambda h: (orden[h.metrica], min(abs(v - h.valor) for v in c.candidatos)))


def verificar_texto(texto: str, pais: str, anio: int,
                    referencias: list[Referencia] | None = None) -> list[ResultadoCifra]:
    texto = normalizar(texto)
    resultados = []
    hechos_ref = [h for h in (_referencia_a_hecho(r) for r in (referencias or [])) if h is not None]
    for c in extraer(enmascarar_definicionales(texto)):
        i0, i1 = _oracion(texto, c.inicio)
        oracion = texto[i0:i1]
        asc = bool(_ASC.search(oracion))
        base = dict(texto=c.texto, inicio=c.inicio, fin=c.fin, valor_texto=c.valor, unidad=c.unidad,
                    marcas=list(c.marcas))

        def ok(h: Hecho, motivo: str, extra: list[str] | None = None) -> ResultadoCifra:
            _, marca = _unidad_compatible(c, h)
            marcas = base["marcas"] + ([marca] if marca else []) + (extra or [])
            return ResultadoCifra(**{**base, "marcas": marcas}, estado="correcta", hecho_id=h.id,
                                  valor_real=h.valor, real_formateado=capa_kpis.fmt_hecho(h), motivo=motivo)

        # 1) referencias estructuradas
        h_ok = next((h for h in hechos_ref if _unidad_compatible(c, h)[0] and _coincide(c, h, asc)), None)
        if h_ok:
            resultados.append(ok(h_ok, "coincide con su referencia"))
            continue
        # 2) contexto de la oración
        grupos = _indicadores_contexto(oracion, c.inicio - i0)
        inds_ref = list(dict.fromkeys(h.indicador for h in hechos_ref))
        principal = list(dict.fromkeys((list(grupos[0]) if grupos else []) + inds_ref))
        otros = [i for g in grupos[1:] for i in g if i not in principal]
        paises = _paises_contexto(oracion, pais)
        anios = _anios_contexto(oracion, anio)
        compat = lambda hs: [h for h in hs if _unidad_compatible(c, h)[0]]  # noqa: E731
        cand = compat(_candidatos(principal, paises, anios))
        h_ok = next((h for h in cand if _coincide(c, h, asc)), None)
        if h_ok:
            resultados.append(ok(h_ok, "coincide con el indicador del contexto"))
            continue
        cand_otros = compat(_candidatos(otros, paises, anios))
        h_ok = next((h for h in cand_otros if _coincide(c, h, asc)), None)
        if h_ok:
            resultados.append(ok(h_ok, "coincide con otro indicador de la oración", ["indicador_secundario"]))
            continue
        if cand:
            h = _cercano(c, cand)
            resultados.append(ResultadoCifra(**base, estado="incorrecta", hecho_id=h.id, valor_real=h.valor,
                                             real_formateado=capa_kpis.fmt_hecho(h),
                                             motivo=f"no coincide con {h.indicador} ({h.pais}, {h.anio})"))
            continue
        # 3) sin indicador reconocible: búsqueda amplia, solo si la cifra es lo bastante precisa
        if not principal:
            todos = compat(_candidatos([i.clave for i in config.INDICADORES], paises, anios))
            precisa = c.decimales >= 1 or c.escala > 1 or c.unidad in ("USD", "personas") or abs(c.valor) >= 100
            h_ok = next((h for h in todos if _coincide(c, h, asc)), None) if precisa else None
            if h_ok:
                resultados.append(ok(h_ok, "coincide sin indicador explícito", ["atribucion_debil"]))
                continue
        resultados.append(ResultadoCifra(**base, estado="sin_fuente",
                                         motivo="no se encontró ninguna cifra de la capa de KPIs que la respalde"))
    return resultados


def verificar_referencias(referencias: list[Referencia]) -> list[dict]:
    """Comprueba que cada referencia exista en la capa y que su valor coincida (tolerancia de redondeo)."""
    out = []
    for r in referencias:
        h = _referencia_a_hecho(r)
        if h is None:
            out.append({"referencia": r.model_dump(), "estado": "inexistente", "valor_real": None})
            continue
        dec = len(repr(float(r.valor)).split(".")[1].rstrip("0")) if "." in repr(float(r.valor)) else 0
        tol = max(0.5 * 10 ** (-min(dec, 2)), 0.006 * abs(h.valor)) if h.unidad not in ("USD", "personas") \
            else 0.006 * abs(h.valor)
        # Para USD/población el modelo puede escribir la referencia en la escala de la tabla (mil millones, millones)
        escalas = (1, 1e9, 1e6) if h.unidad in ("USD", "personas") else (1,)
        valido = any(abs(r.valor * e - h.valor) <= tol or abs(abs(r.valor * e) - abs(h.valor)) <= tol
                     for e in escalas)
        out.append({"referencia": r.model_dump(), "estado": "correcta" if valido else "incorrecta",
                    "valor_real": h.valor, "hecho_id": h.id})
    return out


def verificar_informe(informe: Informe, pais: str, anio: int) -> dict:
    """Verifica todas las afirmaciones de un informe estructurado."""
    secciones = []
    for nombre, af in informe.afirmaciones():
        cifras = verificar_texto(af.texto, pais, anio, af.referencias)
        refs = verificar_referencias(af.referencias)
        secciones.append({"seccion": nombre, "texto": af.texto, "cifras": [c.dict() for c in cifras],
                          "referencias": refs})
    return resumir(secciones)


def verificar_markdown(texto: str, pais: str, anio: int) -> dict:
    """Verifica un informe en texto libre, línea por línea."""
    secciones = []
    for k, linea in enumerate(l for l in texto.splitlines() if l.strip()):
        cifras = verificar_texto(linea, pais, anio)
        secciones.append({"seccion": f"linea_{k + 1}", "texto": linea, "cifras": [c.dict() for c in cifras],
                          "referencias": []})
    return resumir(secciones)


def resumir(secciones: list[dict]) -> dict:
    cifras = [c for s in secciones for c in s["cifras"]]
    refs = [r for s in secciones for r in s["referencias"]]
    conteo = {e: sum(c["estado"] == e for c in cifras) for e in ESTADOS}
    n = len(cifras)
    return {"secciones": secciones, "n_cifras": n, **{f"n_{e}": conteo[e] for e in ESTADOS},
            "tasa_alucinacion": (conteo["incorrecta"] + conteo["sin_fuente"]) / n if n else 0.0,
            "n_referencias": len(refs), "n_referencias_ok": sum(r["estado"] == "correcta" for r in refs),
            "n_marcas_unidad": sum(any(m.endswith("_como_%") for m in c["marcas"]) for c in cifras),
            "n_atribucion_debil": sum("atribucion_debil" in c["marcas"] for c in cifras),
            "n_refs_en_linea": sum(contar_referencias_en_linea(s["texto"]) for s in secciones),
            "aprobado": n > 0 and conteo["incorrecta"] + conteo["sin_fuente"] == 0}
