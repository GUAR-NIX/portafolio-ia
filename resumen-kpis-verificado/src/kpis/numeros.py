"""Extractor de cifras en texto español de negocios.

Casos que resuelve: coma decimal («3,5 %») y punto decimal («3.5%»), separador de miles («6.718 USD»,
«121.000 millones», «6,718.14»), signo con guion o menos tipográfico («−1,9 %»), escalas («mil millones»,
«millones», «miles», «billones», «MM», «bn»), unidades (%, puntos porcentuales, USD/US$/dólares,
personas/habitantes, puesto/lugar) y ordinales («tercer lugar»). Ignora años sueltos («en 2024»).

Cada cifra se devuelve con una lista de *candidatos*: cuando el texto es ambiguo («6,718» puede ser
6,718 o 6.718; «billones» puede ser 10^12 o el anglicismo 10^9) el verificador prueba todos.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_NUM = r"(?<![\w.,\-−])(?P<signo>[-−–](?=\d)|\+(?=\d))?(?P<num>\d{1,3}(?:[.,  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\d])"
_RE_NUM = re.compile(_NUM)

# Escalas: (patrón, factores candidatos). El orden importa: lo más largo primero.
_ESCALAS = [
    (r"miles de millones|mil millones|mm de usd|bn\b|b\b(?=\s*(?:de\s*)?(?:usd|d[oó]lares))", [1e9]),
    (r"billones", [1e12, 1e9]),
    (r"millones|mill\.|mn\b|m\b(?=\s*(?:de\s*)?(?:usd|d[oó]lares|personas|habitantes))|mm\b", [1e6]),
    (r"miles|mil\b", [1e3]),
]
_UNIDADES = [
    (r"puntos porcentuales|p\.\s?p\.|pp\b|puntos\b", "pp"),
    (r"%|por ciento|porciento", "%"),
    (r"(?:de\s+)?(?:usd|us\$|d[oó]lares)", "USD"),
    (r"(?:de\s+)?(?:personas|habitantes)", "personas"),
]
_ANTES_USD = re.compile(r"(?:usd|us\$|\$)\s*$", re.I)
_PUESTO_ANTES = re.compile(r"(?:puesto|lugar|posici[oó]n|ranking|n\.?º|n[uú]mero)\s*(?:n\.?º\s*)?$", re.I)
# Tras NFKC, «18ª» y «4º» quedan como «18a» y «4o»
_PUESTO_DESPUES = re.compile(r"^\s*(?:º|°|ª|\.?er|\.?[oa])?\s*(?:lugar|puesto|posici[oó]n|(?:pa[ií]s\s+)?(?:con\s+)?"
                             r"(?:mayor|menor|m[aá]s\s|menos\s))", re.I)
_ORDINALES = {"primer": 1, "segund": 2, "tercer": 3, "cuart": 4, "quint": 5, "sext": 6, "s[eé]ptim": 7,
              "octav": 8, "noven": 9, "d[eé]cim": 10}
_RE_ORDINAL = re.compile(r"\b(?P<ord>" + "|".join(_ORDINALES) + r")(?:o|a|os|as)?\s+(?:lugar|puesto|posici[oó]n|"
                         r"(?:pa[ií]s\s+)?(?:con\s+)?(?:mayor|menor|m[aá]s|menos))", re.I)


@dataclass
class Cifra:
    texto: str
    inicio: int
    fin: int
    candidatos: list[float]          # valores en unidad base (USD, personas, %, pp, puesto)
    unidad: str | None               # "%", "pp", "USD", "personas", "puesto" o None
    decimales: int                   # decimales escritos (para la tolerancia de redondeo)
    escala: float = 1.0              # factor aplicado (1e9 en «mil millones»)
    signo_explicito: bool = False
    marcas: list[str] = field(default_factory=list)
    # Decimales de cada candidato: en «5,503» la lectura 5,503 tiene 3 y la lectura 5.503 tiene 0
    decimales_cand: list[int] = field(default_factory=list)
    escalas_cand: list[float] = field(default_factory=list)

    @property
    def valor(self) -> float:
        return self.candidatos[0]


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)
    return texto.replace("−", "-").replace("–", "–").replace(" ", " ").replace(" ", " ")


def interpretar_numero(s: str) -> tuple[list[float], int]:
    """'1.234,5' -> ([1234.5], 1); '6,718' -> ([6.718, 6718.0], 3); '3.5' -> ([3.5], 1)."""
    s = s.replace(" ", "").replace(" ", "")
    tiene_p, tiene_c = "." in s, "," in s
    if tiene_p and tiene_c:
        dec_sep = "," if s.rfind(",") > s.rfind(".") else "."
        miles_sep = "." if dec_sep == "," else ","
        ent, _, dec = s.rpartition(dec_sep)
        return [float(ent.replace(miles_sep, "") + "." + dec)], len(dec)
    sep = "," if tiene_c else "." if tiene_p else None
    if sep is None:
        return [float(s)], 0
    partes = s.split(sep)
    if len(partes) > 2:  # «1.234.567»: separador de miles
        return [float("".join(partes))], 0
    ent, dec = partes
    if len(dec) == 3 and ent != "0":
        # Ambiguo: en español «6.718» son miles y «6,718» decimal; en inglés al revés.
        miles, decimal = float(ent + dec), float(ent + "." + dec)
        return ([miles, decimal], 0) if sep == "." else ([decimal, miles], 3)
    return [float(ent + "." + dec)], len(dec)


def _es_anio(valor: float, decimales: int, unidad: str | None, escala: float, contexto_prev: str) -> bool:
    return (decimales == 0 and unidad is None and escala == 1.0 and 1990 <= valor <= 2035 and valor == int(valor)
            and not _ANTES_USD.search(contexto_prev))


def extraer(texto: str) -> list[Cifra]:
    t = normalizar(texto)
    low = t.lower()
    cifras: list[Cifra] = []
    for m in _RE_NUM.finditer(t):
        bruto = m.group("num")
        candidatos, dec = interpretar_numero(bruto)
        prev = low[max(0, m.start() - 25):m.start()]
        resto = low[m.end():m.end() + 40]
        # Marcador de lista «1.» al inicio de línea o rango «2,1–3,5» (el segundo no lleva signo)
        if re.match(r"^\s*[.)]\s", resto) and (m.start() == 0 or t[m.start() - 1] == "\n"):
            continue
        escala, factores, unidad, marcas = 1.0, [1.0], None, []
        cursor = 0
        r = resto
        for patron, fs in _ESCALAS:
            mm = re.match(r"\s*(?:" + patron + r")", r)
            if mm:
                factores, escala = fs, fs[0]
                cursor = mm.end()
                if len(fs) > 1:
                    marcas.append("escala_ambigua")
                break
        for patron, u in _UNIDADES:
            mm = re.match(r"\s*(?:" + patron + r")", r[cursor:])
            if mm:
                unidad = u
                break
        if unidad is None and _ANTES_USD.search(prev):
            unidad = "USD"
        if unidad is None and (_PUESTO_ANTES.search(prev) or _PUESTO_DESPUES.match(resto)):
            unidad = "puesto"
        if unidad is None and escala > 1 and re.match(r"\s*(?:de\s+)?(?:habitantes|personas)", r[cursor:]):
            unidad = "personas"
        # «de 18» tras un puesto es el total de países, no una afirmación
        if re.search(r"(?:puesto|lugar|posici[oó]n)\s*\d{1,2}\s*de\s*$", prev):
            continue
        if _es_anio(candidatos[0], dec, unidad, escala, prev):
            continue
        signo = m.group("signo")
        if signo == "–" and m.start() > 0 and t[m.start() - 1].isdigit():
            signo = None  # rango
        neg = signo in ("-", "−", "–")
        valores, decs, escalas = [], [], []
        for k, c in enumerate(candidatos):
            # la lectura alternativa de «6,718» / «6.718» es entera o tiene 3 decimales
            d = dec if k == 0 else (0 if c == int(c) else 3)
            for f in factores:
                valores.append((-c if neg else c) * f)
                decs.append(d)
                escalas.append(f)
        cifras.append(Cifra(texto=t[m.start():m.end() + cursor].strip() if cursor else bruto, inicio=m.start(),
                            fin=m.end(), candidatos=valores, unidad=unidad, decimales=dec, escala=escala,
                            signo_explicito=signo is not None, marcas=marcas, decimales_cand=decs,
                            escalas_cand=escalas))
    for m in _RE_ORDINAL.finditer(t):
        clave = next(k for k in _ORDINALES if re.fullmatch(k + r"(?:o|a|os|as)?", m.group("ord").lower())
                     or re.match(k, m.group("ord").lower()))
        cifras.append(Cifra(texto=m.group(0), inicio=m.start(), fin=m.end(), candidatos=[float(_ORDINALES[clave])],
                            unidad="puesto", decimales=0, marcas=["ordinal"]))
    return sorted(cifras, key=lambda c: c.inicio)


def tolerancia(c: Cifra, k: int = 0) -> float:
    """Media unidad del último dígito escrito, en unidad base (redondeo correcto), para el candidato k."""
    dec = c.decimales_cand[k] if k < len(c.decimales_cand) else c.decimales
    esc = c.escalas_cand[k] if k < len(c.escalas_cand) else c.escala
    return 0.5 * 10 ** (-dec) * esc * (1 + 1e-9) + 1e-9
