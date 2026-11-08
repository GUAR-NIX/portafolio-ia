"""Normalización y tokenización en español para BM25, y expansión de consultas con sinónimos coloquiales → término legal."""
from __future__ import annotations

import re
import unicodedata

PALABRAS_VACIAS = set("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquellos aqui asi aun cada como con
contra cual cuales cuando cuanto cuantos cuanta cuantas de del desde donde dos e el ella ellas ellos en entre era es esa
esas ese eso esos esta estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mis mucho muy
no nos o os otra otras otro otros para pero poco por porque que quien quienes se sea sean segun ser si sin sino sobre
son su sus tal tambien tan tanto te tiene tienen todo todos tu tus un una unas uno unos y ya yo puedo puede pueden debo
debe deben mio mia cuanto cuantas cuantos tengo tiene hacer hace
""".split())


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def raiz(p: str) -> str:
    """Raíz muy ligera (plurales y algunas terminaciones): «remuneraciones» y «remuneración» → «remuneracion»."""
    if len(p) <= 4 or p.isdigit():
        return p
    for suf in ("ciones", "siones", "mente", "ces", "es", "s"):
        if p.endswith(suf) and len(p) - len(suf) >= 4:
            base = p[: -len(suf)]
            if suf in ("ciones", "siones"):
                base += "cion" if suf == "ciones" else "sion"
            elif suf == "ces":
                base += "z"
            return base
    return p


def tokens(s: str) -> list[str]:
    s = sin_tildes(s.lower())
    return [raiz(t) for t in re.findall(r"[a-zñ]+|\d+", s) if t not in PALABRAS_VACIAS and len(t) > 1]


# Cómo pregunta la gente → cómo lo dice la ley. Se agregan a la consulta (no la reemplazan).
# v1: escrita ANTES de ver resultados, con frases coloquiales típicas. v2 (abajo): v1 corregida tras el análisis de
# errores del conjunto principal (docs/analisis_errores.md). Las preguntas «nuevas» miden si la v2 se sobreajustó.
SINONIMOS_V1: list[tuple[str, str]] = [
    (r"d[eé]cimo ?tercer[oa]?( sueldo)?|d[eé]cimo 13|aguinaldo|bono navide[nñ]o", "decimotercera remuneración"),
    (r"d[eé]cimo ?cuart[oa]?( sueldo)?|d[eé]cimo 14|bono escolar", "decimocuarta remuneración"),
    (r"horas? extras?|sobretiempo", "horas suplementarias y extraordinarias recargo"),
    (r"me (botaron|despidieron|echaron)|botar|bot[oó]|despid(o|ieron|ir)( sin)?|echar del trabajo",
     "despido intempestivo indemnización"),
    (r"liquidaci[oó]n|finiquito", "liquidación indemnización bonificación desahucio acta de finiquito"),
    (r"renunci(a|ar|o)", "desahucio aviso terminación voluntaria"),
    (r"sueldo|salario mensual|pago mensual", "remuneración sueldo salario"),
    (r"jefe|patr[oó]n|due[nñ]o de la empresa|la empresa", "empleador"),
    (r"embarazada|embarazo|gestaci[oó]n", "mujer embarazada estado de gestación maternidad"),
    (r"(licencia|permiso) (de|por) paternidad|pap[aá] (reci[eé]n|nuevo)|nacimiento de (mi|un) hij", "licencia paternidad padre"),
    (r"utilidades|reparto de ganancias", "participación de trabajadores en las utilidades quince por ciento"),
    (r"prueba|contrato a prueba|periodo de prueba", "período de prueba"),
    (r"menor(es)? de edad|adolescentes?|ni[nñ]os?", "adolescentes menores trabajo infantil"),
    (r"jubilaci[oó]n patronal|jubilarme con la empresa", "jubilación patronal veinticinco años"),
    (r"trabajadora? (del hogar|dom[eé]stic[oa])|empleada dom[eé]stica", "trabajo doméstico servicio doméstico"),
    (r"sindicato", "asociación de trabajadores sindicato organización"),
    (r"huelga|paro", "huelga"),
    (r"vacaciones", "vacaciones anuales descanso"),
    (r"fines? de semana|s[aá]bados? y domingos?|feriados?", "descanso obligatorio días de descanso"),
    (r"lactancia|dar de lactar|amamantar", "lactancia jornada de seis horas"),
    (r"teletrabajo|trabajo (desde casa|remoto)|home office", "teletrabajo"),
    (r"multa|sanci[oó]n al trabajador", "sanciones multas reglamento interno"),
    (r"accidente(s)? (de|en el) trabajo|me lastim[eé] trabajando", "accidente de trabajo riesgos indemnización"),
    (r"enfermedad (profesional|laboral)|me enferm[eé] por el trabajo", "enfermedad profesional riesgos"),
    (r"prescrib|cu[aá]nto tiempo tengo para (reclamar|demandar)|plazo para (reclamar|demandar)",
     "prescripción de las acciones"),
    (r"visto bueno", "visto bueno inspector del trabajo"),
    (r"discapacidad", "personas con discapacidad"),
    (r"comisariato|comedor|alimentaci[oó]n", "comedor alimentación"),
    (r"contrato (fijo|a plazo|temporal|eventual|por obra)", "contrato eventual ocasional por obra tiempo fijo"),
    (r"pasant[ií]a|pasante", "pasantías"),
]


def ventana_relevante(texto: str, consulta: str, max_chars: int) -> str:
    """Artículos muy largos (el 42 tiene 35 numerales y 15 000 caracteres): se conserva el primer párrafo y los
    numerales que comparten más palabras con la consulta, en su orden original, hasta `max_chars`. La cita sigue
    siendo el artículo completo; solo se recorta lo que lee el re-ranker o el LLM."""
    if len(texto) <= max_chars:
        return texto
    partes = texto.split("\n")
    q = set(tokens(consulta))
    puntaje = [len(q & set(tokens(p))) / (1 + len(p) / 400) for p in partes]
    elegidas, usado = {0}, len(partes[0])
    for i in sorted(range(1, len(partes)), key=lambda i: -puntaje[i]):
        if puntaje[i] == 0 or usado + len(partes[i]) > max_chars:
            continue
        elegidas.add(i)
        usado += len(partes[i]) + 1
    salida, previo = [], -1
    for i in sorted(elegidas):
        if i != previo + 1:
            salida.append("[...]")
        salida.append(partes[i][:max_chars])
        previo = i
    return "\n".join(salida)


def _v2() -> list[tuple[str, str]]:
    """v1 con las correcciones del análisis de errores (cada cambio cita la pregunta que lo motivó)."""
    cambios = {
        # P023/P024: el epígrafe del art. 111 dice «décima tercera remuneración o bono navideño»
        r"d[eé]cimo ?tercer[oa]?( sueldo)?|d[eé]cimo 13|aguinaldo|bono navide[nñ]o":
            "decimotercera remuneración décima tercera bono navideño",
        r"d[eé]cimo ?cuart[oa]?( sueldo)?|d[eé]cimo 14|bono escolar": "decimocuarta remuneración décima cuarta",
        # P049: «renunciar a mis derechos» no es renunciar al trabajo → se excluye «renunciar a (mis|los) derechos»
        r"renunci(a|ar|o)": None,
        # P089, P091, P051, P031/P032: términos de más que diluían BM25
        r"accidente(s)? (de|en el) trabajo|me lastim[eé] trabajando": "accidente de trabajo",
        r"sindicato": "sindicato asociación de trabajadores",
        r"multa|sanci[oó]n al trabajador": "multas",
        r"utilidades|reparto de ganancias": "participación utilidades",
    }
    salida = []
    for patron, legal in SINONIMOS_V1:
        if patron in cambios:
            if cambios[patron] is None:
                continue
            legal = cambios[patron]
        salida.append((patron, legal))
    salida.append((r"renunci(o|ar|é|e)\b(?! a (mis|los|sus) derechos)|mi renuncia|voy a renunciar",
                   "desahucio aviso terminación voluntaria"))
    # P088: «jubilado patronal» no activaba el patrón de «jubilación patronal»
    salida.append((r"jubilad[oa] patronal|pensi[oó]n jubilar", "jubilación patronal pensión jubilar"))
    return salida


SINONIMOS: list[tuple[str, str]] = _v2()


def expandir(consulta: str, version: str = "v2") -> str:
    """Devuelve la consulta + los términos legales equivalentes que detecta (el original no se pierde)."""
    extra = []
    baja = consulta.lower()
    for patron, legal in (SINONIMOS_V1 if version == "v1" else SINONIMOS):
        if re.search(patron, baja):
            extra.append(legal)
    return consulta if not extra else f"{consulta} ({'; '.join(extra)})"
