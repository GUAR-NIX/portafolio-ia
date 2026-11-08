"""Los cuatro enfoques comparados:

(a) libre        LLM con la tabla en el prompt, informe en Markdown sin estructura.
(b) estructurado LLM con salida estructurada (pydantic → JSON schema) y una referencia por cifra.
(c) verificado   (b) + verificador automático; si alguna cifra falla, se devuelve al LLM la lista de
                 errores con el valor real y se reintenta (hasta MAX_REINTENTOS).
(d) plantilla    texto determinista armado desde la capa de KPIs, sin LLM.
"""
from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from . import capa_kpis, config, llm, verificador
from .capa_kpis import fmt_hecho
from .esquema import Afirmacion, Informe, Referencia, esquema_json

ENFOQUES = ("libre", "estructurado", "verificado", "plantilla")
NOMBRES = {"libre": "(a) LLM libre", "estructurado": "(b) LLM estructurado", "verificado": "(c) (b) + verificación y reintento",
           "plantilla": "(d) Plantilla determinista"}
VERSIONES_PROMPT = ("v1", "v2")

SISTEMA = ("Eres analista económico senior de un banco de desarrollo. Redactas resúmenes ejecutivos breves "
           "para gerencia, en español neutro, a partir de una tabla de KPIs del Banco Mundial.")

REGLAS_V2 = """Reglas para las cifras:
1. Copia cada cifra exactamente como aparece en la tabla, con el mismo redondeo y la misma unidad (%, pp, USD, mil millones de USD, millones de personas).
2. No calcules cifras nuevas (diferencias, sumas, promedios, tasas) ni uses cifras que recuerdes: si no está en la tabla, no la escribas.
3. Las variaciones de indicadores en % se expresan en puntos porcentuales (pp): «subió 2,0 pp», no «subió 2,0 %».
4. El puesto 1 es el valor más alto de la región. Si dices «el más bajo» o «el menor», comprueba el puesto y el número de países.
5. No inventes proyecciones ni metas numéricas en riesgos o recomendaciones.
6. Prioriza lo que la tabla marca como anomalía, el crecimiento, la inflación, el desempleo y lo que más se aleja de la mediana regional."""

REGLA_JSON_V2 = """7. En las referencias, «valor» es un número JSON con punto decimal, sin separador de miles y en la escala de la tabla: 3,2 % → 3.2; -2,2 pp → -2.2; 120,8 mil millones de USD → 120.8; 6.718 USD → 6718; puesto 14 de 18 → 14. Para la mediana regional usa pais = "LAT"."""


def _pedido(pais: str, anio: int) -> str:
    return (f"País: {config.PAISES[pais]} ({pais}). Año del informe: {anio}. Región de comparación: "
            f"18 países de América Latina (mediana).\n\nTabla de KPIs:\n{capa_kpis.tabla_para_prompt(pais, anio)}")


def mensajes_libre(pais: str, anio: int, version: str) -> list[dict]:
    instr = ("\n\nEscribe un resumen ejecutivo en Markdown con: un titular (una línea que empiece con «## »), "
             "una sección «Hallazgos» con 3 a 4 viñetas, una sección «Riesgos» con 1 a 2 viñetas y una sección "
             "«Recomendaciones» con 1 a 2 viñetas. Máximo 250 palabras. Usa solo cifras de la tabla.")
    if version == "v2":
        instr += "\n\n" + REGLAS_V2
    return [{"role": "system", "content": SISTEMA}, {"role": "user", "content": _pedido(pais, anio) + instr}]


def mensajes_estructurado(pais: str, anio: int, version: str) -> list[dict]:
    instr = ("\n\nDevuelve un objeto JSON con: titular, 3 a 4 hallazgos, 1 a 2 riesgos y 1 a 2 recomendaciones. "
             "Cada elemento tiene «texto» (una o dos oraciones) y «referencias»: una referencia por cada cifra que "
             f"escribas en el texto, con indicador (la clave entre paréntesis de la tabla), pais ({pais}, o LAT para "
             f"la mediana regional), anio, metrica (valor, variacion, mediana_region, brecha_region o ranking) y "
             f"valor (el número de la tabla sin unidades; para ranking, el puesto). Usa solo cifras de la tabla. "
             "Escribe el JSON compacto, en una sola línea y sin sangría.")
    if version == "v2":
        instr += "\n\n" + REGLAS_V2 + "\n" + REGLA_JSON_V2
    return [{"role": "system", "content": SISTEMA}, {"role": "user", "content": _pedido(pais, anio) + instr}]


def _parsear(contenido: str) -> tuple[Informe | None, str | None]:
    try:
        return Informe.model_validate(json.loads(contenido)), None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e)[:300]


# --------------------------------------------------------------------------------- (a)
def generar_libre(pais: str, anio: int, version: str = "v2", proveedor: str | None = None) -> dict[str, Any]:
    r = llm.chat(mensajes_libre(pais, anio, version), etiqueta=f"libre|{version}|{pais}|{anio}", proveedor=proveedor)
    t0 = time.perf_counter()
    ver = verificador.verificar_markdown(r["contenido"], pais, anio)
    return {"enfoque": "libre", "version": version, "pais": pais, "anio": anio, "texto": r["contenido"],
            "verificacion": ver, "segundos_llm": r["segundos"], "segundos_verificacion": time.perf_counter() - t0,
            "llamadas_llm": 1, "json_valido": None, "informe": None}


# --------------------------------------------------------------------------------- (b) y (c)
def _feedback(ver: dict) -> str:
    if ver["n_cifras"] == 0:
        return ("El informe no tiene ninguna cifra: un resumen de KPIs para gerencia debe citar los datos. "
                "Incluye en el titular y en cada hallazgo al menos una cifra copiada de la tabla, con su "
                "referencia. Devuelve el informe completo con el mismo esquema JSON.")
    lineas = []
    for s in ver["secciones"]:
        for c in s["cifras"]:
            if c["estado"] == "incorrecta":
                lineas.append(f"- [{s['seccion']}] escribiste «{c['texto']}»: no coincide con la tabla; "
                              f"el valor correcto es {c['real_formateado']} ({c['hecho_id'].replace('|', ', ')}).")
            elif c["estado"] == "sin_fuente":
                lineas.append(f"- [{s['seccion']}] la cifra «{c['texto']}» no está en la tabla: elimínala o "
                              "reemplázala por una cifra de la tabla.")
        for r in s["referencias"]:
            if r["estado"] == "inexistente":
                ref = r["referencia"]
                lineas.append(f"- [{s['seccion']}] la referencia ({ref['indicador']}, {ref['pais']}, {ref['anio']}, "
                              f"{ref['metrica']}) no existe en la tabla.")
    return ("Revisé cada cifra contra la tabla y hay problemas:\n" + "\n".join(lineas) +
            "\nDevuelve el informe completo corregido con el mismo esquema JSON. Mantén igual lo que estaba bien.")


def _necesita_reintento(ver: dict) -> bool:
    return (not ver["aprobado"]) or any(r["estado"] == "inexistente" for s in ver["secciones"]
                                        for r in s["referencias"])


def generar_estructurado(pais: str, anio: int, version: str = "v2", verificar_y_reintentar: bool = False,
                         proveedor: str | None = None) -> dict[str, Any]:
    msgs = mensajes_estructurado(pais, anio, version)
    enfoque = "verificado" if verificar_y_reintentar else "estructurado"
    intentos, seg_llm, seg_ver = [], 0.0, 0.0
    informe = ver = None
    contenido = ""
    for intento in range(config.MAX_REINTENTOS + 1 if verificar_y_reintentar else 1):
        r = llm.chat(msgs, formato=esquema_json(), etiqueta=f"estructurado|{version}|{pais}|{anio}|i{intento}",
                     proveedor=proveedor)
        seg_llm += r["segundos"]
        contenido = r["contenido"]
        t0 = time.perf_counter()
        nuevo, error = _parsear(contenido)
        if nuevo is not None:
            informe = nuevo
            ver = verificador.verificar_informe(informe, pais, anio)
        seg_ver += time.perf_counter() - t0
        intentos.append({"intento": intento, "json_valido": nuevo is not None, "error": error,
                         "tasa_alucinacion": None if nuevo is None else ver["tasa_alucinacion"],
                         "n_cifras": None if nuevo is None else ver["n_cifras"]})
        if not verificar_y_reintentar:
            break
        if nuevo is None:
            msgs = msgs + [{"role": "assistant", "content": contenido},
                           {"role": "user", "content": f"El JSON no es válido ({error}). Devuélvelo completo y válido."}]
            continue
        if not _necesita_reintento(ver):
            break
        msgs = msgs + [{"role": "assistant", "content": contenido}, {"role": "user", "content": _feedback(ver)}]
    return {"enfoque": enfoque, "version": version, "pais": pais, "anio": anio,
            "texto": informe.texto_markdown() if informe else contenido,
            "informe": informe.model_dump() if informe else None, "verificacion": ver,
            "segundos_llm": seg_llm, "segundos_verificacion": seg_ver, "llamadas_llm": len(intentos),
            "json_valido": informe is not None, "intentos": intentos}


# --------------------------------------------------------------------------------- (d)
CORTO = {"pib_usd": "PIB", "pib_pc": "PIB per cápita", "crecimiento": "Crecimiento del PIB real",
         "inflacion": "Inflación", "desempleo": "Desempleo", "exportaciones": "Exportaciones (% del PIB)",
         "importaciones": "Importaciones (% del PIB)", "cuenta_corriente": "Cuenta corriente (% del PIB)",
         "ied": "Inversión extranjera directa (% del PIB)", "remesas": "Remesas (% del PIB)",
         "deuda_externa": "Deuda externa (% del INB)", "pobreza": "Pobreza extrema", "gini": "Índice de Gini",
         "poblacion": "Población"}


def _ref(h: capa_kpis.Hecho) -> Referencia:
    return Referencia(indicador=h.indicador, pais=h.pais, anio=h.anio, metrica=h.metrica, valor=h.valor)


_RECOMENDACION = {
    "crecimiento": "Revisar los supuestos de demanda y ventas del próximo año con un escenario de bajo crecimiento.",
    "inflacion": "Revisar precios, contratos indexados y costos de insumos ante la presión inflacionaria.",
    "desempleo": "Vigilar la calidad de cartera de consumo y la demanda de los hogares por el mayor desempleo.",
    "deuda_externa": "Seguir el perfil de vencimientos externos y el riesgo de refinanciamiento del país.",
    "pobreza": "Considerar el efecto de la pobreza en el consumo masivo y en la segmentación de clientes.",
    "gini": "Tener en cuenta la alta desigualdad al dimensionar mercados y precios por segmento.",
    "cuenta_corriente": "Vigilar la dependencia del financiamiento externo y el riesgo cambiario.",
    "ied": "Evaluar el clima de inversión y las condiciones para atraer capital extranjero.",
    "pib_pc": "Ajustar la oferta a un poder de compra por habitante menor que el de la región.",
}


def _texto_hallazgo(h_val: capa_kpis.Hecho, hs: dict) -> tuple[str, list[capa_kpis.Hecho]]:
    ind = config.IND_POR_CLAVE[h_val.indicador]
    usados = [h_val]
    partes = [f"{CORTO[ind.clave]}: {fmt_hecho(h_val)} en {h_val.anio}"]
    var = hs.get((h_val.indicador, h_val.pais, h_val.anio, "variacion"))
    if var:
        partes.append(f"({fmt_hecho(var, True)} frente a {h_val.anio - 1})")
        usados.append(var)
    med = hs.get((h_val.indicador, "LAT", h_val.anio, "mediana_region"))
    rk = hs.get((h_val.indicador, h_val.pais, h_val.anio, "ranking"))
    cola = []
    if med:
        cola.append(f"mediana regional de {fmt_hecho(med)}")
        usados.append(med)
    if rk:
        cola.append(fmt_hecho(rk))
        usados.append(rk)
    texto = " ".join(partes) + ("; " + ", ".join(cola) if cola else "") + "."
    if h_val.anomalia:
        texto += " Es un cambio fuera de lo normal para el país."
    return texto, usados


def generar_plantilla(pais: str, anio: int, **_: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    hs_lista = capa_kpis.hechos(pais, anio)
    hs = {(h.indicador, h.pais, h.anio, h.metrica): h for h in hs_lista}
    t = capa_kpis.tabla_larga()

    def actual(clave: str) -> capa_kpis.Hecho | None:
        v = [h for h in hs_lista if h.indicador == clave and h.metrica == "valor" and h.pais == pais]
        return max(v, key=lambda h: h.anio) if v else None

    crec = actual("crecimiento")
    med_c = hs.get(("crecimiento", "LAT", anio, "mediana_region"))
    verbo = "creció" if crec and crec.valor >= 0 else "se contrajo"
    titular = Afirmacion(
        texto=(f"{config.PAISES[pais]} {anio}: la economía {verbo} {fmt_hecho(crec).lstrip('-')}"
               + (f", frente a una mediana regional de {fmt_hecho(med_c)}." if med_c else ".")),
        referencias=[_ref(h) for h in (crec, med_c) if h])

    # Hallazgos: crecimiento, inflación, desempleo y los dos cambios más anómalos del resto (|z| mayor)
    fijos = ["inflacion", "desempleo"]
    resto = []
    for ind in config.INDICADORES:
        if ind.clave in ["crecimiento", *fijos, "poblacion"]:
            continue
        h = actual(ind.clave)
        if h is None:
            continue
        z = t[(t["indicador"] == ind.clave) & (t["iso3"] == pais) & (t["anio"] == h.anio)]["z_anomalia"]
        resto.append((abs(float(z.iloc[0])) if len(z) and z.notna().iloc[0] else 0.0, ind.clave))
    elegidos = fijos + [c for _, c in sorted(resto, reverse=True)[:2]]
    hallazgos = []
    for clave in elegidos:
        h = actual(clave)
        if h:
            texto, usados = _texto_hallazgo(h, hs)
            hallazgos.append(Afirmacion(texto=texto, referencias=[_ref(u) for u in usados]))

    # Riesgos: indicadores con dirección conocida donde el país está peor que la región (brecha mayor)
    riesgos_cand = []
    for ind in config.INDICADORES:
        if ind.mayor_es_mejor is None:
            continue
        h = actual(ind.clave)
        if h is None:
            continue
        b = hs.get((ind.clave, pais, h.anio, "brecha_region"))
        if b is None:
            continue
        peor = -b.valor if ind.mayor_es_mejor else b.valor
        if peor > 0:
            riesgos_cand.append((peor / (abs(hs[(ind.clave, "LAT", h.anio, "mediana_region")].valor) + 1), h, b))
    riesgos_cand.sort(key=lambda x: -x[0])
    riesgos, recomendaciones = [], []
    for _, h, b in riesgos_cand[:2]:
        ind = config.IND_POR_CLAVE[h.indicador]
        med = hs[(h.indicador, "LAT", h.anio, "mediana_region")]
        direccion = "por debajo" if ind.mayor_es_mejor else "por encima"
        riesgos.append(Afirmacion(
            texto=(f"{CORTO[ind.clave]} de {fmt_hecho(h)} en {h.anio}, {fmt_hecho(b).lstrip('-')} {direccion} de la "
                   f"mediana regional ({fmt_hecho(med)})."), referencias=[_ref(h), _ref(b), _ref(med)]))
        if h.indicador in _RECOMENDACION:
            recomendaciones.append(Afirmacion(texto=_RECOMENDACION[h.indicador], referencias=[]))
    if not riesgos:
        riesgos.append(Afirmacion(texto="Ningún indicador con dirección clara está peor que la mediana regional.",
                                  referencias=[]))
    if not recomendaciones:
        recomendaciones.append(Afirmacion(texto="Mantener el seguimiento trimestral de los indicadores clave.",
                                          referencias=[]))
    informe = Informe(titular=titular, hallazgos=hallazgos[:5], riesgos=riesgos, recomendaciones=recomendaciones)
    ver = verificador.verificar_informe(informe, pais, anio)
    return {"enfoque": "plantilla", "version": None, "pais": pais, "anio": anio, "texto": informe.texto_markdown(),
            "informe": informe.model_dump(), "verificacion": ver, "segundos_llm": 0.0,
            "segundos_verificacion": time.perf_counter() - t0, "llamadas_llm": 0, "json_valido": True}


def generar(enfoque: str, pais: str, anio: int, version: str = "v2", proveedor: str | None = None) -> dict[str, Any]:
    if enfoque == "libre":
        return generar_libre(pais, anio, version, proveedor)
    if enfoque == "estructurado":
        return generar_estructurado(pais, anio, version, False, proveedor)
    if enfoque == "verificado":
        return generar_estructurado(pais, anio, version, True, proveedor)
    if enfoque == "plantilla":
        return generar_plantilla(pais, anio)
    raise ValueError(enfoque)
