"""Motor de los 4 enfoques del Agente de Datos Abiertos (bucle real de herramientas) y cliente Ollama con grabación SHA-1."""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Literal

import httpx

from . import config, datos, herramientas

EnfoqueAgente = Literal[
    "zeroshot_sin_tools",
    "enrutador_reglas",
    "react_texto",
    "tool_calling_nativo",
]

_CACHE_MEMORIA_LLM: dict[str, dict[str, Any]] | None = None
HUELLAS_USADAS: set[str] = set()


# ===========================================================================
# 1. Cliente Ollama con grabación determinista por huella SHA-1
# ===========================================================================
def _cargar_cache_llm() -> dict[str, dict[str, Any]]:
    global _CACHE_MEMORIA_LLM
    if _CACHE_MEMORIA_LLM is None:
        _CACHE_MEMORIA_LLM = {}
        if config.GRABACION_LLM.exists():
            for linea in config.GRABACION_LLM.read_text(encoding="utf-8").splitlines():
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    item = json.loads(linea)
                    if "huella" in item:
                        _CACHE_MEMORIA_LLM[item["huella"]] = item
                except json.JSONDecodeError:
                    continue
    return _CACHE_MEMORIA_LLM


def _calcular_huella(payload_clave: dict[str, Any]) -> str:
    canonico = json.dumps(payload_clave, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonico.encode("utf-8")).hexdigest()


def llamar_ollama_chat(
    mensajes: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    formato_json: bool = False,
    num_predict: int = config.MAX_TOKENS_SALIDA,
    etiqueta_modo: str = "chat",
    forzar_ollama: bool = False,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Invoca /api/chat de Ollama (o recupera de data/grabacion_llm.jsonl si ya está grabado)."""
    cache = _cargar_cache_llm()
    nombres_tools = [t["function"]["name"] for t in tools] if tools else []
    clave_obj = {
        "modelo": config.MODELO_LLM,
        "modo": etiqueta_modo,
        "mensajes": mensajes,
        "tools": nombres_tools,
        "formato_json": formato_json,
    }
    if tools and etiqueta_modo.startswith("agente_"):
        # Las descripciones forman parte del prompt: si cambian, la grabación ya no es válida.
        clave_obj["tools_esquema"] = _calcular_huella({"t": tools})
    huella = _calcular_huella(clave_obj)
    HUELLAS_USADAS.add(huella)

    if huella in cache and not forzar_ollama:
        return cache[huella]

    proveedor_actual = "ollama" if forzar_ollama else config.PROVEEDOR
    if proveedor_actual == "grabado" and huella in cache:
        return cache[huella]

    body: dict[str, Any] = {
        "model": config.MODELO_LLM,
        "messages": mensajes,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": config.TEMPERATURA,
            "num_predict": num_predict,
            # 2048 en todas las llamadas: comparte la instancia cargada de qwen2.5:7b con otros
            # proyectos locales y evita recargar el modelo (no altera prompts tan cortos).
            "num_ctx": num_ctx or 2048,
            "seed": config.SEMILLA,
        },
    }
    if tools:
        body["tools"] = tools
    if formato_json:
        body["format"] = "json"

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=600.0) as cliente:
            resp = cliente.post(f"{config.OLLAMA_URL}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "La consulta no está grabada en data/grabacion_llm.jsonl y Ollama no responde en "
            f"{config.OLLAMA_URL}. Arranca `ollama serve` o usa una pregunta de los conjuntos evaluados."
        ) from exc
    latencia_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    msg = data.get("message", {})
    registro = {
        "huella": huella,
        "modo": etiqueta_modo,
        "modelo": config.MODELO_LLM,
        "content": msg.get("content", "") or "",
        "tool_calls": msg.get("tool_calls", []) or [],
        "tokens_prompt": int(data.get("prompt_eval_count", 0) or 0),
        "tokens_salida": int(data.get("eval_count", 0) or 0),
        "latencia_ms": latencia_ms,
    }

    cache[huella] = registro
    config.DATOS_DIR.mkdir(parents=True, exist_ok=True)
    with config.GRABACION_LLM.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro


# ===========================================================================
# 2. Extractores léxicos y guardrails de dominio (Abstención y Normalización)
# ===========================================================================
def analizar_intencion_consulta(pregunta: str) -> dict[str, Any]:
    """Extrae entidades, años, variables y detecta si la consulta está fuera del dominio soportado."""
    texto_min = pregunta.lower()
    texto_sin = datos.quitar_acentos(texto_min)

    # 1. Detectar años mencionados
    anios = [int(a) for a in re.findall(r"\b(19\d{2}|20\d{2})\b", pregunta)]

    # 2. Detectar países mencionados (orden de aparición)
    # Los códigos ISO3 de 3 letras (PER, USA, COL, etc.) solo se buscan en MAYÚSCULAS
    # para no confundir "per cápita" con PER ni el verbo "usa la calculadora" con USA.
    paises_detectados: list[str] = []
    posiciones_paises: list[tuple[int, str]] = []
    codigos_iso3 = set(config.PAISES.keys())
    for alias, iso3 in config.ALIAS_PAISES.items():
        alias_sin = datos.quitar_acentos(alias)
        if alias.upper() in codigos_iso3 and len(alias) <= 3:
            m = re.search(rf"\b{re.escape(alias.upper())}\b", pregunta)
        else:
            m = re.search(rf"\b{re.escape(alias_sin)}\b", texto_sin)
        if m:
            posiciones_paises.append((m.start(), iso3))
    posiciones_paises.sort(key=lambda x: x[0])
    for _, iso3 in posiciones_paises:
        if iso3 not in paises_detectados:
            paises_detectados.append(iso3)

    # 3. Detectar ciudades mencionadas
    ciudades_detectadas: list[str] = []
    posiciones_ciudades: list[tuple[int, str]] = []
    for alias_c, ciudad_can in config.ALIAS_CIUDADES.items():
        alias_c_sin = datos.quitar_acentos(alias_c)
        m = re.search(rf"\b{re.escape(alias_c_sin)}\b", texto_sin)
        if m:
            posiciones_ciudades.append((m.start(), ciudad_can))
    posiciones_ciudades.sort(key=lambda x: x[0])
    for _, c_can in posiciones_ciudades:
        if c_can not in ciudades_detectadas:
            ciudades_detectadas.append(c_can)

    # Si menciona ciudad pero no país explícito, podemos inferir el país asociado cuando hay cruce
    paises_inferidos = list(paises_detectados)
    for c_can in ciudades_detectadas:
        iso_c = str(config.CIUDADES_CLIMA[c_can]["codigo_pais"])
        if iso_c not in paises_inferidos:
            paises_inferidos.append(iso_c)

    # 4. Detectar indicador económico del Banco Mundial
    indicador_bm: str | None = None
    reglas_ind = [
        (r"pib\s+per\s+c[aá]pita|ingreso\s+per\s+c[aá]pita|ny\.gdp\.pcap", "pib_per_capita_usd"),
        (r"crecimiento\s+del\s+pib|crecimiento\s+econ[oó]mico|ny\.gdp\.mktp", "crecimiento_pib_pct"),
        (r"inflaci[oó]n|precios\s+al\s+consumidor|fp\.cpi\.totl", "inflacion_anual_pct"),
        (r"desempleo|paro\s+laboral|sl\.uem\.totl", "desempleo_pct"),
        (r"energ[ií]a[s]?\s+renovable[s]?|matriz\s+renovable|eg\.fec\.rnew", "energias_renovables_pct"),
        (r"esperanza\s+de\s+vida|longevidad|sp\.dyn\.le00", "esperanza_vida_anios"),
        (r"internet|conectividad\s+digital|it\.net\.user", "usuarios_internet_pct"),
        (r"gini|desigualdad\s+de\s+ingresos|si\.pov\.gini", "indice_gini"),
    ]
    for patron, ind_key in reglas_ind:
        if re.search(patron, texto_min):
            indicador_bm = ind_key
            break

    # 5. Detectar variable climática de Open-Meteo
    variable_clima: str | None = None
    pide_resumen_clima = False
    if re.search(r"resumen\s+clim[aá]tico|perfil\s+clim[aá]tico|todas\s+las\s+m[eé]tricas\s+clim[aá]ticas|temperatura\s+media.*precipitaci[oó]n|precipitaci[oó]n.*temperatura\s+media|lluvia.*d[ií]as\s+de\s+lluvia|d[ií]as\s+de\s+lluvia", texto_min):
        pide_resumen_clima = True

    reglas_clima = [
        (r"precipitaci[oó]n|lluvia\s+acumulada|mil[ií]metros\s+de\s+lluvia|llovi[oó]", "precipitacion_mm"),
        (r"temperatura\s+m[aá]xima|pico\s+de\s+calor|temp_max", "temp_max_c"),
        (r"temperatura\s+m[ií]nima|fr[ií]o\s+extremo|temp_min", "temp_min_c"),
        (r"radiaci[oó]n\s+solar|onda\s+corta", "radiacion_solar_mj_m2"),
        (r"viento\s+m[aá]ximo|r[aá]faga|velocidad\s+del\s+viento", "viento_max_kmh"),
        (r"temperatura\s+media|temperatura\s+promedio|clima\s+hist[oó]rico", "temp_media_c"),
    ]
    for patron, var_key in reglas_clima:
        if re.search(patron, texto_min):
            variable_clima = var_key
            break

    # 6. Detectar si pide cálculo matemático explícito (variación, diferencia, promedio, correlación)
    operacion_calculo: str | None = None
    if re.search(r"correlaci[oó]n\s+de\s+pearson|correlaci[oó]n\s+entre", texto_min):
        operacion_calculo = "correlacion"
    elif re.search(r"variaci[oó]n\s+porcentual|tasa\s+de\s+variaci[oó]n|cambio\s+porcentual|crecimiento\s+porcentual\s+acumulado|cu[aá]nto\s+vari[oó]\s+porcentualmente", texto_min):
        operacion_calculo = "variacion_porcentual"
    elif re.search(r"diferencia\s+exacta\s+y\s+porcentual|calcular\s+la\s+variaci[oó]n", texto_min):
        operacion_calculo = "variacion_porcentual"

    # 7. Fuera de dominio por CAPACIDADES (sin listas de temas prohibidos): la consulta no
    #    nombra ninguna medida que las herramientas sepan calcular, ningún país o ciudad del
    #    catálogo, o pide años fuera de la cobertura de las fuentes.
    motivo_fuera_dominio: str | None = None
    a_min, a_max = datos.cobertura_bm()
    c_ini, c_fin = datos.cobertura_clima()
    if not indicador_bm and not variable_clima and not pide_resumen_clima:
        motivo_fuera_dominio = "La consulta no pide ninguno de los indicadores o variables climáticas del catálogo."
    elif not paises_inferidos:
        motivo_fuera_dominio = "La consulta no nombra ningún país o ciudad del catálogo."
    elif anios and (max(anios) > a_max or min(anios) < a_min):
        motivo_fuera_dominio = f"El año pedido está fuera de la cobertura del Banco Mundial ({a_min}-{a_max})."
    elif anios and (variable_clima or pide_resumen_clima) and not indicador_bm and (
        min(anios) < int(c_ini[:4]) or max(anios) > int(c_fin[:4])
    ):
        motivo_fuera_dominio = f"El año pedido está fuera de la cobertura climática ({c_ini} a {c_fin})."

    return {
        "anios": anios,
        "paises": paises_detectados,
        "paises_inferidos": paises_inferidos,
        "ciudades": ciudades_detectadas,
        "indicador_bm": indicador_bm,
        "variable_clima": variable_clima,
        "pide_resumen_clima": pide_resumen_clima,
        "operacion_calculo": operacion_calculo,
        "es_fuera_dominio": motivo_fuera_dominio is not None,
        "motivo_fuera_dominio": motivo_fuera_dominio,
    }


def sintetizar_respuesta_verificada(
    pregunta: str,
    ejecuciones: list[dict[str, Any]],
) -> str:
    """Construye una respuesta en español concisa y 100 % fiel a las observaciones de las herramientas."""
    if not ejecuciones:
        a_min, a_max = datos.cobertura_bm()
        c_ini, c_fin = datos.cobertura_clima()
        return (
            "ABSTENCIÓN: No puedo responder a esta consulta porque solicita datos, países, "
            f"fechas o indicadores fuera del catálogo del Banco Mundial ({a_min}-{a_max}) "
            f"y Open-Meteo ({c_ini[:4]}-{c_fin[:4]}) disponible en el modo '{config.MODO_DATOS}'."
        )

    bloques: list[str] = []
    for ej in ejecuciones:
        if not ej.get("exito"):
            continue
        nombre = ej["herramienta"]
        res = ej["resultado"]

        if nombre == "consultar_indicador_banco_mundial":
            serie = res.get("serie_anual", {})
            pares = ", ".join(f"{a}: {v:.2f} {res['unidad']}" for a, v in serie.items())
            resumen = res.get("resumen", {})
            extra = ""
            if resumen.get("n_observaciones", 0) > 1:
                extra = (
                    f" (inicial {resumen['valor_inicial']:.2f}, final {resumen['valor_final']:.2f}, "
                    f"promedio {resumen['promedio_periodo']:.2f} {res['unidad']})"
                )
            bloques.append(
                f"Según el Banco Mundial ({res['codigo_bm']}), {res['nombre_indicador']} en "
                f"{res['pais']} ({res['codigo_pais']}) registró [{pares}]{extra}."
            )

        elif nombre == "comparar_paises_indicador":
            ranking = res.get("ranking", [])
            items = ", ".join(
                f"{r['pais']} ({r['codigo_pais']}): {r['valor']:.2f} {res['unidad']}"
                for r in ranking
            )
            resumen = res.get("resumen", {})
            brecha_txt = ""
            if resumen:
                brecha_txt = (
                    f" Lidera {resumen['pais_mayor']} ({resumen['valor_mayor']:.2f}) frente a "
                    f"{resumen['pais_menor']} ({resumen['valor_menor']:.2f}), con una brecha de "
                    f"{resumen['brecha_mayor_menor']:.2f} {res['unidad']} y promedio grupal de "
                    f"{resumen['promedio_grupo']:.2f} {res['unidad']}."
                )
            bloques.append(
                f"Comparativa {res['anio_solicitado']} de {res['nombre_indicador']} ({res['codigo_bm']}): "
                f"{items}.{brecha_txt}"
            )

        elif nombre == "consultar_clima_historico_ciudad":
            bloques.append(
                f"En {res['ciudad']} ({res['codigo_pais']}), entre {res['fecha_inicio']} y "
                f"{res['fecha_fin']}, {res['nombre_variable']} fue de {res['valor_periodo']:.2f} {res['unidad']} "
                f"(promedio mensual {res['promedio_mensual']:.2f} {res['unidad']}, máximo mensual "
                f"{res['maximo_mensual']:.2f} en {res['mes_maximo']} y mínimo mensual "
                f"{res['minimo_mensual']:.2f} en {res['mes_minimo']})."
            )

        elif nombre == "resumen_climatico_anual_ciudad":
            m = res["metricas_anuales"]
            bloques.append(
                f"Resumen climático de Open-Meteo para {res['ciudad']} ({res['codigo_pais']}) en {res['anio']}: "
                f"temperatura media de {m['temp_media_c']:.2f} °C (máxima {m['temp_max_c']:.2f} °C, "
                f"mínima {m['temp_min_c']:.2f} °C), precipitación acumulada de {m['precipitacion_mm']:.2f} mm "
                f"en {m['dias_lluvia']} días de lluvia, radiación solar media de "
                f"{m['radiacion_solar_mj_m2']:.2f} MJ/m² y viento máximo de {m['viento_max_kmh']:.2f} km/h."
            )

        elif nombre == "calcular_correlacion_o_cambio":
            op = res["operacion"]
            if op == "variacion_porcentual":
                bloques.append(
                    f"Cálculo determinista: al pasar de {res['valor_inicial']:.2f} a {res['valor_final']:.2f}, "
                    f"el cambio absoluto es {res['cambio_absoluto']:.2f} y la variación porcentual exacta "
                    f"es {res['variacion_porcentual']:.2f} %."
                )
            elif op == "diferencia":
                bloques.append(
                    f"Cálculo determinista: la diferencia entre {res['primer_valor']:.2f} y "
                    f"{res['segundo_valor']:.2f} es {res['diferencia_primero_menos_segundo']:.2f} "
                    f"(diferencia relativa de {res['diferencia_relativa_pct']:.2f} %)."
                )
            elif op == "promedio":
                bloques.append(
                    f"Cálculo determinista: promedio {res['promedio']:.2f}, mediana {res['mediana']:.2f}, "
                    f"mínimo {res['minimo']:.2f} y máximo {res['maximo']:.2f}."
                )
            elif op == "correlacion":
                bloques.append(
                    f"Cálculo determinista: correlación de Pearson r = {res['correlacion_pearson']:.4f} "
                    f"({res['correlacion_pearson_2dec']:.2f}), Spearman = {res['correlacion_spearman']:.4f}."
                )

    return " ".join(bloques)


# ===========================================================================
# 3. Implementación de las 4 Arquitecturas de Agente
# ===========================================================================
def ejecutar_enfoque_1_zeroshot_sin_tools(
    pregunta: str,
    forzar_ollama: bool = False,
) -> dict[str, Any]:
    """Enfoque 1: LLM directo sin herramientas (qwen2.5:7b responde de memoria)."""
    mensajes = [
        {
            "role": "system",
            "content": (
                "Analista económico y climático. Responde en español en 1 oración con cifras exactas. "
                "Alcance: 12 países (ECU, COL, PER, CHL, ARG, BRA, MEX, URY, CRI, BOL, PRY, USA), "
                "Banco Mundial (1990-2024) y Open-Meteo (2020-2024). Si piden cripto/bolsa, otros países, "
                "años >2024, clima <2020 o datos ficticios, responde solo: 'ABSTENCION: Fuera de dominio'."
            ),
        },
        {"role": "user", "content": pregunta},
    ]
    resp = llamar_ollama_chat(
        mensajes=mensajes,
        tools=None,
        formato_json=False,
        num_predict=42,
        etiqueta_modo="enfoque1_zeroshot",
        forzar_ollama=forzar_ollama,
    )
    texto = resp["content"].strip()
    abstuvo = bool(
        re.search(
            r"abstenci[oó]n|fuera\s+de\s+dominio|no\s+dispongo|no\s+tengo\s+acceso|no\s+puedo\s+proporcionar|inexistente",
            texto.lower(),
        )
    )
    return {
        "enfoque": "zeroshot_sin_tools",
        "pregunta": pregunta,
        "llamadas_herramientas": [],
        "ejecuciones_herramientas": [],
        "respuesta_final": texto,
        "se_abstuvo": abstuvo,
        "reintentos_validacion": 0,
        "latencia_ms": resp["latencia_ms"],
        "tokens_prompt": resp["tokens_prompt"],
        "tokens_salida": resp["tokens_salida"],
    }


def ejecutar_enfoque_2_enrutador_reglas(pregunta: str) -> dict[str, Any]:
    """Enfoque 2: Enrutador determinístico léxico por reglas (1 sola llamada, sin encadenamiento)."""
    t0 = time.perf_counter()
    info = analizar_intencion_consulta(pregunta)

    abstencion_basica = info["es_fuera_dominio"]
    if abstencion_basica or (not info["indicador_bm"] and not info["variable_clima"] and not info["pide_resumen_clima"]):
        lat = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "enfoque": "enrutador_reglas",
            "pregunta": pregunta,
            "llamadas_herramientas": [],
            "ejecuciones_herramientas": [],
            "respuesta_final": "ABSTENCIÓN: Consulta fuera del catálogo léxico de reglas.",
            "se_abstuvo": True,
            "reintentos_validacion": 0,
            "latencia_ms": lat,
            "tokens_prompt": 0,
            "tokens_salida": 0,
        }

    # Limitación estructural del enrutador por reglas: solo invoca UNA herramienta primaria
    # y no sabe encadenar multi-API (Clima + Economía) ni invocar la calculadora determinista.
    llamada: dict[str, Any] | None = None
    anios = info["anios"]

    if info["indicador_bm"]:
        if len(info["paises"]) >= 2:
            anio_comp = anios[0] if anios else 2022
            llamada = {
                "herramienta": "comparar_paises_indicador",
                "argumentos": {
                    "codigos_paises": info["paises"],
                    "indicador": info["indicador_bm"],
                    "anio": anio_comp,
                },
            }
        elif len(info["paises_inferidos"]) >= 1:
            a_ini = min(anios) if anios else 2020
            a_fin = max(anios) if anios else 2022
            llamada = {
                "herramienta": "consultar_indicador_banco_mundial",
                "argumentos": {
                    "codigo_pais": info["paises_inferidos"][0],
                    "indicador": info["indicador_bm"],
                    "anio_inicio": a_ini,
                    "anio_fin": a_fin,
                },
            }
    elif info["ciudades"]:
        ciudad = info["ciudades"][0]
        anio_c = anios[0] if anios else 2023
        if info["pide_resumen_clima"] and not info["variable_clima"]:
            llamada = {
                "herramienta": "resumen_climatico_anual_ciudad",
                "argumentos": {"ciudad": ciudad, "anio": anio_c},
            }
        else:
            var_c = info["variable_clima"] or "temp_media_c"
            a_ini = min(anios) if anios else 2023
            a_fin = max(anios) if anios else a_ini
            llamada = {
                "herramienta": "consultar_clima_historico_ciudad",
                "argumentos": {
                    "ciudad": ciudad,
                    "fecha_inicio": f"{a_ini}-01-01",
                    "fecha_fin": f"{a_fin}-12-31",
                    "variable": var_c,
                },
            }

    if llamada is None:
        lat = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "enfoque": "enrutador_reglas",
            "pregunta": pregunta,
            "llamadas_herramientas": [],
            "ejecuciones_herramientas": [],
            "respuesta_final": "ABSTENCIÓN: No se identificó entidad válida.",
            "se_abstuvo": True,
            "reintentos_validacion": 0,
            "latencia_ms": lat,
            "tokens_prompt": 0,
            "tokens_salida": 0,
        }

    ejec = herramientas.ejecutar_herramienta_segura(llamada["herramienta"], llamada["argumentos"])
    ejecuciones = [ejec]
    if ejec["exito"]:
        resp_txt = sintetizar_respuesta_verificada(pregunta, ejecuciones)
        abstuvo = False
    else:
        resp_txt = f"Error en enrutador de reglas: {ejec['mensaje_error']}"
        abstuvo = False

    lat = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "enfoque": "enrutador_reglas",
        "pregunta": pregunta,
        "llamadas_herramientas": [llamada],
        "ejecuciones_herramientas": ejecuciones,
        "respuesta_final": resp_txt,
        "se_abstuvo": abstuvo,
        "reintentos_validacion": 0,
        "latencia_ms": lat,
        "tokens_prompt": 0,
        "tokens_salida": 0,
    }


def ejecutar_enfoque_3_react_texto(
    pregunta: str,
    forzar_ollama: bool = False,
) -> dict[str, Any]:
    """Enfoque 3: Agente ReAct en texto (Thought/Action/Observation sin tools nativo ni guardrails Pydantic)."""
    prompt_sistema = (
        "Agente ReAct JSON. Herramientas:\n"
        "1. consultar_indicador_banco_mundial(codigo_pais, indicador, anio_inicio, anio_fin) "
        "[paises: ECU,COL,PER,CHL,ARG,BRA,MEX,URY,CRI,BOL,PRY,USA; indicadores: pib_per_capita_usd, "
        "crecimiento_pib_pct, inflacion_anual_pct, desempleo_pct, energias_renovables_pct, "
        "esperanza_vida_anios, usuarios_internet_pct, indice_gini]\n"
        "2. comparar_paises_indicador(codigos_paises, indicador, anio)\n"
        "3. consultar_clima_historico_ciudad(ciudad, fecha_inicio, fecha_fin, variable) "
        "[variable: temp_media_c, temp_max_c, temp_min_c, precipitacion_mm, radiacion_solar_mj_m2, viento_max_kmh; 2020-2024]\n"
        "4. resumen_climatico_anual_ciudad(ciudad, anio)\n"
        "5. calcular_correlacion_o_cambio(valores_x, valores_y, operacion)\n"
        "Si está fuera de dominio devuelve {\"abstenerse\":true,\"acciones\":[]}. "
        "Si es válida devuelve {\"abstenerse\":false,\"acciones\":[{\"herramienta\":\"...\",\"argumentos\":{...}}]}."
    )
    mensajes = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": pregunta},
    ]
    resp = llamar_ollama_chat(
        mensajes=mensajes,
        tools=None,
        formato_json=True,
        num_predict=70,
        etiqueta_modo="enfoque3_react_texto",
        forzar_ollama=forzar_ollama,
    )

    contenido = resp["content"].strip()
    acciones_raw: list[dict[str, Any]] = []
    abstenerse = False
    try:
        parsed = json.loads(contenido)
        abstenerse = bool(parsed.get("abstenerse", False))
        acciones_raw = parsed.get("acciones", []) or []
        if not isinstance(acciones_raw, list):
            acciones_raw = []
    except Exception:
        abstenerse = "abstenerse\": true" in contenido.lower()

    if abstenerse or not acciones_raw:
        return {
            "enfoque": "react_texto",
            "pregunta": pregunta,
            "llamadas_herramientas": [],
            "ejecuciones_herramientas": [],
            "respuesta_final": (
                "ABSTENCIÓN (ReAct): La consulta solicita información fuera del alcance de las herramientas."
            ),
            "se_abstuvo": True,
            "reintentos_validacion": 0,
            "latencia_ms": resp["latencia_ms"],
            "tokens_prompt": resp["tokens_prompt"],
            "tokens_salida": resp["tokens_salida"],
        }

    llamadas_realizadas: list[dict[str, Any]] = []
    ejecuciones: list[dict[str, Any]] = []

    for acc in acciones_raw:
        if not isinstance(acc, dict):
            continue
        nom = str(acc.get("herramienta", "")).strip()
        args = acc.get("argumentos", {}) or {}
        if not isinstance(args, dict):
            args = {}
        llamadas_realizadas.append({"herramienta": nom, "argumentos": args})
        # En ReAct texto sin guardrails, si falla Pydantic NO hay bucle de auto-reparación
        ejec = herramientas.ejecutar_herramienta_segura(nom, args)
        ejecuciones.append(ejec)

    exitosas = [e for e in ejecuciones if e.get("exito")]
    if not exitosas:
        errores = "; ".join(e.get("mensaje_error", "error") for e in ejecuciones)
        return {
            "enfoque": "react_texto",
            "pregunta": pregunta,
            "llamadas_herramientas": llamadas_realizadas,
            "ejecuciones_herramientas": ejecuciones,
            "respuesta_final": f"No se pudo completar la consulta en ReAct: {errores}",
            "se_abstuvo": False,
            "reintentos_validacion": 0,
            "latencia_ms": resp["latencia_ms"],
            "tokens_prompt": resp["tokens_prompt"],
            "tokens_salida": resp["tokens_salida"],
        }

    respuesta_final = sintetizar_respuesta_verificada(pregunta, exitosas)
    return {
        "enfoque": "react_texto",
        "pregunta": pregunta,
        "llamadas_herramientas": llamadas_realizadas,
        "ejecuciones_herramientas": ejecuciones,
        "respuesta_final": respuesta_final,
        "se_abstuvo": False,
        "reintentos_validacion": 0,
        "latencia_ms": resp["latencia_ms"],
        "tokens_prompt": resp["tokens_prompt"],
        "tokens_salida": resp["tokens_salida"],
    }


# ===========================================================================
# 4. Agente con herramientas nativas (bucle real de varios turnos)
#    El LLM decide QUÉ herramienta llamar, CON QUÉ argumentos y CUÁNDO abstenerse.
#    El código solo (a) valida tipos y rangos con Pydantic, (b) verifica que cada
#    argumento esté anclado en la pregunta, (c) ejecuta y (d) redacta la respuesta
#    final con una plantilla a partir de las observaciones.
# ===========================================================================
def _normalizar_texto(texto: str) -> str:
    return datos.quitar_acentos(texto.lower())


def ciudades_mencionadas(pregunta: str) -> set[str]:
    texto = _normalizar_texto(pregunta)
    return {
        ciudad
        for alias, ciudad in config.ALIAS_CIUDADES.items()
        if re.search(rf"\b{re.escape(datos.quitar_acentos(alias))}\b", texto)
    }


def paises_mencionados(pregunta: str) -> set[str]:
    """Países nombrados (nombre, alias o código ISO3 en mayúsculas) o implícitos por una ciudad nombrada."""
    texto = _normalizar_texto(pregunta)
    encontrados: set[str] = set()
    for alias, iso3 in config.ALIAS_PAISES.items():
        if len(alias) <= 3:  # códigos cortos solo en MAYÚSCULAS ("per cápita" no es PER)
            if re.search(rf"\b{re.escape(alias.upper())}\b", pregunta):
                encontrados.add(iso3)
        elif re.search(rf"\b{re.escape(datos.quitar_acentos(alias))}\b", texto):
            encontrados.add(iso3)
    for ciudad in ciudades_mencionadas(pregunta):
        encontrados.add(str(config.CIUDADES_CLIMA[ciudad]["codigo_pais"]))
    return encontrados


def anios_mencionados(pregunta: str) -> list[int]:
    return [int(a) for a in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", pregunta)]


def _menciona_alguno(pregunta: str, terminos: list[str]) -> bool:
    texto = _normalizar_texto(pregunta)
    return any(re.search(rf"\b{re.escape(t)}", texto) for t in terminos)


def _numeros_en(objeto: Any) -> list[float]:
    if isinstance(objeto, bool):
        return []
    if isinstance(objeto, (int, float)):
        return [float(objeto)]
    if isinstance(objeto, dict):
        return [n for v in objeto.values() for n in _numeros_en(v)]
    if isinstance(objeto, (list, tuple)):
        return [n for v in objeto for n in _numeros_en(v)]
    return []


def verificar_anclaje(
    nombre: str,
    args: dict[str, Any],
    pregunta: str,
    ejecuciones_previas: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Regla general basada en las capacidades de las herramientas: cada argumento que elige el LLM
    (país, ciudad, años, medida, cifras de la calculadora) debe tener respaldo en la pregunta o en
    observaciones anteriores. No usa listas de temas prohibidos."""
    anios = anios_mencionados(pregunta)

    def _anios_ok(*valores: int) -> str | None:
        if not anios:
            return "la pregunta no indica ningún año o periodo; no inventes fechas"
        fuera = [v for v in valores if not (min(anios) <= v <= max(anios))]
        if fuera:
            return f"los años {fuera} no están en el periodo que menciona la pregunta ({min(anios)}-{max(anios)})"
        return None

    problemas: list[str] = []
    if nombre in ("consultar_indicador_banco_mundial", "comparar_paises_indicador"):
        codigos = args.get("codigos_paises") or [args.get("codigo_pais")]
        no_nombrados = [c for c in codigos if c not in paises_mencionados(pregunta)]
        if no_nombrados:
            problemas.append(f"los países {no_nombrados} no aparecen en la pregunta")
        if not _menciona_alguno(pregunta, config.TERMINOS_INDICADORES[args["indicador"]]):
            problemas.append(f"la pregunta no pide el indicador '{args['indicador']}'")
        if nombre == "comparar_paises_indicador":
            problema_anios = _anios_ok(args["anio"])
        else:
            problema_anios = _anios_ok(args["anio_inicio"], args["anio_fin"])
        if problema_anios:
            problemas.append(problema_anios)

    elif nombre in ("consultar_clima_historico_ciudad", "resumen_climatico_anual_ciudad"):
        if args["ciudad"] not in ciudades_mencionadas(pregunta):
            problemas.append(f"la ciudad '{args['ciudad']}' no aparece en la pregunta")
        if nombre == "consultar_clima_historico_ciudad":
            if not _menciona_alguno(pregunta, config.TERMINOS_VARIABLES_CLIMA[args["variable"]]):
                problemas.append(f"la pregunta no pide la variable '{args['variable']}'")
            problema_anios = _anios_ok(int(args["fecha_inicio"][:4]), int(args["fecha_fin"][:4]))
        else:
            if not _menciona_alguno(pregunta, config.TERMINOS_RESUMEN_CLIMA):
                problemas.append("la pregunta no pide datos de clima")
            problema_anios = _anios_ok(args["anio"])
        if problema_anios:
            problemas.append(problema_anios)

    elif nombre == "calcular_correlacion_o_cambio":
        disponibles = _numeros_en([e["resultado"] for e in ejecuciones_previas]) + [
            float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", pregunta)
        ]
        valores = list(args.get("valores_x") or []) + list(args.get("valores_y") or [])
        sin_respaldo = [
            v for v in valores if not any(abs(v - d) <= max(0.01, 0.001 * abs(d)) for d in disponibles)
        ]
        if sin_respaldo:
            problemas.append(
                f"los valores {sin_respaldo} no salen de observaciones anteriores ni de la pregunta; "
                "consulta primero los datos y copia los valores observados"
            )

    if problemas:
        return False, "ERROR de anclaje: " + "; ".join(problemas) + "."
    return True, ""


def observacion_para_llm(ejecucion: dict[str, Any]) -> str:
    """Resultado compacto que se devuelve al LLM como mensaje `tool`."""
    if not ejecucion["exito"]:
        return (
            f"ERROR: {ejecucion['mensaje_error']} Corrige los argumentos o, si la pregunta no cabe "
            "en las herramientas, no llames más herramientas y responde 'ABSTENCION'."
        )
    campos = {
        "consultar_indicador_banco_mundial": ["codigo_pais", "indicador", "unidad", "serie_anual"],
        "comparar_paises_indicador": ["indicador", "unidad", "anio_solicitado", "ranking"],
        "consultar_clima_historico_ciudad": [
            "ciudad", "variable", "unidad", "fecha_inicio", "fecha_fin", "tipo_agregacion", "valor_periodo"
        ],
        "resumen_climatico_anual_ciudad": ["ciudad", "anio", "metricas_anuales"],
    }
    res = ejecucion["resultado"]
    claves = campos.get(ejecucion["herramienta"]) or [k for k in res if k not in ("cifras_clave", "fuente")]
    compacto = {k: res[k] for k in claves if k in res}
    if "ranking" in compacto:
        compacto["ranking"] = [{k: r[k] for k in ("codigo_pais", "anio", "valor")} for r in compacto["ranking"]]
    return json.dumps(compacto, ensure_ascii=False, separators=(",", ":"))


def prompt_sistema_agente(version: str) -> str:
    a_min, a_max = datos.cobertura_bm()
    c_ini, c_fin = datos.cobertura_clima()
    base = (
        "Eres un agente de datos abiertos. Solo puedes responder con las herramientas: indicadores del "
        f"Banco Mundial ({a_min}-{a_max}) y clima histórico de Open-Meteo ({c_ini} a {c_fin}) para los "
        "países y ciudades que listan sus descripciones. Usa exactamente "
        "los países, ciudades y años que menciona la pregunta. Si la pregunta pide algo que estas "
        "herramientas no cubren (otro tema, otro lugar u otro periodo), no llames herramientas y "
        "responde 'ABSTENCION' explicando por qué. Cuando ya tengas los datos, responde en español "
        "en 1-2 oraciones con las cifras exactas."
    )
    if version == "v1":
        return base
    return base + (
        " Pasos: (1) si la pregunta nombra 2 o más países para un mismo año, usa comparar_paises_indicador; "
        "(2) si pide un dato económico de un país y el clima de una ciudad, llama las dos herramientas "
        "en el mismo turno; (3) si pide una variación porcentual, diferencia o promedio, primero consulta "
        "los datos y en el turno siguiente llama calcular_correlacion_o_cambio con los valores observados; "
        "(4) si una herramienta devuelve ERROR, corrige los argumentos o abstente."
    )


def ejecutar_agente_herramientas(
    pregunta: str,
    version_prompt: str | None = None,
    forzar_ollama: bool = False,
    max_pasos: int | None = None,
    usar_anclaje: bool = True,
) -> dict[str, Any]:
    """Enfoque 4: bucle de tool-calling nativo de Ollama (hasta `max_pasos` turnos del LLM)."""
    version = version_prompt or config.VERSION_PROMPT_AGENTE
    esquemas = herramientas.obtener_esquemas_ollama(version)
    mensajes: list[dict[str, Any]] = [
        {"role": "system", "content": prompt_sistema_agente(version)},
        {"role": "user", "content": pregunta},
    ]
    pasos: list[dict[str, Any]] = []
    ejecuciones_ok: list[dict[str, Any]] = []
    llamadas_finales: list[dict[str, Any]] = []
    texto_llm = ""
    latencia = 0.0
    tokens_prompt = tokens_salida = 0

    for n_paso in range(1, (max_pasos or config.MAX_PASOS_AGENTE) + 1):
        resp = llamar_ollama_chat(
            mensajes=mensajes,
            tools=esquemas,
            num_predict=160 if n_paso == 1 else 96,  # el 1.er turno puede traer 2 llamadas
            etiqueta_modo=f"agente_{version}",
            forzar_ollama=forzar_ollama,
            num_ctx=2048,
        )
        latencia += float(resp["latencia_ms"])
        tokens_prompt += int(resp["tokens_prompt"])
        tokens_salida += int(resp["tokens_salida"])
        tool_calls = resp.get("tool_calls") or []
        paso: dict[str, Any] = {"paso": n_paso, "texto_llm": resp["content"], "llamadas": []}
        pasos.append(paso)
        if not tool_calls:
            texto_llm = resp["content"].strip()
            break

        mensajes.append({"role": "assistant", "content": resp["content"], "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            nombre = str(fn.get("name", "")).strip()
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            ejec = herramientas.ejecutar_herramienta_segura(nombre, args)
            if ejec["exito"] and usar_anclaje:
                anclado, motivo = verificar_anclaje(nombre, ejec["argumentos_validados"], pregunta, ejecuciones_ok)
                if not anclado:
                    ejec = {
                        "exito": False,
                        "herramienta": nombre,
                        "argumentos_originales": args,
                        "argumentos_validados": ejec["argumentos_validados"],
                        "error_tipo": "anclaje",
                        "mensaje_error": motivo,
                    }
            paso["llamadas"].append({
                "herramienta": nombre,
                "argumentos_llm": args,
                "argumentos_validados": ejec.get("argumentos_validados"),
                "exito": ejec["exito"],
                "error_tipo": ejec.get("error_tipo"),
                "mensaje_error": ejec.get("mensaje_error"),
            })
            if ejec["exito"]:
                llamada = {"herramienta": nombre, "argumentos": ejec["argumentos_validados"]}
                if llamada not in llamadas_finales:  # el LLM a veces repite una llamada ya hecha
                    llamadas_finales.append(llamada)
                    ejecuciones_ok.append(ejec)
            mensajes.append({"role": "tool", "tool_name": nombre, "content": observacion_para_llm(ejec)})

    return {
        "enfoque": "tool_calling_nativo",
        "version_prompt": version,
        "pregunta": pregunta,
        "llamadas_herramientas": llamadas_finales,
        "ejecuciones_herramientas": ejecuciones_ok,
        "pasos": pasos,
        "n_pasos_llm": len(pasos),
        "respuesta_final": sintetizar_respuesta_verificada(pregunta, ejecuciones_ok),
        "texto_llm": texto_llm,
        "se_abstuvo": len(ejecuciones_ok) == 0,
        "reintentos_validacion": sum(1 for p in pasos for c in p["llamadas"] if not c["exito"]),
        "latencia_ms": round(latencia, 2),
        "tokens_prompt": tokens_prompt,
        "tokens_salida": tokens_salida,
    }


def consultar_agente(
    pregunta: str,
    enfoque: EnfoqueAgente = "tool_calling_nativo",
    forzar_ollama: bool = False,
    version_prompt: str | None = None,
) -> dict[str, Any]:
    """Punto de entrada unificado para consultar cualquiera de las 4 arquitecturas."""
    if enfoque == "zeroshot_sin_tools":
        return ejecutar_enfoque_1_zeroshot_sin_tools(pregunta, forzar_ollama=forzar_ollama)
    if enfoque == "enrutador_reglas":
        return ejecutar_enfoque_2_enrutador_reglas(pregunta)
    if enfoque == "react_texto":
        return ejecutar_enfoque_3_react_texto(pregunta, forzar_ollama=forzar_ollama)
    if enfoque == "tool_calling_nativo":
        return ejecutar_agente_herramientas(pregunta, version_prompt=version_prompt, forzar_ollama=forzar_ollama)
    raise ValueError(f"Enfoque '{enfoque}' desconocido.")
