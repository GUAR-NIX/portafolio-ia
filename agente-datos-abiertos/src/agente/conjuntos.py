"""Conjunto de PRUEBA nuevo (60 preguntas, T01-T60), redactado después de congelar el código del agente.

El conjunto de VALIDACIÓN son las 44 preguntas originales (V01-V44, `evaluacion._especificaciones_base_preguntas`):
sobre ellas se ajustaron las reglas del enrutador y se eligió la versión de prompt. Este conjunto de prueba usa
otros países, años, ciudades y redacciones, y 16 preguntas fuera de dominio de tipos distintos a las de validación
(sin cripto, bolsa, Marte, España ni Japón).
"""
from __future__ import annotations

from typing import Any


def _bm(pais: str, indicador: str, a_ini: int, a_fin: int | None = None) -> dict[str, Any]:
    return {
        "herramienta": "consultar_indicador_banco_mundial",
        "argumentos": {"codigo_pais": pais, "indicador": indicador, "anio_inicio": a_ini, "anio_fin": a_fin or a_ini},
    }


def _comp(paises: list[str], indicador: str, anio: int) -> dict[str, Any]:
    return {
        "herramienta": "comparar_paises_indicador",
        "argumentos": {"codigos_paises": paises, "indicador": indicador, "anio": anio},
    }


def _clima(ciudad: str, variable: str, f_ini: str, f_fin: str) -> dict[str, Any]:
    return {
        "herramienta": "consultar_clima_historico_ciudad",
        "argumentos": {"ciudad": ciudad, "fecha_inicio": f_ini, "fecha_fin": f_fin, "variable": variable},
    }


def _anio_clima(ciudad: str, variable: str, anio: int) -> dict[str, Any]:
    return _clima(ciudad, variable, f"{anio}-01-01", f"{anio}-12-31")


def _resumen(ciudad: str, anio: int) -> dict[str, Any]:
    return {"herramienta": "resumen_climatico_anual_ciudad", "argumentos": {"ciudad": ciudad, "anio": anio}}


_CALC = {"herramienta": "calcular_correlacion_o_cambio", "argumentos": {"operacion": "variacion_porcentual"}}


def _q(
    id_: str,
    categoria: str,
    pregunta: str,
    llamadas: list[dict[str, Any]],
    alternativas: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "id_pregunta": id_,
        "categoria": categoria,
        "pregunta": pregunta,
        "debe_abstenerse": categoria == "fuera_dominio",
        "llamadas_esperadas": llamadas,
        # Otras secuencias igual de válidas (p. ej. resumen anual en vez de una variable suelta).
        "alternativas": alternativas or [],
    }


def especificaciones_prueba() -> list[dict[str, Any]]:
    s, c, x, k, o = "simple_1hop", "comparativa_paises", "cruce_clima_economia", "calculo_variacion", "fuera_dominio"
    return [
        # --- simple_1hop (14) ---
        _q("T01", s, "¿Cómo evolucionó el crecimiento económico de Bolivia de 2016 a 2020?",
           [_bm("BOL", "crecimiento_pib_pct", 2016, 2020)]),
        _q("T02", s, "Necesito la inflación de Argentina para los años 2021 y 2022.",
           [_bm("ARG", "inflacion_anual_pct", 2021, 2022)]),
        _q("T03", s, "Dame la esperanza de vida de México entre 2015 y 2019.",
           [_bm("MEX", "esperanza_vida_anios", 2015, 2019)]),
        _q("T04", s, "¿Qué tan alto fue el desempleo en Brasil de 2019 a 2021?",
           [_bm("BRA", "desempleo_pct", 2019, 2021)]),
        _q("T05", s, "Muéstrame el índice de Gini de Paraguay entre 2017 y 2021.",
           [_bm("PRY", "indice_gini", 2017, 2021)]),
        _q("T06", s, "¿Qué porcentaje de los mexicanos usaba internet en 2020?",
           [_bm("MEX", "usuarios_internet_pct", 2020)]),
        _q("T07", s, "Consulta el PIB per cápita de Estados Unidos en 2010 y 2011.",
           [_bm("USA", "pib_per_capita_usd", 2010, 2011)]),
        _q("T08", s, "¿Cuánto llovió en Lima durante 2021?",
           [_anio_clima("Lima", "precipitacion_mm", 2021)], [[_resumen("Lima", 2021)]]),
        _q("T09", s, "¿Cuál fue la temperatura más baja registrada en Bogotá entre enero y junio de 2022?",
           [_clima("Bogotá", "temp_min_c", "2022-01-01", "2022-06-30")]),
        _q("T10", s, "Resumen del clima de Montevideo en 2020.", [_resumen("Montevideo", 2020)]),
        _q("T11", s, "¿Qué tan fuerte sopló el viento en Asunción en 2022? Dame la ráfaga máxima del año.",
           [_anio_clima("Asunción", "viento_max_kmh", 2022)], [[_resumen("Asunción", 2022)]]),
        _q("T12", s, "Temperatura media de San José (Costa Rica) en 2021.",
           [_anio_clima("San José", "temp_media_c", 2021)], [[_resumen("San José", 2021)]]),
        _q("T13", s, "¿Cuál fue la radiación solar promedio en Guayaquil entre 2020-03-01 y 2020-08-31?",
           [_clima("Guayaquil", "radiacion_solar_mj_m2", "2020-03-01", "2020-08-31")]),
        _q("T14", s, "¿Cuál fue el pico de calor en Ciudad de México en 2023?",
           [_anio_clima("Ciudad de México", "temp_max_c", 2023)], [[_resumen("Ciudad de México", 2023)]]),
        # --- comparativa_paises (10) ---
        _q("T15", c, "Ordena de mayor a menor la esperanza de vida de Argentina, Uruguay, Chile y Paraguay en 2019.",
           [_comp(["ARG", "URY", "CHL", "PRY"], "esperanza_vida_anios", 2019)]),
        _q("T16", c, "¿Quién tuvo más inflación en 2021: Brasil, Colombia o Perú?",
           [_comp(["BRA", "COL", "PER"], "inflacion_anual_pct", 2021)]),
        _q("T17", c, "Compara el PIB per cápita de México y Costa Rica en 2023.",
           [_comp(["MEX", "CRI"], "pib_per_capita_usd", 2023)]),
        _q("T18", c, "Entre Ecuador, Bolivia y Paraguay, ¿cuál tuvo mayor desempleo en 2023?",
           [_comp(["ECU", "BOL", "PRY"], "desempleo_pct", 2023)]),
        _q("T19", c, "Compara el uso de internet en 2021 en Argentina, Brasil, México y Colombia.",
           [_comp(["ARG", "BRA", "MEX", "COL"], "usuarios_internet_pct", 2021)]),
        _q("T20", c, "Crecimiento del PIB en 2023: Estados Unidos frente a México y Brasil.",
           [_comp(["USA", "MEX", "BRA"], "crecimiento_pib_pct", 2023)]),
        _q("T21", c, "¿Qué país tenía más energía renovable en 2020, Ecuador o Colombia?",
           [_comp(["ECU", "COL"], "energias_renovables_pct", 2020)]),
        _q("T22", c, "Compara la desigualdad (Gini) de Chile, México y Ecuador en 2020.",
           [_comp(["CHL", "MEX", "ECU"], "indice_gini", 2020)]),
        _q("T23", c, "Ranking de PIB per cápita 2021 para Uruguay, Chile, Argentina, Perú y Bolivia.",
           [_comp(["URY", "CHL", "ARG", "PER", "BOL"], "pib_per_capita_usd", 2021)]),
        _q("T24", c, "Compara el desempleo de Estados Unidos y Chile en 2020.",
           [_comp(["USA", "CHL"], "desempleo_pct", 2020)]),
        # --- cruce_clima_economia (10) ---
        _q("T25", x, "Para México en 2022, dame la inflación y la precipitación total de Ciudad de México en ese año.",
           [_bm("MEX", "inflacion_anual_pct", 2022), _anio_clima("Ciudad de México", "precipitacion_mm", 2022)],
           [[_bm("MEX", "inflacion_anual_pct", 2022), _resumen("Ciudad de México", 2022)]]),
        _q("T26", x, "¿Cuál fue el desempleo de Argentina en 2021 y la temperatura media de Buenos Aires ese año?",
           [_bm("ARG", "desempleo_pct", 2021), _anio_clima("Buenos Aires", "temp_media_c", 2021)],
           [[_bm("ARG", "desempleo_pct", 2021), _resumen("Buenos Aires", 2021)]]),
        _q("T27", x, "Colombia 2023: crecimiento del PIB y resumen climático de Bogotá.",
           [_bm("COL", "crecimiento_pib_pct", 2023), _resumen("Bogotá", 2023)]),
        _q("T28", x, "Ecuador en 2020: esperanza de vida y lluvia acumulada en Guayaquil.",
           [_bm("ECU", "esperanza_vida_anios", 2020), _anio_clima("Guayaquil", "precipitacion_mm", 2020)],
           [[_bm("ECU", "esperanza_vida_anios", 2020), _resumen("Guayaquil", 2020)]]),
        _q("T29", x, "Para Perú en 2024, ¿cuál fue la inflación y la temperatura máxima de Lima?",
           [_bm("PER", "inflacion_anual_pct", 2024), _anio_clima("Lima", "temp_max_c", 2024)],
           [[_bm("PER", "inflacion_anual_pct", 2024), _resumen("Lima", 2024)]]),
        _q("T30", x, "Uruguay 2021: porcentaje de energía renovable y radiación solar media en Montevideo.",
           [_bm("URY", "energias_renovables_pct", 2021), _anio_clima("Montevideo", "radiacion_solar_mj_m2", 2021)],
           [[_bm("URY", "energias_renovables_pct", 2021), _resumen("Montevideo", 2021)]]),
        _q("T31", x, "Costa Rica en 2022: PIB per cápita y resumen climático de San José.",
           [_bm("CRI", "pib_per_capita_usd", 2022), _resumen("San José", 2022)]),
        _q("T32", x, "¿Cuánto creció la economía de Chile en 2022 y cuánto llovió en Santiago ese año?",
           [_bm("CHL", "crecimiento_pib_pct", 2022), _anio_clima("Santiago", "precipitacion_mm", 2022)],
           [[_bm("CHL", "crecimiento_pib_pct", 2022), _resumen("Santiago", 2022)]]),
        _q("T33", x, "Bolivia 2023: usuarios de internet y temperatura mínima en La Paz.",
           [_bm("BOL", "usuarios_internet_pct", 2023), _anio_clima("La Paz", "temp_min_c", 2023)],
           [[_bm("BOL", "usuarios_internet_pct", 2023), _resumen("La Paz", 2023)]]),
        _q("T34", x, "Brasil en 2020: desempleo y temperatura media de São Paulo.",
           [_bm("BRA", "desempleo_pct", 2020), _anio_clima("São Paulo", "temp_media_c", 2020)],
           [[_bm("BRA", "desempleo_pct", 2020), _resumen("São Paulo", 2020)]]),
        # --- calculo_variacion (10) ---
        _q("T35", k, "¿En qué porcentaje cambió el PIB per cápita de Perú entre 2012 y 2022?",
           [_bm("PER", "pib_per_capita_usd", 2012, 2022), _CALC]),
        _q("T36", k, "Calcula la variación porcentual de la esperanza de vida de Bolivia de 2000 a 2020.",
           [_bm("BOL", "esperanza_vida_anios", 2000, 2020), _CALC]),
        _q("T37", k, "¿Cuánto aumentó, en porcentaje, el uso de internet en Ecuador entre 2010 y 2020?",
           [_bm("ECU", "usuarios_internet_pct", 2010, 2020), _CALC]),
        _q("T38", k, "Variación porcentual del PIB per cápita de Argentina entre 2017 y 2023.",
           [_bm("ARG", "pib_per_capita_usd", 2017, 2023), _CALC]),
        _q("T39", k, "¿Cuál fue el cambio porcentual del desempleo en Colombia entre 2019 y 2021?",
           [_bm("COL", "desempleo_pct", 2019, 2021), _CALC]),
        _q("T40", k, "Calcula cuánto varió porcentualmente el PIB per cápita de Paraguay de 2005 a 2015.",
           [_bm("PRY", "pib_per_capita_usd", 2005, 2015), _CALC]),
        _q("T41", k, "¿Qué tanto creció en porcentaje el PIB per cápita de Brasil entre 2020 y 2022?",
           [_bm("BRA", "pib_per_capita_usd", 2020, 2022), _CALC]),
        _q("T42", k, "Variación porcentual de la energía renovable en Chile entre 2000 y 2020.",
           [_bm("CHL", "energias_renovables_pct", 2000, 2020), _CALC]),
        _q("T43", k, "¿Cuánto cambió porcentualmente la esperanza de vida en Perú de 1990 a 2020?",
           [_bm("PER", "esperanza_vida_anios", 1990, 2020), _CALC]),
        _q("T44", k, "Calcula la variación porcentual de la inflación de Estados Unidos entre 2020 y 2022.",
           [_bm("USA", "inflacion_anual_pct", 2020, 2022), _CALC]),
        # --- fuera_dominio (16): temas, lugares o periodos que las herramientas no cubren ---
        _q("T45", o, "¿Cuál era la población total de Ecuador en 2022?", []),
        _q("T46", o, "¿Qué tasa de pobreza tuvo Bolivia en 2021?", []),
        _q("T47", o, "Temperatura media de Cuenca (Ecuador) en 2023.", []),
        _q("T48", o, "PIB per cápita de Venezuela en 2022.", []),
        _q("T49", o, "¿Qué temperatura hará mañana en Lima?", []),
        _q("T50", o, "¿Cuál fue la inflación de Perú en 1985?", []),
        _q("T51", o, "Proyección del desempleo de Colombia para 2027.", []),
        _q("T52", o, "¿Cuánto llovió en Guayaquil en 2010?", []),
        _q("T53", o, "Compara la deuda pública de Argentina y Brasil en 2022.", []),
        _q("T54", o, "Nivel de PM2.5 (calidad del aire) en Santiago en 2023.", []),
        _q("T55", o, "¿Quién ganó las elecciones presidenciales de Ecuador en 2023?", []),
        _q("T56", o, "Humedad relativa promedio de Bogotá en 2022.", []),
        _q("T57", o, "¿Cuántos turistas visitaron Costa Rica en 2019?", []),
        _q("T58", o, "Precio promedio del petróleo WTI en 2023.", []),
        _q("T59", o, "Temperatura media de Toronto en 2021.", []),
        _q("T60", o, "Escribe un poema sobre la lluvia de Quito.", []),
    ]
