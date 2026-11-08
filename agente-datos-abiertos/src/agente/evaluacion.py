"""Conjuntos de validación (44) y prueba (60), métricas separadas del LLM y del sistema, e IC 95 %."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from . import config, herramientas, motor_agente

ETIQUETAS_ENFOQUES: dict[str, str] = {
    "zeroshot_sin_tools": "1 · LLM sin herramientas (Zero-Shot qwen2.5:7b)",
    "enrutador_reglas": "2 · Enrutador léxico por reglas (Baseline)",
    "react_texto": "3 · Agente ReAct en texto (qwen2.5:7b)",
    "tool_calling_nativo": "4 · Agente tool-calling (bucle LLM + Pydantic + anclaje)",
}

ETIQUETAS_CATEGORIAS: dict[str, str] = {
    "simple_1hop": "1-Hop Simple (BM / Clima)",
    "comparativa_paises": "Comparativa Multi-País",
    "cruce_clima_economia": "Cruce Multi-API (Clima + Economía)",
    "calculo_variacion": "Cálculo + Calculadora Determinista",
    "fuera_dominio": "Fuera de Dominio / Abstención",
}


def _especificaciones_base_preguntas() -> list[dict[str, Any]]:
    """Define las 44 preguntas del benchmark en 5 niveles de complejidad."""
    return [
        # -------------------------------------------------------------------
        # Categoría 1: simple_1hop (12 preguntas: 6 Banco Mundial + 6 Clima)
        # -------------------------------------------------------------------
        {
            "id_pregunta": "Q01",
            "categoria": "simple_1hop",
            "pregunta": "¿Cuál fue el PIB per cápita en USD de Ecuador (ECU) entre 2020 y 2023 según el Banco Mundial?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "ECU",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2020,
                        "anio_fin": 2023,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q02",
            "categoria": "simple_1hop",
            "pregunta": "Consulta la inflación anual al consumidor de Colombia (COL) desde 2019 hasta 2023.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "COL",
                        "indicador": "inflacion_anual_pct",
                        "anio_inicio": 2019,
                        "anio_fin": 2023,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q03",
            "categoria": "simple_1hop",
            "pregunta": "¿Qué porcentaje de la población de Chile (CHL) utilizó internet entre 2018 y 2022?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "CHL",
                        "indicador": "usuarios_internet_pct",
                        "anio_inicio": 2018,
                        "anio_fin": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q04",
            "categoria": "simple_1hop",
            "pregunta": "¿Cuál fue la tasa de desempleo total en Perú (PER) entre 2020 y 2023?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "PER",
                        "indicador": "desempleo_pct",
                        "anio_inicio": 2020,
                        "anio_fin": 2023,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q05",
            "categoria": "simple_1hop",
            "pregunta": "Muestra la evolución de la esperanza de vida al nacer en Costa Rica (CRI) entre 2018 y 2022.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "CRI",
                        "indicador": "esperanza_vida_anios",
                        "anio_inicio": 2018,
                        "anio_fin": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q06",
            "categoria": "simple_1hop",
            "pregunta": "¿Qué porcentaje de energía renovable sobre el consumo final tuvo Uruguay (URY) entre 2017 y 2021?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "URY",
                        "indicador": "energias_renovables_pct",
                        "anio_inicio": 2017,
                        "anio_fin": 2021,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q07",
            "categoria": "simple_1hop",
            "pregunta": "Dame el resumen climático anual completo de Quito en el año 2023.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "resumen_climatico_anual_ciudad",
                    "argumentos": {"ciudad": "Quito", "anio": 2023},
                }
            ],
        },
        {
            "id_pregunta": "Q08",
            "categoria": "simple_1hop",
            "pregunta": "¿Cuál fue el resumen climático anual de todas las métricas climáticas en Buenos Aires durante 2022?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "resumen_climatico_anual_ciudad",
                    "argumentos": {"ciudad": "Buenos Aires", "anio": 2022},
                }
            ],
        },
        {
            "id_pregunta": "Q09",
            "categoria": "simple_1hop",
            "pregunta": "¿Cuánta precipitación acumulada en milímetros se registró en Guayaquil entre 2023-01-01 y 2023-12-31?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Guayaquil",
                        "fecha_inicio": "2023-01-01",
                        "fecha_fin": "2023-12-31",
                        "variable": "precipitacion_mm",
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q10",
            "categoria": "simple_1hop",
            "pregunta": "Consulta la temperatura máxima extrema registrada en Santiago entre 2024-01-01 y 2024-12-31.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Santiago",
                        "fecha_inicio": "2024-01-01",
                        "fecha_fin": "2024-12-31",
                        "variable": "temp_max_c",
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q11",
            "categoria": "simple_1hop",
            "pregunta": "¿Cuál fue la radiación solar media en La Paz entre 2022-01-01 y 2022-12-31?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "La Paz",
                        "fecha_inicio": "2022-01-01",
                        "fecha_fin": "2022-12-31",
                        "variable": "radiacion_solar_mj_m2",
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q12",
            "categoria": "simple_1hop",
            "pregunta": "Obtén el perfil climático anual y días de lluvia de Ciudad de México en 2024.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "resumen_climatico_anual_ciudad",
                    "argumentos": {"ciudad": "Ciudad de México", "anio": 2024},
                }
            ],
        },
        # -------------------------------------------------------------------
        # Categoría 2: comparativa_paises (8 preguntas)
        # -------------------------------------------------------------------
        {
            "id_pregunta": "Q13",
            "categoria": "comparativa_paises",
            "pregunta": "Compara el PIB per cápita en USD entre Ecuador (ECU), Colombia (COL), Perú (PER) y Chile (CHL) en el año 2022.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["ECU", "COL", "PER", "CHL"],
                        "indicador": "pib_per_capita_usd",
                        "anio": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q14",
            "categoria": "comparativa_paises",
            "pregunta": "Compara la inflación anual al consumidor en 2023 entre Argentina (ARG), Brasil (BRA), México (MEX) y Estados Unidos (USA).",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["ARG", "BRA", "MEX", "USA"],
                        "indicador": "inflacion_anual_pct",
                        "anio": 2023,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q15",
            "categoria": "comparativa_paises",
            "pregunta": "¿Cómo se comparan Uruguay (URY), Costa Rica (CRI), Brasil (BRA) y Paraguay (PRY) en consumo de energía renovable en 2021?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["URY", "CRI", "BRA", "PRY"],
                        "indicador": "energias_renovables_pct",
                        "anio": 2021,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q16",
            "categoria": "comparativa_paises",
            "pregunta": "Compara el índice de Gini de desigualdad de ingresos entre Colombia (COL), Brasil (BRA), Perú (PER) y Uruguay (URY) en 2022.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["COL", "BRA", "PER", "URY"],
                        "indicador": "indice_gini",
                        "anio": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q17",
            "categoria": "comparativa_paises",
            "pregunta": "Compara la tasa de desempleo en 2022 entre Colombia (COL), Chile (CHL), México (MEX) y Bolivia (BOL).",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["COL", "CHL", "MEX", "BOL"],
                        "indicador": "desempleo_pct",
                        "anio": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q18",
            "categoria": "comparativa_paises",
            "pregunta": "¿Qué país tuvo mayor crecimiento del PIB real en 2022 al comparar Colombia (COL), Argentina (ARG), Ecuador (ECU) y Chile (CHL)?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["COL", "ARG", "ECU", "CHL"],
                        "indicador": "crecimiento_pib_pct",
                        "anio": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q19",
            "categoria": "comparativa_paises",
            "pregunta": "Compara el porcentaje de usuarios de internet en 2022 entre Chile (CHL), Uruguay (URY), Ecuador (ECU) y Bolivia (BOL).",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["CHL", "URY", "ECU", "BOL"],
                        "indicador": "usuarios_internet_pct",
                        "anio": 2022,
                    },
                }
            ],
        },
        {
            "id_pregunta": "Q20",
            "categoria": "comparativa_paises",
            "pregunta": "Compara la esperanza de vida al nacer en 2021 entre Costa Rica (CRI), Chile (CHL), Perú (PER) y Bolivia (BOL).",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "comparar_paises_indicador",
                    "argumentos": {
                        "codigos_paises": ["CRI", "CHL", "PER", "BOL"],
                        "indicador": "esperanza_vida_anios",
                        "anio": 2021,
                    },
                }
            ],
        },
        # -------------------------------------------------------------------
        # Categoría 3: cruce_clima_economia (8 preguntas multi-API)
        # -------------------------------------------------------------------
        {
            "id_pregunta": "Q21",
            "categoria": "cruce_clima_economia",
            "pregunta": "Cruza datos económicos y climáticos de Ecuador (ECU) en 2023: ¿cuál fue su crecimiento del PIB real y cuál fue la precipitación acumulada en Quito entre 2023-01-01 y 2023-12-31?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "ECU",
                        "indicador": "crecimiento_pib_pct",
                        "anio_inicio": 2023,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Quito",
                        "fecha_inicio": "2023-01-01",
                        "fecha_fin": "2023-12-31",
                        "variable": "precipitacion_mm",
                    },
                },
            ],
        },
        {
            "id_pregunta": "Q22",
            "categoria": "cruce_clima_economia",
            "pregunta": "Para Colombia (COL) en 2022, consulta la inflación anual en el Banco Mundial y el resumen climático anual completo de Bogotá en 2022.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "COL",
                        "indicador": "inflacion_anual_pct",
                        "anio_inicio": 2022,
                        "anio_fin": 2022,
                    },
                },
                {
                    "herramienta": "resumen_climatico_anual_ciudad",
                    "argumentos": {"ciudad": "Bogotá", "anio": 2022},
                },
            ],
        },
        {
            "id_pregunta": "Q23",
            "categoria": "cruce_clima_economia",
            "pregunta": "En Chile (CHL) durante 2021, ¿qué porcentaje de energía renovable tuvo el país y cuál fue la radiación solar media en Santiago entre 2021-01-01 y 2021-12-31?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "CHL",
                        "indicador": "energias_renovables_pct",
                        "anio_inicio": 2021,
                        "anio_fin": 2021,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Santiago",
                        "fecha_inicio": "2021-01-01",
                        "fecha_fin": "2021-12-31",
                        "variable": "radiacion_solar_mj_m2",
                    },
                },
            ],
        },
        {
            "id_pregunta": "Q24",
            "categoria": "cruce_clima_economia",
            "pregunta": "Consulta el PIB per cápita de Perú (PER) en 2023 y la temperatura media histórica de Lima entre 2023-01-01 y 2023-12-31.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "PER",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2023,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Lima",
                        "fecha_inicio": "2023-01-01",
                        "fecha_fin": "2023-12-31",
                        "variable": "temp_media_c",
                    },
                },
            ],
        },
        {
            # Antes pedía 2022, año sin dato de EG.FEC.RNEW.ZS para Brasil en WDI (verdad terreno vacía).
            "id_pregunta": "Q25",
            "categoria": "cruce_clima_economia",
            "pregunta": "Para Brasil (BRA) en 2021, obtén el porcentaje de energías renovables del Banco Mundial y el resumen climático anual de São Paulo en 2021.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "BRA",
                        "indicador": "energias_renovables_pct",
                        "anio_inicio": 2021,
                        "anio_fin": 2021,
                    },
                },
                {
                    "herramienta": "resumen_climatico_anual_ciudad",
                    "argumentos": {"ciudad": "São Paulo", "anio": 2021},
                },
            ],
        },
        {
            "id_pregunta": "Q26",
            "categoria": "cruce_clima_economia",
            "pregunta": "En Uruguay (URY) en 2023, ¿cuál fue el PIB per cápita nacional y cuál fue la velocidad máxima del viento en Montevideo entre 2023-01-01 y 2023-12-31?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "URY",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2023,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Montevideo",
                        "fecha_inicio": "2023-01-01",
                        "fecha_fin": "2023-12-31",
                        "variable": "viento_max_kmh",
                    },
                },
            ],
        },
        {
            "id_pregunta": "Q27",
            "categoria": "cruce_clima_economia",
            "pregunta": "Para Bolivia (BOL) en 2021, consulta la esperanza de vida al nacer y la temperatura mínima extrema registrada en La Paz entre 2021-01-01 y 2021-12-31.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "BOL",
                        "indicador": "esperanza_vida_anios",
                        "anio_inicio": 2021,
                        "anio_fin": 2021,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "La Paz",
                        "fecha_inicio": "2021-01-01",
                        "fecha_fin": "2021-12-31",
                        "variable": "temp_min_c",
                    },
                },
            ],
        },
        {
            "id_pregunta": "Q28",
            "categoria": "cruce_clima_economia",
            "pregunta": "En Paraguay (PRY) durante 2023, consulta el crecimiento del PIB real y la temperatura máxima extrema de Asunción entre 2023-01-01 y 2023-12-31.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "PRY",
                        "indicador": "crecimiento_pib_pct",
                        "anio_inicio": 2023,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "consultar_clima_historico_ciudad",
                    "argumentos": {
                        "ciudad": "Asunción",
                        "fecha_inicio": "2023-01-01",
                        "fecha_fin": "2023-12-31",
                        "variable": "temp_max_c",
                    },
                },
            ],
        },
        # -------------------------------------------------------------------
        # Categoría 4: calculo_variacion (8 preguntas con calculadora determinista)
        # -------------------------------------------------------------------
        {
            "id_pregunta": "Q29",
            "categoria": "calculo_variacion",
            "pregunta": "Consulta el PIB per cápita de Ecuador (ECU) entre 2015 y 2023 y usa la calculadora para determinar la variación porcentual exacta en ese periodo.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "ECU",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2015,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q30",
            "categoria": "calculo_variacion",
            "pregunta": "¿Cuánto varió porcentualmente el PIB per cápita de Chile (CHL) entre 2010 y 2023 según el Banco Mundial y la herramienta calculadora?",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "CHL",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2010,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q31",
            "categoria": "calculo_variacion",
            "pregunta": "Consulta el porcentaje de usuarios de internet en Perú (PER) entre 2015 y 2022 y calcula la tasa de variación porcentual acumulada.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "PER",
                        "indicador": "usuarios_internet_pct",
                        "anio_inicio": 2015,
                        "anio_fin": 2022,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q32",
            "categoria": "calculo_variacion",
            "pregunta": "Obtén el PIB per cápita de Costa Rica (CRI) desde 2015 hasta 2023 y calcula su cambio porcentual con la calculadora determinista.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "CRI",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2015,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q33",
            "categoria": "calculo_variacion",
            "pregunta": "Consulta el acceso a internet en Bolivia (BOL) entre 2014 y 2022 y calcula la variación porcentual exacta.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "BOL",
                        "indicador": "usuarios_internet_pct",
                        "anio_inicio": 2014,
                        "anio_fin": 2022,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q34",
            "categoria": "calculo_variacion",
            "pregunta": "En México (MEX), consulta el PIB per cápita entre 2018 y 2023 y calcula la variación porcentual exacta.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "MEX",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2018,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q35",
            "categoria": "calculo_variacion",
            "pregunta": "Consulta el PIB per cápita de Uruguay (URY) entre 2016 y 2023 y determina el crecimiento porcentual acumulado con la calculadora.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "URY",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2016,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        {
            "id_pregunta": "Q36",
            "categoria": "calculo_variacion",
            "pregunta": "Consulta el PIB per cápita de Estados Unidos (USA) entre 2019 y 2023 y calcula su variación porcentual exacta.",
            "debe_abstenerse": False,
            "llamadas_esperadas": [
                {
                    "herramienta": "consultar_indicador_banco_mundial",
                    "argumentos": {
                        "codigo_pais": "USA",
                        "indicador": "pib_per_capita_usd",
                        "anio_inicio": 2019,
                        "anio_fin": 2023,
                    },
                },
                {
                    "herramienta": "calcular_correlacion_o_cambio",
                    "argumentos": {"operacion": "variacion_porcentual"},
                },
            ],
        },
        # -------------------------------------------------------------------
        # Categoría 5: fuera_dominio (8 preguntas trampa / abstención obligatoria)
        # -------------------------------------------------------------------
        {
            "id_pregunta": "Q37",
            "categoria": "fuera_dominio",
            "pregunta": "¿Cuál es el precio actual de Bitcoin (BTC) y de las acciones de Apple en el NASDAQ hoy?",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q38",
            "categoria": "fuera_dominio",
            "pregunta": "¿Cuáles son las reservas probadas de litio marciano en el cráter Jezero de Marte en 2023?",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q39",
            "categoria": "fuera_dominio",
            "pregunta": "Consulta en el Banco Mundial el PIB per cápita de España y la temperatura media de Madrid en 2023.",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q40",
            "categoria": "fuera_dominio",
            "pregunta": "¿Cuál será la inflación anual proyectada de Ecuador (ECU) y la lluvia de Quito en el año 2029?",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q41",
            "categoria": "fuera_dominio",
            "pregunta": "¿Cuál fue la precipitación acumulada en milímetros en Quito entre 1995-01-01 y 1995-12-31 según Open-Meteo?",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q42",
            "categoria": "fuera_dominio",
            "pregunta": "Compara el número de camas de hospital por habitante entre Japón y Alemania en 2022.",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q43",
            "categoria": "fuera_dominio",
            "pregunta": "¿Cuál es el tipo de cambio en vivo del dólar blue en Buenos Aires en este minuto?",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
        {
            "id_pregunta": "Q44",
            "categoria": "fuera_dominio",
            "pregunta": "Consulta el resumen climático anual de Tokio en 2023 y las exportaciones de soja de Francia.",
            "debe_abstenerse": True,
            "llamadas_esperadas": [],
        },
    ]


def _resolver_llamadas(llamadas: list[dict[str, Any]], id_pregunta: str) -> tuple[list[dict[str, Any]], list[float]]:
    """Ejecuta la secuencia esperada sobre la instantánea: devuelve argumentos canónicos y cifras de referencia."""
    cifras_ref: list[float] = []
    completas: list[dict[str, Any]] = []
    ultimo_resultado_bm: dict[str, Any] | None = None
    for ll in llamadas:
        nom = ll["herramienta"]
        args = dict(ll["argumentos"])
        if nom == "calcular_correlacion_o_cambio" and "valores_x" not in args and ultimo_resultado_bm:
            serie = list(ultimo_resultado_bm.get("serie_anual", {}).values())
            args["valores_x"] = [float(serie[0]), float(serie[-1])]
        ejec = herramientas.ejecutar_herramienta_segura(nom, args)
        if not ejec["exito"]:
            raise RuntimeError(f"Error al generar verdad terreno en {id_pregunta} ({nom}): {ejec['mensaje_error']}")
        res = ejec["resultado"]
        completas.append({"herramienta": nom, "argumentos": ejec["argumentos_validados"]})
        if nom == "consultar_indicador_banco_mundial":
            ultimo_resultado_bm = res
            serie_vals = list(res.get("serie_anual", {}).values())
            if not serie_vals:
                raise RuntimeError(f"{id_pregunta}: serie vacía en la instantánea para {args}")
            cifras_ref.append(round(float(serie_vals[0]), 2))
            if len(serie_vals) > 1:
                cifras_ref.append(round(float(serie_vals[-1]), 2))
        elif nom == "comparar_paises_indicador":
            rk = res.get("ranking", [])
            if len(rk) < len(args["codigos_paises"]):
                raise RuntimeError(f"{id_pregunta}: faltan países en la comparación {args}")
            cifras_ref.append(round(float(rk[0]["valor"]), 2))
            cifras_ref.append(round(float(rk[-1]["valor"]), 2))
        elif nom == "consultar_clima_historico_ciudad":
            cifras_ref.append(round(float(res["valor_periodo"]), 2))
        elif nom == "resumen_climatico_anual_ciudad":
            m = res["metricas_anuales"]
            cifras_ref.append(round(float(m["temp_media_c"]), 2))
            cifras_ref.append(round(float(m["precipitacion_mm"]), 2))
        elif nom == "calcular_correlacion_o_cambio" and res["operacion"] == "variacion_porcentual":
            cifras_ref.append(round(float(res["variacion_porcentual"]), 2))
    return completas, list(dict.fromkeys(cifras_ref))


def especificaciones_validacion() -> list[dict[str, Any]]:
    """Las 44 preguntas originales pasan a ser el conjunto de VALIDACIÓN (V01-V44)."""
    salida = []
    for item in _especificaciones_base_preguntas():
        salida.append({**item, "id_pregunta": item["id_pregunta"].replace("Q", "V"), "alternativas": []})
    return salida


def construir_conjunto(nombre: str = "validacion", guardar: bool = True) -> list[dict[str, Any]]:
    """Construye un conjunto ("validacion" o "prueba") con verdad terreno calculada sobre la instantánea."""
    from . import conjuntos

    if nombre == "validacion":
        base, ruta = especificaciones_validacion(), config.CONJUNTO_VALIDACION
    elif nombre == "prueba":
        base, ruta = conjuntos.especificaciones_prueba(), config.CONJUNTO_PRUEBA
    else:
        raise ValueError(f"Conjunto '{nombre}' desconocido.")

    modo_previo = config.MODO_DATOS
    config.MODO_DATOS = "instantanea"
    try:
        conjunto: list[dict[str, Any]] = []
        for item in base:
            llamadas, cifras_ref = _resolver_llamadas(item["llamadas_esperadas"], item["id_pregunta"])
            alternativas = [_resolver_llamadas(alt, item["id_pregunta"])[0] for alt in item.get("alternativas", [])]
            conjunto.append({
                "id_pregunta": item["id_pregunta"],
                "conjunto": nombre,
                "categoria": item["categoria"],
                "pregunta": item["pregunta"],
                "debe_abstenerse": item["debe_abstenerse"],
                "llamadas_esperadas": llamadas,
                "alternativas": alternativas,
                "herramientas_esperadas": [c["herramienta"] for c in llamadas],
                "cifras_referencia": cifras_ref,
            })
    finally:
        config.MODO_DATOS = modo_previo

    if guardar:
        ruta.write_text(json.dumps(conjunto, ensure_ascii=False, indent=2), encoding="utf-8")
    return conjunto


def cargar_conjunto(nombre: str) -> list[dict[str, Any]]:
    ruta = config.CONJUNTO_VALIDACION if nombre == "validacion" else config.CONJUNTO_PRUEBA
    if not ruta.exists():
        return construir_conjunto(nombre, guardar=True)
    return json.loads(ruta.read_text(encoding="utf-8"))


def _interpretaciones_numero(token: str) -> list[float]:
    """Valores posibles de un número escrito con separadores de miles o decimales (es/en)."""
    token = token.strip(".,")
    if not token:
        return []
    if "." in token and "," in token:
        decimal = "." if token.rfind(".") > token.rfind(",") else ","
        miles = "," if decimal == "." else "."
        return [float(token.replace(miles, "").replace(decimal, "."))]
    sep = "." if "." in token else ("," if "," in token else "")
    if not sep:
        return [float(token)]
    partes = token.split(sep)
    if len(partes) > 2:  # 1.234.567 o 1,234,567
        return [float(token.replace(sep, ""))]
    candidatos = [float(token.replace(sep, "."))]
    if len(partes[1]) == 3:  # ambiguo: 6.718 puede ser 6718 o 6,718
        candidatos.append(float(token.replace(sep, "")))
    return candidatos


def tokens_numericos(texto: str) -> list[list[float]]:
    """Cada número del texto con sus interpretaciones posibles."""
    if not texto:
        return []
    salida = []
    for m in re.finditer(r"(?<![\w])-?\d+(?:[.,]\d+)*", texto):
        try:
            valores = _interpretaciones_numero(m.group(0).lstrip("-"))
        except ValueError:
            continue
        if m.group(0).startswith("-"):
            valores = [-v for v in valores]
        salida.append(valores)
    return salida


def extraer_numeros_de_texto(texto: str) -> list[float]:
    """Todas las cifras del texto (con todas sus interpretaciones de separadores)."""
    return [v for valores in tokens_numericos(texto) for v in valores]


def _coincide(valor: float, ref: float, tolerancia_rel: float = config.TOLERANCIA_ERROR_RELATIVO) -> bool:
    return abs(valor - ref) <= max(abs(ref) * tolerancia_rel, 0.05)


def calcular_fidelidad_numerica(
    texto_respuesta: str,
    cifras_referencia: list[float],
    tolerancia_rel: float = config.TOLERANCIA_ERROR_RELATIVO,
) -> dict[str, float]:
    """Proporción de cifras de referencia que aparecen en el texto con error relativo <= 1 %."""
    if not cifras_referencia:
        return {"fidelidad_numerica": 1.0, "error_relativo_medio_pct": 0.0, "cifras_verificadas": 0, "cifras_totales": 0}
    nums_texto = extraer_numeros_de_texto(texto_respuesta)
    if not nums_texto:
        return {
            "fidelidad_numerica": 0.0,
            "error_relativo_medio_pct": 100.0,
            "cifras_verificadas": 0,
            "cifras_totales": len(cifras_referencia),
        }
    aciertos = 0
    errores_rel_pct: list[float] = []
    for ref in cifras_referencia:
        if any(_coincide(n, ref, tolerancia_rel) for n in nums_texto):
            aciertos += 1
        min_err = min(abs(n - ref) / max(abs(ref), 1e-6) for n in nums_texto)
        errores_rel_pct.append(min(min_err * 100.0, 100.0))
    return {
        "fidelidad_numerica": round(aciertos / len(cifras_referencia), 4),
        "error_relativo_medio_pct": round(float(np.mean(errores_rel_pct)), 2),
        "cifras_verificadas": aciertos,
        "cifras_totales": len(cifras_referencia),
    }


def cifras_sin_respaldo(texto: str, respaldo: list[float]) -> int:
    """Números de un texto (sin contar años) que no salen de ninguna observación ni de la pregunta."""
    n = 0
    for valores in tokens_numericos(texto):
        if all(float(v).is_integer() and 1900 <= v <= 2100 for v in valores):
            continue  # años y fechas
        if not any(_coincide(v, r, 0.01) for v in valores for r in respaldo):
            n += 1
    return n


def _normalizar_llamada(nombre: str, args: dict[str, Any]) -> dict[str, Any]:
    """Pasa los argumentos por el esquema Pydantic (solo normaliza alias y formatos) para compararlos."""
    meta = herramientas.REGISTRO_HERRAMIENTAS.get(nombre)
    normal = dict(args or {})
    if meta:
        try:
            normal = meta["modelo_pydantic"].model_validate(args or {}).model_dump()
        except Exception:
            pass
    if nombre == "calcular_correlacion_o_cambio":
        xs = normal.get("valores_x") or []
        if normal.get("operacion") in ("variacion_porcentual", "diferencia") and len(xs) >= 2:
            normal["valores_x"] = [xs[0], xs[-1]]
        normal.pop("valores_y", None) if not normal.get("valores_y") else None
    return normal


def _comparar_argumentos_llamada(args_pred: dict[str, Any], args_esp: dict[str, Any]) -> float:
    """Proporción de argumentos esperados que coinciden (listas como conjuntos; números con tolerancia)."""
    claves = {k: v for k, v in (args_esp or {}).items() if v is not None}
    if not claves:
        return 1.0
    if not args_pred:
        return 0.0
    aciertos = 0
    for k, v_esp in claves.items():
        v_pred = args_pred.get(k)
        if v_pred is None:
            continue
        if isinstance(v_esp, list) and isinstance(v_pred, list):
            if all(isinstance(a, (int, float)) for a in v_esp):
                if len(v_esp) == len(v_pred) and all(
                    isinstance(b, (int, float)) and _coincide(float(b), float(a), 0.001) for a, b in zip(v_esp, v_pred)
                ):
                    aciertos += 1
            elif {str(x).upper() for x in v_esp} == {str(x).upper() for x in v_pred}:
                aciertos += 1
        elif isinstance(v_esp, (int, float)) and isinstance(v_pred, (int, float)):
            if abs(float(v_esp) - float(v_pred)) <= 0.05:
                aciertos += 1
        elif str(v_esp).strip().lower() == str(v_pred).strip().lower():
            aciertos += 1
    return round(aciertos / len(claves), 4)


def puntuar_llamadas(
    llamadas_pred: list[dict[str, Any]],
    item: dict[str, Any],
) -> dict[str, float]:
    """Selección exacta (conjunto de herramientas), F1 y exactitud de argumentos frente a la mejor
    secuencia esperada (la principal o una alternativa válida)."""
    pred = [(c["herramienta"], _normalizar_llamada(c["herramienta"], c.get("argumentos", {}))) for c in llamadas_pred]
    mejor = {"seleccion_exacta": 0.0, "f1": 0.0, "exactitud_args": 0.0}
    for esperadas in [item["llamadas_esperadas"], *item.get("alternativas", [])]:
        set_esp = {c["herramienta"] for c in esperadas}
        set_pred = {n for n, _ in pred}
        inter = len(set_esp & set_pred)
        prec = inter / len(set_pred) if set_pred else 0.0
        rec = inter / len(set_esp) if set_esp else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        scores = []
        for ll in esperadas:
            esp = _normalizar_llamada(ll["herramienta"], ll["argumentos"])
            candidatas = [a for n, a in pred if n == ll["herramienta"]]
            scores.append(max((_comparar_argumentos_llamada(a, esp) for a in candidatas), default=0.0))
        actual = {
            "seleccion_exacta": 1.0 if set_esp == set_pred else 0.0,
            "f1": round(f1, 4),
            "exactitud_args": round(float(np.mean(scores)), 4) if scores else 0.0,
        }
        if (actual["seleccion_exacta"], actual["exactitud_args"], actual["f1"]) > (
            mejor["seleccion_exacta"], mejor["exactitud_args"], mejor["f1"]
        ):
            mejor = actual
    return mejor


def _llamadas_propuestas_por_llm(salida: dict[str, Any], solo_turno_1: bool = False) -> list[dict[str, Any]] | None:
    """Lo que propuso el LLM ANTES de cualquier intervención del código (None si el enfoque no usa LLM para elegir)."""
    if salida["enfoque"] == "tool_calling_nativo":
        pasos = salida.get("pasos", [])[:1] if solo_turno_1 else salida.get("pasos", [])
        return [{"herramienta": c["herramienta"], "argumentos": c["argumentos_llm"]} for p in pasos for c in p["llamadas"]]
    if salida["enfoque"] == "react_texto":
        return list(salida.get("llamadas_herramientas", []))  # ReAct ejecuta tal cual lo que propone el LLM
    return None


def evaluar_respuesta_individual(item: dict[str, Any], salida: dict[str, Any]) -> dict[str, Any]:
    """Métricas de una respuesta. Separa lo que hizo el LLM (llm_*) de lo que entregó el sistema completo."""
    debe_abstenerse = bool(item["debe_abstenerse"])
    se_abstuvo = bool(salida.get("se_abstuvo", False))
    ejecutadas = salida.get("llamadas_herramientas", []) or []
    if salida["enfoque"] == "react_texto":
        ejecutadas = [
            {"herramienta": e["herramienta"], "argumentos": e["argumentos_validados"]}
            for e in salida.get("ejecuciones_herramientas", [])
            if e.get("exito")
        ]
    fila: dict[str, Any] = {
        "id_pregunta": item["id_pregunta"],
        "conjunto": item.get("conjunto", ""),
        "categoria": item["categoria"],
        "enfoque": salida["enfoque"],
        "version_prompt": salida.get("version_prompt", ""),
        "debe_abstenerse": debe_abstenerse,
        "se_abstuvo": se_abstuvo,
        "n_herramientas_ejecutadas": len(ejecutadas),
        "n_pasos_llm": int(salida.get("n_pasos_llm", 0 if salida["enfoque"] == "enrutador_reglas" else 1)),
        "reintentos_validacion": int(salida.get("reintentos_validacion", 0)),
        "latencia_ms": float(salida.get("latencia_ms", 0.0)),
        "tokens_prompt": int(salida.get("tokens_prompt", 0)),
        "tokens_salida": int(salida.get("tokens_salida", 0)),
    }

    propuestas = _llamadas_propuestas_por_llm(salida)
    propuestas_t1 = _llamadas_propuestas_por_llm(salida, solo_turno_1=True)

    if debe_abstenerse:
        fila.update({
            "respuesta_correcta": 1.0 if (se_abstuvo and not ejecutadas) else 0.0,
            "abstencion_correcta": 1.0 if (se_abstuvo and not ejecutadas) else 0.0,
            "ejecuto_herramienta_ood": 1.0 if ejecutadas else 0.0,
            "llm_se_abstuvo_ood": (1.0 if not propuestas else 0.0) if propuestas is not None else np.nan,
        })
        return fila

    final = puntuar_llamadas(ejecutadas, item)
    fid = calcular_fidelidad_numerica(salida.get("respuesta_final", ""), item["cifras_referencia"])
    fila.update({
        "falsa_abstencion": 1.0 if se_abstuvo else 0.0,
        "sistema_seleccion_exacta": final["seleccion_exacta"],
        "sistema_f1_herramientas": final["f1"],
        "sistema_exactitud_args": final["exactitud_args"],
        "fidelidad_respuesta_final": fid["fidelidad_numerica"],
        "error_relativo_medio_pct": fid["error_relativo_medio_pct"],
        "respuesta_correcta": 1.0 if (not se_abstuvo and fid["fidelidad_numerica"] >= 0.999) else 0.0,
    })
    if propuestas is not None:
        llm = puntuar_llamadas(propuestas, item)
        llm_t1 = puntuar_llamadas(propuestas_t1, item)
        fila.update({
            "llm_seleccion_turno1": llm_t1["seleccion_exacta"],
            "llm_seleccion_exacta": llm["seleccion_exacta"],
            "llm_f1_herramientas": llm["f1"],
            "llm_exactitud_args": llm["exactitud_args"],
            "llm_args_perfectos": 1.0 if (llm["seleccion_exacta"] == 1.0 and llm["exactitud_args"] >= 0.999) else 0.0,
        })
    if salida["enfoque"] == "tool_calling_nativo":
        texto = salida.get("texto_llm", "")
        respaldo = herramientas_cifras(salida) + extraer_numeros_de_texto(item["pregunta"])
        fid_llm = calcular_fidelidad_numerica(texto, item["cifras_referencia"])
        fila.update({
            "texto_llm_presente": 1.0 if texto else 0.0,
            "fidelidad_texto_llm": fid_llm["fidelidad_numerica"] if texto else 0.0,
            "cifras_sin_respaldo_texto_llm": cifras_sin_respaldo(texto, respaldo) if texto else 0,
        })
    if salida["enfoque"] == "zeroshot_sin_tools":
        fila["cifras_sin_respaldo_texto_llm"] = cifras_sin_respaldo(
            salida.get("respuesta_final", ""), item["cifras_referencia"] + extraer_numeros_de_texto(item["pregunta"])
        )
    return fila


def herramientas_cifras(salida: dict[str, Any]) -> list[float]:
    """Todos los números de las observaciones de herramientas de una salida."""
    from .motor_agente import _numeros_en

    return _numeros_en([e["resultado"] for e in salida.get("ejecuciones_herramientas", []) if e.get("exito")])


def ic95(valores: list[float] | np.ndarray, n_boot: int = 2000) -> tuple[float, float, str]:
    """IC 95 %. Bootstrap percentil (2000 remuestreos) salvo cuando todos los valores son iguales:
    ahí el bootstrap es degenerado («1,00-1,00») y, si la métrica es binaria, se usa Wilson."""
    arr = np.asarray([v for v in valores if not pd.isna(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), "sin datos")
    if np.all(arr == arr[0]) and set(np.unique(arr)) <= {0.0, 1.0}:
        p = float(arr.mean())
        z = 1.959964
        centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return (round(max(0.0, centro - margen), 4), round(min(1.0, centro + margen), 4), "wilson")
    rng = np.random.default_rng(config.SEMILLA)
    medias = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return (round(float(np.percentile(medias, 2.5)), 4), round(float(np.percentile(medias, 97.5)), 4), "bootstrap")


def bootstrap_ic95(valores: list[float] | np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    inf, sup, _ = ic95(valores, n_boot)
    return inf, sup


def elegir_version_prompt() -> str:
    """Elige v1 o v2 SOLO con el conjunto de validación (respuesta correcta y, de empate, selección del LLM)."""
    puntajes = {}
    for version in ("v1", "v2"):
        filas = [
            evaluar_respuesta_individual(q, motor_agente.consultar_agente(q["pregunta"], "tool_calling_nativo", version_prompt=version))
            for q in cargar_conjunto("validacion")
        ]
        df = pd.DataFrame(filas)
        puntajes[version] = (
            round(float(df["respuesta_correcta"].mean()), 4),
            round(float(df["llm_seleccion_exacta"].mean()), 4),
        )
    return max(("v2", "v1"), key=lambda v: puntajes[v])


def evaluar_traza_externa(
    pregunta: str,
    llamadas_herramientas: list[dict[str, Any]],
    respuesta_final: str,
) -> dict[str, Any]:
    """Re-ejecuta las llamadas de una traza arbitraria y verifica qué cifras de la respuesta están respaldadas."""
    ejecuciones: list[dict[str, Any]] = []
    cifras_oficiales: list[float] = []
    for ll in llamadas_herramientas:
        ej = herramientas.ejecutar_herramienta_segura(str(ll.get("herramienta", "")), ll.get("argumentos", {}) or {})
        ejecuciones.append(ej)
        if ej["exito"]:
            cifras_oficiales.extend(motor_agente._numeros_en(ej["resultado"]))
    respaldo = cifras_oficiales + extraer_numeros_de_texto(pregunta)
    tokens = [
        t for t in tokens_numericos(respuesta_final)
        if not all(float(v).is_integer() and 1900 <= v <= 2100 for v in t)
    ]
    sin_respaldo = cifras_sin_respaldo(respuesta_final, respaldo)
    respaldados = len(tokens) - sin_respaldo
    fidelidad_citas = round(respaldados / len(tokens), 4) if tokens else 1.0
    validez_esquema = round(sum(1.0 for e in ejecuciones if e["exito"]) / len(ejecuciones), 4) if ejecuciones else 1.0
    return {
        "pregunta": pregunta,
        "n_llamadas_evaluadas": len(llamadas_herramientas),
        "validez_esquema_pydantic": validez_esquema,
        "n_cifras_detectadas_respuesta": len(tokens),
        "n_cifras_respaldadas_herramientas": respaldados,
        "fidelidad_numerica_groundedness": fidelidad_citas,
        "cumple_umbral_1pct": fidelidad_citas >= 0.99 and validez_esquema >= 0.99,
        "detalle_ejecuciones": [
            {"herramienta": e["herramienta"], "exito": e["exito"], "mensaje_error": e.get("mensaje_error")}
            for e in ejecuciones
        ],
    }
