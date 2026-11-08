"""Catálogo de herramientas tipadas con esquemas JSON (Pydantic + formato nativo Ollama/OpenAI)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from scipy import stats

from . import config, datos


# ===========================================================================
# 1. Esquemas Pydantic con validación estricta de dominio y tipos
# ===========================================================================
class ArgumentosConsultarIndicadorBM(BaseModel):
    """Esquema de argumentos para consultar_indicador_banco_mundial."""

    codigo_pais: str = Field(
        ...,
        description=(
            "Código ISO3 del país (ECU, COL, PER, CHL, ARG, BRA, MEX, URY, CRI, BOL, PRY, USA) "
            "o nombre del país."
        ),
    )
    indicador: str = Field(
        ...,
        description=(
            "Indicador económico o social: 'pib_per_capita_usd', 'crecimiento_pib_pct', "
            "'inflacion_anual_pct', 'desempleo_pct', 'energias_renovables_pct', "
            "'esperanza_vida_anios', 'usuarios_internet_pct' o 'indice_gini'."
        ),
    )
    anio_inicio: int = Field(
        default=2018,
        description="Año inicial de consulta (entre 1990 y 2024).",
    )
    anio_fin: int = Field(
        default=2023,
        description="Año final de consulta (entre 1990 y 2024, mayor o igual a anio_inicio).",
    )

    @field_validator("codigo_pais")
    @classmethod
    def _validar_pais(cls, v: str) -> str:
        return datos.normalizar_pais(v)

    @field_validator("indicador")
    @classmethod
    def _validar_indicador(cls, v: str) -> str:
        return datos.normalizar_indicador(v)

    @model_validator(mode="after")
    def _validar_rango_anios(self) -> "ArgumentosConsultarIndicadorBM":
        a_min, a_max = datos.cobertura_bm()
        for campo, valor in (("anio_inicio", self.anio_inicio), ("anio_fin", self.anio_fin)):
            if not (a_min <= valor <= a_max):
                raise ValueError(
                    f"{campo}={valor} fuera de rango. "
                    f"El Banco Mundial en el modo '{config.MODO_DATOS}' cubre de {a_min} a {a_max}."
                )
        if self.anio_inicio > self.anio_fin:
            raise ValueError(
                f"anio_inicio ({self.anio_inicio}) no puede ser mayor que anio_fin ({self.anio_fin}). "
                "Invierte el orden de los años."
            )
        return self


class ArgumentosCompararPaisesIndicador(BaseModel):
    """Esquema de argumentos para comparar_paises_indicador."""

    codigos_paises: list[str] = Field(
        ...,
        description=(
            "Lista de al menos 2 códigos ISO3 de países a comparar "
            "(ej. ['ECU', 'COL', 'PER', 'CHL'])."
        ),
    )
    indicador: str = Field(
        ...,
        description=(
            "Indicador del Banco Mundial a comparar: 'pib_per_capita_usd', 'crecimiento_pib_pct', "
            "'inflacion_anual_pct', 'desempleo_pct', 'energias_renovables_pct', "
            "'esperanza_vida_anios', 'usuarios_internet_pct' o 'indice_gini'."
        ),
    )
    anio: int = Field(
        default=2022,
        description="Año exacto de comparación (entre 1990 y 2024).",
    )

    @field_validator("codigos_paises", mode="before")
    @classmethod
    def _normalizar_lista_paises(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            partes = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
        elif isinstance(v, (list, tuple)):
            partes = [str(p).strip() for p in v if str(p).strip()]
        else:
            raise ValueError("codigos_paises debe ser una lista de códigos ISO3.")
        normalizados: list[str] = []
        for p in partes:
            iso = datos.normalizar_pais(p)
            if iso not in normalizados:
                normalizados.append(iso)
        if len(normalizados) < 2:
            raise ValueError(
                "comparar_paises_indicador requiere al menos 2 países distintos en codigos_paises."
            )
        return normalizados

    @field_validator("indicador")
    @classmethod
    def _validar_indicador(cls, v: str) -> str:
        return datos.normalizar_indicador(v)

    @field_validator("anio")
    @classmethod
    def _validar_anio(cls, v: int) -> int:
        a_min, a_max = datos.cobertura_bm()
        if not (a_min <= int(v) <= a_max):
            raise ValueError(f"anio={v} fuera de rango permitido ({a_min}-{a_max}).")
        return int(v)


class ArgumentosClimaHistoricoCiudad(BaseModel):
    """Esquema de argumentos para consultar_clima_historico_ciudad."""

    ciudad: str = Field(
        ...,
        description=(
            "Nombre de la ciudad: 'Quito', 'Guayaquil', 'Bogotá', 'Lima', 'Santiago', "
            "'Buenos Aires', 'São Paulo', 'Ciudad de México', 'Montevideo', 'San José', "
            "'La Paz' o 'Asunción'."
        ),
    )
    fecha_inicio: str = Field(
        ...,
        description="Fecha inicial en formato YYYY-MM-DD (entre 2020-01-01 y 2024-12-31).",
    )
    fecha_fin: str = Field(
        ...,
        description="Fecha final en formato YYYY-MM-DD (entre 2020-01-01 y 2024-12-31).",
    )
    variable: str = Field(
        default="temp_media_c",
        description=(
            "Variable meteorológica: 'temp_media_c', 'temp_max_c', 'temp_min_c', "
            "'precipitacion_mm', 'radiacion_solar_mj_m2' o 'viento_max_kmh'."
        ),
    )

    @field_validator("ciudad")
    @classmethod
    def _validar_ciudad(cls, v: str) -> str:
        return datos.normalizar_ciudad(v)

    @field_validator("variable")
    @classmethod
    def _validar_variable(cls, v: str) -> str:
        return datos.normalizar_variable_clima(v)

    @model_validator(mode="after")
    def _validar_fechas(self) -> "ArgumentosClimaHistoricoCiudad":
        def _parse_fecha(cad: str, es_fin: bool) -> str:
            limpia = str(cad).strip()
            if len(limpia) == 4 and limpia.isdigit():
                return f"{limpia}-12-31" if es_fin else f"{limpia}-01-01"
            if len(limpia) == 7 and limpia[:4].isdigit() and limpia[5:7].isdigit():
                anio_i, mes_i = int(limpia[:4]), int(limpia[5:7])
                if not es_fin:
                    return f"{anio_i:04d}-{mes_i:02d}-01"
                dias_mes = [31, 29 if anio_i % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                return f"{anio_i:04d}-{mes_i:02d}-{dias_mes[mes_i - 1]:02d}"
            try:
                dt = datetime.strptime(limpia[:10], "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    f"Fecha '{cad}' inválida. Usa el formato ISO 'YYYY-MM-DD' entre 2020-01-01 y 2024-12-31."
                ) from exc

        f_ini = _parse_fecha(self.fecha_inicio, es_fin=False)
        f_fin = _parse_fecha(self.fecha_fin, es_fin=True)
        c_ini, c_fin = datos.cobertura_clima()
        if f_ini < c_ini or f_fin > c_fin:
            raise ValueError(
                f"Rango de fechas ({f_ini} a {f_fin}) fuera de cobertura. "
                f"El clima histórico en el modo '{config.MODO_DATOS}' cubre de {c_ini} a {c_fin}."
            )
        if f_ini > f_fin:
            raise ValueError(
                f"fecha_inicio ({f_ini}) no puede ser posterior a fecha_fin ({f_fin})."
            )
        self.fecha_inicio = f_ini
        self.fecha_fin = f_fin
        return self


class ArgumentosResumenClimaticoAnual(BaseModel):
    """Esquema de argumentos para resumen_climatico_anual_ciudad."""

    ciudad: str = Field(
        ...,
        description=(
            "Ciudad a consultar: 'Quito', 'Guayaquil', 'Bogotá', 'Lima', 'Santiago', "
            "'Buenos Aires', 'São Paulo', 'Ciudad de México', 'Montevideo', 'San José', "
            "'La Paz' o 'Asunción'."
        ),
    )
    anio: int = Field(
        ...,
        description="Año del resumen climático anual (entre 2020 y 2024).",
    )

    @field_validator("ciudad")
    @classmethod
    def _validar_ciudad(cls, v: str) -> str:
        return datos.normalizar_ciudad(v)

    @field_validator("anio")
    @classmethod
    def _validar_anio(cls, v: int) -> int:
        anio_int = int(v)
        c_ini, c_fin = datos.cobertura_clima()
        if not (int(c_ini[:4]) <= anio_int <= int(c_fin[:4])):
            raise ValueError(
                f"anio={v} fuera de rango climático. En el modo '{config.MODO_DATOS}' "
                f"el clima histórico cubre de {c_ini[:4]} a {c_fin[:4]}."
            )
        return anio_int


OPERACIONES_VALIDAS = {
    "variacion_porcentual": "variacion_porcentual",
    "cambio_porcentual": "variacion_porcentual",
    "tasa_variacion": "variacion_porcentual",
    "crecimiento": "variacion_porcentual",
    "promedio": "promedio",
    "media": "promedio",
    "estadisticas": "promedio",
    "diferencia": "diferencia",
    "resta": "diferencia",
    "brecha": "diferencia",
    "extremos": "extremos",
    "min_max": "extremos",
    "correlacion": "correlacion",
    "correlación": "correlacion",
    "pearson": "correlacion",
}


class ArgumentosCalculadora(BaseModel):
    """Esquema de argumentos para calcular_correlacion_o_cambio."""

    valores_x: list[float] = Field(
        ...,
        description=(
            "Lista numérica principal (ej. [valor_inicial, valor_final] para 'variacion_porcentual' "
            "o 'diferencia', o serie completa de valores para 'promedio', 'extremos' o 'correlacion')."
        ),
    )
    valores_y: list[float] | None = Field(
        default=None,
        description=(
            "Segunda lista numérica opcional (obligatoria cuando operacion='correlacion', "
            "del mismo tamaño que valores_x)."
        ),
    )
    operacion: str = Field(
        default="variacion_porcentual",
        description=(
            "Operación matemática determinista: 'variacion_porcentual', 'diferencia', "
            "'promedio', 'extremos' o 'correlacion'."
        ),
    )

    @field_validator("valores_x", mode="before")
    @classmethod
    def _validar_x(cls, v: Any) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v)]
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError("valores_x debe ser una lista no vacía de números.")
        return [float(x) for x in v]

    @field_validator("valores_y", mode="before")
    @classmethod
    def _validar_y(cls, v: Any) -> list[float] | None:
        if v is None or v == []:
            return None
        if isinstance(v, (int, float)):
            return [float(v)]
        if not isinstance(v, (list, tuple)):
            raise ValueError("valores_y debe ser una lista de números o null.")
        return [float(y) for y in v]

    @field_validator("operacion")
    @classmethod
    def _validar_op(cls, v: str) -> str:
        clave = datos.quitar_acentos(str(v).strip().lower()).replace(" ", "_")
        if clave in OPERACIONES_VALIDAS:
            return OPERACIONES_VALIDAS[clave]
        raise ValueError(
            f"Operación '{v}' no válida. Usa una de: "
            "['variacion_porcentual', 'diferencia', 'promedio', 'extremos', 'correlacion']."
        )

    @model_validator(mode="after")
    def _validar_consistencia(self) -> "ArgumentosCalculadora":
        if (
            self.operacion in ("variacion_porcentual", "diferencia")
            and len(self.valores_x) == 1
            and self.valores_y
            and len(self.valores_y) >= 1
        ):
            self.valores_x = [self.valores_x[0], self.valores_y[0]]

        if self.operacion in ("variacion_porcentual", "diferencia") and len(self.valores_x) < 2:
            raise ValueError(
                f"La operación '{self.operacion}' requiere al menos 2 valores en valores_x "
                "(por ejemplo [valor_inicial, valor_final])."
            )
        if self.operacion == "correlacion":
            if not self.valores_y or len(self.valores_y) != len(self.valores_x):
                raise ValueError(
                    "Para operacion='correlacion' debes proporcionar valores_x y valores_y "
                    "con la misma cantidad de elementos (mínimo 3 pares)."
                )
            if len(self.valores_x) < 3:
                raise ValueError(
                    "Para calcular correlación se requieren al menos 3 pares de observaciones."
                )
        return self


# ===========================================================================
# 2. Implementación determinista de las 5 herramientas
# ===========================================================================
def consultar_indicador_banco_mundial(
    codigo_pais: str,
    indicador: str,
    anio_inicio: int = 2018,
    anio_fin: int = 2023,
) -> dict[str, Any]:
    """Consulta la serie histórica anual de un indicador del Banco Mundial para un país."""
    args = ArgumentosConsultarIndicadorBM(
        codigo_pais=codigo_pais,
        indicador=indicador,
        anio_inicio=anio_inicio,
        anio_fin=anio_fin,
    )
    serie_completa = datos.serie_indicador(args.codigo_pais, args.indicador, args.anio_inicio, args.anio_fin)
    meta_ind = config.INDICADORES_BM[args.indicador]
    meta_pais = config.PAISES[args.codigo_pais]

    serie_filtrada: dict[str, float] = {}
    for anio_s, val in serie_completa.items():
        a_int = int(anio_s)
        if args.anio_inicio <= a_int <= args.anio_fin and val is not None:
            serie_filtrada[anio_s] = round(float(val), 2)

    valores = list(serie_filtrada.values())
    anios = list(serie_filtrada.keys())
    resumen: dict[str, Any] = {}
    cifras_clave: list[float] = []

    if valores:
        v_ini = valores[0]
        v_fin = valores[-1]
        v_prom = round(float(np.mean(valores)), 2)
        v_min = round(float(np.min(valores)), 2)
        v_max = round(float(np.max(valores)), 2)
        # Las variaciones las calcula solo calcular_correlacion_o_cambio (una sola fuente de cálculo).
        resumen = {
            "anio_inicial_disponible": int(anios[0]),
            "valor_inicial": v_ini,
            "anio_final_disponible": int(anios[-1]),
            "valor_final": v_fin,
            "promedio_periodo": v_prom,
            "minimo_periodo": v_min,
            "maximo_periodo": v_max,
            "n_observaciones": len(valores),
        }
        cifras_clave = list(dict.fromkeys([*valores, v_prom, v_min, v_max]))

    return {
        "herramienta": "consultar_indicador_banco_mundial",
        "codigo_pais": args.codigo_pais,
        "pais": meta_pais["nombre"],
        "indicador": args.indicador,
        "codigo_bm": meta_ind["codigo_bm"],
        "nombre_indicador": meta_ind["nombre"],
        "unidad": meta_ind["unidad"],
        "anio_inicio": args.anio_inicio,
        "anio_fin": args.anio_fin,
        "serie_anual": serie_filtrada,
        "resumen": resumen,
        "cifras_clave": cifras_clave,
        "fuente": f"Banco Mundial WDI ({meta_ind['codigo_bm']}) · modo {config.MODO_DATOS}",
    }


def comparar_paises_indicador(
    codigos_paises: list[str],
    indicador: str,
    anio: int = 2022,
) -> dict[str, Any]:
    """Compara el valor de un indicador oficial del Banco Mundial entre varios países en un año."""
    args = ArgumentosCompararPaisesIndicador(
        codigos_paises=codigos_paises,
        indicador=indicador,
        anio=anio,
    )
    meta_ind = config.INDICADORES_BM[args.indicador]
    anio_str = str(args.anio)
    a_min, a_max = datos.cobertura_bm()

    comparativa: list[dict[str, Any]] = []
    for iso3 in args.codigos_paises:
        serie_pais = datos.serie_indicador(
            iso3, args.indicador, max(a_min, args.anio - 2), min(a_max, args.anio + 2)
        )
        val = serie_pais.get(anio_str)
        anio_efectivo = args.anio
        if val is None and serie_pais:
            # Busca el año más cercano disponible si el año exacto no tiene reporte
            anios_disp = [int(a) for a in serie_pais.keys()]
            mas_cercano = min(anios_disp, key=lambda x: abs(x - args.anio))
            if abs(mas_cercano - args.anio) <= 2:
                val = serie_pais[str(mas_cercano)]
                anio_efectivo = mas_cercano
        if val is not None:
            comparativa.append({
                "codigo_pais": iso3,
                "pais": config.PAISES[iso3]["nombre"],
                "anio": anio_efectivo,
                "valor": round(float(val), 2),
            })

    comparativa.sort(key=lambda r: r["valor"], reverse=True)
    valores = [r["valor"] for r in comparativa]
    cifras_clave = list(valores)
    resumen: dict[str, Any] = {}
    if len(comparativa) >= 2:
        lider = comparativa[0]
        menor = comparativa[-1]
        brecha = round(lider["valor"] - menor["valor"], 2)
        prom = round(float(np.mean(valores)), 2)
        resumen = {
            "pais_mayor": lider["pais"],
            "codigo_mayor": lider["codigo_pais"],
            "valor_mayor": lider["valor"],
            "pais_menor": menor["pais"],
            "codigo_menor": menor["codigo_pais"],
            "valor_menor": menor["valor"],
            "brecha_mayor_menor": brecha,
            "promedio_grupo": prom,
        }
        cifras_clave.extend([brecha, prom])

    return {
        "herramienta": "comparar_paises_indicador",
        "indicador": args.indicador,
        "codigo_bm": meta_ind["codigo_bm"],
        "nombre_indicador": meta_ind["nombre"],
        "unidad": meta_ind["unidad"],
        "anio_solicitado": args.anio,
        "ranking": comparativa,
        "resumen": resumen,
        "cifras_clave": list(dict.fromkeys(cifras_clave)),
        "fuente": f"Banco Mundial WDI ({meta_ind['codigo_bm']}) · modo {config.MODO_DATOS}",
    }


def consultar_clima_historico_ciudad(
    ciudad: str,
    fecha_inicio: str,
    fecha_fin: str,
    variable: str = "temp_media_c",
) -> dict[str, Any]:
    """Consulta una variable climática de Open-Meteo para una ciudad en un intervalo de fechas (2020-2024)."""
    args = ArgumentosClimaHistoricoCiudad(
        ciudad=ciudad,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        variable=variable,
    )
    meta_var = config.VARIABLES_CLIMA[args.variable]
    meta_ciudad = config.CIUDADES_CLIMA[args.ciudad]
    meses_seleccionados = datos.clima_mensual(args.ciudad, args.fecha_inicio, args.fecha_fin)
    if not meses_seleccionados:
        raise ValueError(
            f"No hay registros para {args.ciudad} entre {args.fecha_inicio} y {args.fecha_fin}."
        )

    valores_mes = [float(m[args.variable]) for m in meses_seleccionados.values()]
    dias_mes = [int(m.get("dias_registrados", 30)) for m in meses_seleccionados.values()]
    agregacion = meta_var["agregacion"]

    if agregacion == "suma":
        valor_agregado = round(float(np.sum(valores_mes)), 2)
    elif agregacion == "maximo":
        valor_agregado = round(float(np.max(valores_mes)), 2)
    elif agregacion == "minimo":
        valor_agregado = round(float(np.min(valores_mes)), 2)
    else:
        valor_agregado = round(float(np.average(valores_mes, weights=dias_mes)), 2)

    val_max_mes = round(float(np.max(valores_mes)), 2)
    val_min_mes = round(float(np.min(valores_mes)), 2)
    prom_mensual = round(float(np.mean(valores_mes)), 2)
    mes_pico = max(meses_seleccionados.items(), key=lambda kv: kv[1][args.variable])[0]
    mes_bajo = min(meses_seleccionados.items(), key=lambda kv: kv[1][args.variable])[0]

    cifras_clave = list(dict.fromkeys([
        valor_agregado,
        val_max_mes,
        val_min_mes,
        prom_mensual,
        * [round(v, 2) for v in valores_mes],
    ]))

    return {
        "herramienta": "consultar_clima_historico_ciudad",
        "ciudad": args.ciudad,
        "codigo_pais": meta_ciudad["codigo_pais"],
        "altitud_m": meta_ciudad["altitud_m"],
        "fecha_inicio": args.fecha_inicio,
        "fecha_fin": args.fecha_fin,
        "variable": args.variable,
        "nombre_variable": meta_var["nombre"],
        "unidad": meta_var["unidad"],
        "tipo_agregacion": agregacion,
        "valor_periodo": valor_agregado,
        "promedio_mensual": prom_mensual,
        "maximo_mensual": val_max_mes,
        "mes_maximo": mes_pico,
        "minimo_mensual": val_min_mes,
        "mes_minimo": mes_bajo,
        "n_meses": len(meses_seleccionados),
        "cifras_clave": cifras_clave,
        "fuente": f"Open-Meteo Historical Weather API (ERA5) · modo {config.MODO_DATOS}",
    }


def resumen_climatico_anual_ciudad(
    ciudad: str,
    anio: int,
) -> dict[str, Any]:
    """Devuelve el resumen climático completo de una ciudad en un año específico (2020-2024)."""
    args = ArgumentosResumenClimaticoAnual(ciudad=ciudad, anio=anio)
    meta_ciudad = config.CIUDADES_CLIMA[args.ciudad]
    datos_anio = datos.clima_anual(args.ciudad, args.anio)
    if not datos_anio:
        raise ValueError(f"No existen registros anuales para {args.ciudad} en {args.anio}.")

    cifras_clave = [
        round(float(datos_anio["temp_media_c"]), 2),
        round(float(datos_anio["temp_max_c"]), 2),
        round(float(datos_anio["temp_min_c"]), 2),
        round(float(datos_anio["precipitacion_mm"]), 2),
        round(float(datos_anio["radiacion_solar_mj_m2"]), 2),
        round(float(datos_anio["viento_max_kmh"]), 2),
        float(datos_anio["dias_lluvia"]),
    ]

    return {
        "herramienta": "resumen_climatico_anual_ciudad",
        "ciudad": args.ciudad,
        "codigo_pais": meta_ciudad["codigo_pais"],
        "altitud_m": meta_ciudad["altitud_m"],
        "anio": args.anio,
        "metricas_anuales": datos_anio,
        "unidades": {
            "temp_media_c": "°C",
            "temp_max_c": "°C",
            "temp_min_c": "°C",
            "precipitacion_mm": "mm",
            "radiacion_solar_mj_m2": "MJ/m²",
            "viento_max_kmh": "km/h",
            "dias_lluvia": "días (>=1 mm)",
        },
        "cifras_clave": list(dict.fromkeys(cifras_clave)),
        "fuente": f"Open-Meteo Historical Weather API (ERA5) · modo {config.MODO_DATOS}",
    }


def calcular_correlacion_o_cambio(
    valores_x: list[float],
    valores_y: list[float] | None = None,
    operacion: str = "variacion_porcentual",
) -> dict[str, Any]:
    """Calculadora determinista para variaciones porcentuales, diferencias, promedios, extremos y correlación."""
    args = ArgumentosCalculadora(
        valores_x=valores_x,
        valores_y=valores_y,
        operacion=operacion,
    )
    arr_x = np.asarray(args.valores_x, dtype=float)
    resultado: dict[str, Any] = {
        "herramienta": "calcular_correlacion_o_cambio",
        "operacion": args.operacion,
        "n_elementos": len(arr_x),
    }
    cifras_clave: list[float] = []

    if args.operacion == "variacion_porcentual":
        v_ini = float(arr_x[0])
        v_fin = float(arr_x[-1])
        cambio_abs = round(v_fin - v_ini, 2)
        cambio_pct = round(((v_fin - v_ini) / abs(v_ini)) * 100.0, 2) if abs(v_ini) > 1e-9 else 0.0
        razon = round(v_fin / v_ini, 4) if abs(v_ini) > 1e-9 else 0.0
        resultado.update({
            "valor_inicial": round(v_ini, 2),
            "valor_final": round(v_fin, 2),
            "cambio_absoluto": cambio_abs,
            "variacion_porcentual": cambio_pct,
            "razon_multiplicativa": razon,
            "unidad": "%",
        })
        cifras_clave.extend([round(v_ini, 2), round(v_fin, 2), cambio_abs, cambio_pct, round(razon, 2)])

    elif args.operacion == "diferencia":
        v_a = float(arr_x[0])
        v_b = float(arr_x[-1])
        dif_ab = round(v_a - v_b, 2)
        dif_ba = round(v_b - v_a, 2)
        dif_abs = round(abs(v_a - v_b), 2)
        pct_sobre_b = round(((v_a - v_b) / abs(v_b)) * 100.0, 2) if abs(v_b) > 1e-9 else 0.0
        resultado.update({
            "primer_valor": round(v_a, 2),
            "segundo_valor": round(v_b, 2),
            "diferencia_primero_menos_segundo": dif_ab,
            "diferencia_segundo_menos_primero": dif_ba,
            "diferencia_absoluta": dif_abs,
            "diferencia_relativa_pct": pct_sobre_b,
        })
        cifras_clave.extend([round(v_a, 2), round(v_b, 2), dif_ab, dif_ba, dif_abs, pct_sobre_b])

    elif args.operacion == "promedio":
        media = round(float(np.mean(arr_x)), 2)
        mediana = round(float(np.median(arr_x)), 2)
        desv = round(float(np.std(arr_x, ddof=0)), 2)
        suma = round(float(np.sum(arr_x)), 2)
        resultado.update({
            "promedio": media,
            "mediana": mediana,
            "desviacion_estandar": desv,
            "suma_total": suma,
            "minimo": round(float(np.min(arr_x)), 2),
            "maximo": round(float(np.max(arr_x)), 2),
        })
        cifras_clave.extend([media, mediana, desv, suma, round(float(np.min(arr_x)), 2), round(float(np.max(arr_x)), 2)])

    elif args.operacion == "extremos":
        v_min = round(float(np.min(arr_x)), 2)
        v_max = round(float(np.max(arr_x)), 2)
        rango = round(v_max - v_min, 2)
        resultado.update({
            "minimo": v_min,
            "maximo": v_max,
            "rango": rango,
            "promedio": round(float(np.mean(arr_x)), 2),
        })
        cifras_clave.extend([v_min, v_max, rango, round(float(np.mean(arr_x)), 2)])

    elif args.operacion == "correlacion":
        arr_y = np.asarray(args.valores_y, dtype=float)
        r_pearson, p_val = stats.pearsonr(arr_x, arr_y)
        r_spearman, _ = stats.spearmanr(arr_x, arr_y)
        r_p = round(float(r_pearson), 4)
        r_s = round(float(r_spearman), 4)
        resultado.update({
            "correlacion_pearson": r_p,
            "correlacion_pearson_2dec": round(r_p, 2),
            "p_valor": round(float(p_val), 4),
            "correlacion_spearman": r_s,
            "promedio_x": round(float(np.mean(arr_x)), 2),
            "promedio_y": round(float(np.mean(arr_y)), 2),
        })
        cifras_clave.extend([
            r_p,
            round(r_p, 2),
            r_s,
            round(r_s, 2),
            round(float(np.mean(arr_x)), 2),
            round(float(np.mean(arr_y)), 2),
        ])

    resultado["cifras_clave"] = list(dict.fromkeys(cifras_clave))
    resultado["fuente"] = "Calculadora determinista SciPy/NumPy"
    return resultado


# ===========================================================================
# 3. Registro unificado y exportación de esquemas JSON para Ollama / OpenAI
# ===========================================================================
REGISTRO_HERRAMIENTAS: dict[str, dict[str, Any]] = {
    "consultar_indicador_banco_mundial": {
        "funcion": consultar_indicador_banco_mundial,
        "modelo_pydantic": ArgumentosConsultarIndicadorBM,
    },
    "comparar_paises_indicador": {
        "funcion": comparar_paises_indicador,
        "modelo_pydantic": ArgumentosCompararPaisesIndicador,
    },
    "consultar_clima_historico_ciudad": {
        "funcion": consultar_clima_historico_ciudad,
        "modelo_pydantic": ArgumentosClimaHistoricoCiudad,
    },
    "resumen_climatico_anual_ciudad": {
        "funcion": resumen_climatico_anual_ciudad,
        "modelo_pydantic": ArgumentosResumenClimaticoAnual,
    },
    "calcular_correlacion_o_cambio": {
        "funcion": calcular_correlacion_o_cambio,
        "modelo_pydantic": ArgumentosCalculadora,
    },
}


def descripciones_herramientas(version: str = "v1") -> dict[str, str]:
    """Descripciones que ve el LLM. v1 = originales; v2 = mejora de prompt evaluada en validación
    (cuándo usar cada herramienta, formato de los argumentos y un ejemplo)."""
    a_min, a_max = datos.cobertura_bm()
    c_ini, c_fin = datos.cobertura_clima()
    paises = ", ".join(config.PAISES)
    indicadores = ", ".join(config.INDICADORES_BM)
    ciudades = ", ".join(config.CIUDADES_CLIMA)
    variables = ", ".join(config.VARIABLES_CLIMA)
    v1 = {
        "consultar_indicador_banco_mundial": (
            f"Consulta la serie anual oficial del Banco Mundial ({a_min}-{a_max}) para un país "
            f"({paises}) y un indicador ({indicadores})."
        ),
        "comparar_paises_indicador": (
            f"Compara un indicador del Banco Mundial entre 2 o más países (códigos ISO3) "
            f"en un año específico ({a_min}-{a_max}) y devuelve el ranking ordenado, el líder y la brecha."
        ),
        "consultar_clima_historico_ciudad": (
            f"Consulta series climáticas históricas de Open-Meteo ({c_ini} a {c_fin}) "
            f"para una ciudad ({ciudades}) y una variable ({variables})."
        ),
        "resumen_climatico_anual_ciudad": (
            "Obtiene todas las métricas climáticas anuales de Open-Meteo (temperatura media, máxima, "
            "mínima, precipitación acumulada en mm, radiación solar, viento máximo y días de lluvia) "
            f"para una ciudad en un año entre {c_ini[:4]} y {c_fin[:4]}."
        ),
        "calcular_correlacion_o_cambio": (
            "Ejecuta cálculos matemáticos exactos sin error de redondeo: 'variacion_porcentual' "
            "entre dos cifras, 'diferencia', 'promedio', 'extremos' o 'correlacion' de Pearson "
            "entre dos series valores_x y valores_y."
        ),
    }
    if version == "v1":
        return v1
    return {
        "consultar_indicador_banco_mundial": v1["consultar_indicador_banco_mundial"] + (
            " Úsala para UN solo país; anio_inicio y anio_fin son los años que menciona la pregunta "
            "(si dice 'en 2023', ambos son 2023). Ejemplo: {\"codigo_pais\": \"BRA\", "
            "\"indicador\": \"desempleo_pct\", \"anio_inicio\": 2015, \"anio_fin\": 2018}."
        ),
        "comparar_paises_indicador": v1["comparar_paises_indicador"] + (
            " Úsala cuando la pregunta nombra 2 o más países para un mismo año, en una sola llamada. "
            "Ejemplo: {\"codigos_paises\": [\"ARG\", \"BRA\"], \"indicador\": \"desempleo_pct\", \"anio\": 2022}."
        ),
        "consultar_clima_historico_ciudad": v1["consultar_clima_historico_ciudad"] + (
            " Úsala para UNA variable concreta en un rango de fechas YYYY-MM-DD (un año completo = "
            "YYYY-01-01 a YYYY-12-31). Ejemplo: {\"ciudad\": \"Lima\", \"fecha_inicio\": \"2022-01-01\", "
            "\"fecha_fin\": \"2022-12-31\", \"variable\": \"precipitacion_mm\"}."
        ),
        "resumen_climatico_anual_ciudad": v1["resumen_climatico_anual_ciudad"] + (
            " Úsala cuando piden el clima o resumen climático general de un año, no una sola variable. "
            "Ejemplo: {\"ciudad\": \"Quito\", \"anio\": 2021}."
        ),
        "calcular_correlacion_o_cambio": v1["calcular_correlacion_o_cambio"] + (
            " Llámala DESPUÉS de obtener los datos, copiando los valores observados; nunca calcules "
            "de memoria. Para variación porcentual: valores_x = [valor_inicial, valor_final]. "
            "Ejemplo: {\"valores_x\": [1200.0, 1500.0], \"operacion\": \"variacion_porcentual\"}."
        ),
    }


def obtener_esquemas_ollama(version: str = "v1") -> list[dict[str, Any]]:
    """Genera la lista de definiciones de herramientas en el formato estándar de Ollama/OpenAI."""
    descripciones = descripciones_herramientas(version)
    esquemas: list[dict[str, Any]] = []
    for nombre, meta in REGISTRO_HERRAMIENTAS.items():
        modelo: type[BaseModel] = meta["modelo_pydantic"]
        schema_raw = modelo.model_json_schema()
        propiedades = {}
        for prop_k, prop_v in schema_raw.get("properties", {}).items():
            limpio_v = {k: v for k, v in prop_v.items() if k in ("type", "items", "default")}
            if "anyOf" in prop_v:  # list[float] | None -> array
                limpio_v = {"type": "array", "items": {"type": "number"}}
            propiedades[prop_k] = limpio_v
        esquemas.append({
            "type": "function",
            "function": {
                "name": nombre,
                "description": descripciones[nombre],
                "parameters": {
                    "type": "object",
                    "properties": propiedades,
                    "required": schema_raw.get("required", []),
                },
            },
        })
    return esquemas


def ejecutar_herramienta_segura(
    nombre_herramienta: str,
    argumentos: dict[str, Any],
) -> dict[str, Any]:
    """Valida los argumentos con Pydantic y ejecuta la herramienta devolviendo estado y resultado."""
    if nombre_herramienta not in REGISTRO_HERRAMIENTAS:
        return {
            "exito": False,
            "herramienta": nombre_herramienta,
            "error_tipo": "herramienta_inexistente",
            "mensaje_error": (
                f"La herramienta '{nombre_herramienta}' no existe. "
                f"Herramientas válidas: {list(REGISTRO_HERRAMIENTAS.keys())}."
            ),
        }

    meta = REGISTRO_HERRAMIENTAS[nombre_herramienta]
    modelo: type[BaseModel] = meta["modelo_pydantic"]
    funcion: Callable[..., dict[str, Any]] = meta["funcion"]

    try:
        instancia_validada = modelo.model_validate(argumentos or {})
        args_limpios = instancia_validada.model_dump()
        salida = funcion(**args_limpios)
        return {
            "exito": True,
            "herramienta": nombre_herramienta,
            "argumentos_validados": args_limpios,
            "resultado": salida,
        }
    except ValidationError as exc:
        mensajes = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in exc.errors())
        return {
            "exito": False,
            "herramienta": nombre_herramienta,
            "argumentos_originales": argumentos,
            "error_tipo": "validacion_pydantic",
            "mensaje_error": f"Error de validación en '{nombre_herramienta}': {mensajes}",
        }
    except Exception as exc:
        return {
            "exito": False,
            "herramienta": nombre_herramienta,
            "argumentos_originales": argumentos,
            "error_tipo": "error_ejecucion",
            "mensaje_error": f"Error al ejecutar '{nombre_herramienta}': {exc}",
        }
