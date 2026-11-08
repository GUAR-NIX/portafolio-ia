"""Pruebas automáticas (pytest) del Agente de Datos Abiertos. Corren sin Ollama ni red:
las respuestas del LLM salen de data/grabacion_llm.jsonl y las APIs se simulan con httpx.MockTransport."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from agente import clientes_api, config, datos, evaluacion, herramientas, motor_agente
from api.main import app

cliente_api = TestClient(app)
RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _sin_ollama_ni_red(monkeypatch):
    """Cualquier llamada real a Ollama hace fallar la prueba: todo debe salir de la grabación."""
    class _ClienteProhibido:
        def __init__(self, *_a, **_k):
            raise AssertionError("La prueba intentó llamar a Ollama: falta una respuesta grabada.")

    monkeypatch.setattr(motor_agente, "httpx", SimpleNamespace(Client=_ClienteProhibido, ConnectError=httpx.ConnectError))
    datos.establecer_modo("instantanea")
    yield
    datos.establecer_modo("instantanea")


def _resultados() -> dict:
    return json.loads((config.REPORTES_DIR / "resultados.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Datos y herramientas
# ---------------------------------------------------------------------------
def test_01_instantaneas_datos_existen_y_cobertura():
    bm = datos.cargar_instantanea_bm()
    clima = datos.cargar_instantanea_clima()
    assert len(bm["paises"]) == 12 and len(bm["indicadores"]) == 8
    assert sum(r["n_observaciones"] for r in bm["resumen_cobertura"]) >= 3000
    assert len(clima["ciudades"]) == 12 and "2023" in clima["anual"]["Quito"]


def test_02_normalizacion_paises_indicadores_ciudades():
    assert datos.normalizar_pais("perú") == "PER"
    assert datos.normalizar_indicador("NY.GDP.PCAP.CD") == "pib_per_capita_usd"
    assert datos.normalizar_ciudad("CDMX") == "Ciudad de México"
    assert datos.normalizar_variable_clima("lluvia") == "precipitacion_mm"
    for funcion, valor in ((datos.normalizar_pais, "España"), (datos.normalizar_ciudad, "Madrid"),
                           (datos.normalizar_indicador, "precio_bitcoin")):
        with pytest.raises(ValueError):
            funcion(valor)


def test_03_cifras_reales_pib_ecuador():
    """Las cifras que citan README y guía salen de la instantánea (antes la guía usaba 5.645,20 y 6.609,80)."""
    serie = herramientas.consultar_indicador_banco_mundial("ECU", "pib_per_capita_usd", 2015, 2023)["serie_anual"]
    assert serie["2015"] == 5976.16 and serie["2020"] == 5463.65 and serie["2023"] == 6718.14
    calc = herramientas.calcular_correlacion_o_cambio([5976.16, 6718.14], operacion="variacion_porcentual")
    assert calc["variacion_porcentual"] == 12.42
    guia = (RAIZ / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")
    for cifra_inventada in ("5645", "5.645", "6609", "6.609", "6131", "6.131"):
        assert cifra_inventada not in guia


def test_04_validacion_pydantic_y_cobertura_por_modo():
    ej = herramientas.ejecutar_herramienta_segura(
        "consultar_indicador_banco_mundial",
        {"codigo_pais": "ECU", "indicador": "pib_per_capita_usd", "anio_inicio": 2023, "anio_fin": 2019},
    )
    assert ej["exito"] is False and ej["error_tipo"] == "validacion_pydantic"
    args_1995 = {"ciudad": "Quito", "fecha_inicio": "1995-01-01", "fecha_fin": "1995-12-31", "variable": "precipitacion_mm"}
    assert herramientas.ejecutar_herramienta_segura("consultar_clima_historico_ciudad", args_1995)["exito"] is False
    datos.establecer_modo("vivo")  # Open-Meteo Archive llega a 1940: el esquema lo acepta en modo vivo
    assert herramientas.ArgumentosClimaHistoricoCiudad(**args_1995).fecha_inicio == "1995-01-01"


def test_05_herramientas_comparar_clima_resumen_y_calculadora():
    comp = herramientas.comparar_paises_indicador(["ECU", "COL", "PER", "CHL"], "pib_per_capita_usd", 2022)
    assert [r["codigo_pais"] for r in comp["ranking"]][0] == "CHL"
    lluvia = herramientas.consultar_clima_historico_ciudad("Guayaquil", "2023-01-01", "2023-12-31", "precipitacion_mm")
    assert lluvia["n_meses"] == 12 and lluvia["valor_periodo"] > 500
    assert 8 <= herramientas.resumen_climatico_anual_ciudad("Quito", 2023)["metricas_anuales"]["temp_media_c"] <= 22
    corr = herramientas.calcular_correlacion_o_cambio([10, 20, 30, 40, 50], [15, 29, 46, 60, 74], "correlacion")
    assert corr["correlacion_pearson"] > 0.99


def test_06_esquemas_ollama_v1_y_v2():
    v1, v2 = herramientas.obtener_esquemas_ollama("v1"), herramientas.obtener_esquemas_ollama("v2")
    assert [e["function"]["name"] for e in v1] == [e["function"]["name"] for e in v2]
    assert len(v1) == 5
    assert all("Ejemplo" in e["function"]["description"] for e in v2)
    assert not any("Ejemplo" in e["function"]["description"] for e in v1)


# ---------------------------------------------------------------------------
# Cliente HTTP real (simulado con MockTransport): caché, pausas y reintentos
# ---------------------------------------------------------------------------
def test_07_cliente_http_reintenta_429_y_cachea(tmp_path):
    llamadas = {"n": 0}

    def manejador(request: httpx.Request) -> httpx.Response:
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=[{"page": 1}, [
            {"countryiso3code": "ECU", "date": "2021", "value": 6061.3},
            {"countryiso3code": "ECU", "date": "2020", "value": 5463.6},
            {"countryiso3code": "ECU", "date": "2019", "value": None},
        ]])

    pausas: list[float] = []
    cli = clientes_api.ClienteHTTPCacheado(dir_cache=tmp_path, transporte=httpx.MockTransport(manejador),
                                           dormir=pausas.append)
    serie = clientes_api.serie_banco_mundial("ECU", "NY.GDP.PCAP.CD", 2019, 2021, cliente=cli)
    assert serie == {"2020": 5463.6, "2021": 6061.3}
    assert llamadas["n"] == 2 and cli.estadisticas["reintentos"] == 1 and 1.0 in pausas
    clientes_api.serie_banco_mundial("ECU", "NY.GDP.PCAP.CD", 2019, 2021, cliente=cli)  # segunda vez: caché
    assert llamadas["n"] == 2 and cli.estadisticas["aciertos_cache"] == 1


def test_08_cliente_http_falla_tras_reintentos(tmp_path):
    cli = clientes_api.ClienteHTTPCacheado(
        dir_cache=tmp_path, reintentos=3, dormir=lambda s: None,
        transporte=httpx.MockTransport(lambda r: httpx.Response(503)),
    )
    with pytest.raises(clientes_api.ErrorAPI):
        cli.get_json("https://ejemplo.invalido/api", {"a": 1})


def test_09_modo_vivo_open_meteo_agrega_igual_que_la_instantanea(tmp_path, monkeypatch):
    """En modo vivo la herramienta usa el cliente HTTP; con los días del archivo crudo reproduce la instantánea."""
    dias = 31
    diario = {
        "time": [f"2023-01-{d:02d}" for d in range(1, dias + 1)],
        "temperature_2m_mean": [14.0] * dias, "temperature_2m_max": [20.0] * dias,
        "temperature_2m_min": [8.0] * dias, "precipitation_sum": [2.0] * dias,
        "shortwave_radiation_sum": [18.0] * dias, "wind_speed_10m_max": [12.0] * dias,
    }
    cli = clientes_api.ClienteHTTPCacheado(
        dir_cache=tmp_path, dormir=lambda s: None,
        transporte=httpx.MockTransport(lambda r: httpx.Response(200, json={"daily": diario})),
    )
    monkeypatch.setattr(clientes_api, "_CLIENTE", cli)
    datos.establecer_modo("vivo")
    res = herramientas.consultar_clima_historico_ciudad("Quito", "2023-01-01", "2023-01-31", "precipitacion_mm")
    assert res["valor_periodo"] == 62.0 and "modo vivo" in res["fuente"]


# ---------------------------------------------------------------------------
# Detección de dominio por capacidades y anclaje de argumentos
# ---------------------------------------------------------------------------
def test_10_sin_listas_de_palabras_del_benchmark_en_el_motor():
    fuente = (RAIZ / "src" / "agente" / "motor_agente.py").read_text(encoding="utf-8").lower()
    for palabra in ("bitcoin", "nasdaq", "marte", "litio", "dolar blue", "camas de hospital", "japon", "tokio"):
        assert palabra not in fuente


def test_11_anclaje_rechaza_argumentos_inventados_y_acepta_los_de_la_pregunta():
    ok, _ = motor_agente.verificar_anclaje(
        "consultar_indicador_banco_mundial",
        {"codigo_pais": "ECU", "indicador": "pib_per_capita_usd", "anio_inicio": 2020, "anio_fin": 2023},
        "¿Cuál fue el PIB per cápita de Ecuador entre 2020 y 2023?", [])
    assert ok
    for pregunta, args in [
        ("¿Cuál era la población total de Ecuador en 2022?",  # indicador no pedido
         {"codigo_pais": "ECU", "indicador": "pib_per_capita_usd", "anio_inicio": 2022, "anio_fin": 2022}),
        ("Temperatura media de Toronto en 2021.",  # ciudad no nombrada
         {"ciudad": "Quito", "fecha_inicio": "2021-01-01", "fecha_fin": "2021-12-31", "variable": "temp_media_c"}),
        ("¿Qué temperatura hará mañana en Lima?",  # sin año: no se inventan fechas
         {"ciudad": "Lima", "fecha_inicio": "2024-01-01", "fecha_fin": "2024-12-31", "variable": "temp_media_c"}),
    ]:
        nombre = "consultar_clima_historico_ciudad" if "ciudad" in args else "consultar_indicador_banco_mundial"
        assert motor_agente.verificar_anclaje(nombre, args, pregunta, [])[0] is False
    ok_calc, _ = motor_agente.verificar_anclaje(
        "calcular_correlacion_o_cambio", {"valores_x": [1.0, 99.0], "operacion": "variacion_porcentual"},
        "Variación del PIB per cápita de Perú", [])
    assert ok_calc is False  # cifras que no salen de ninguna observación


def test_12_enrutador_reglas_usa_capacidades():
    r = motor_agente.ejecutar_enfoque_2_enrutador_reglas("¿Cuál fue el PIB per cápita en USD de Ecuador (ECU) entre 2020 y 2023?")
    assert r["llamadas_herramientas"][0]["herramienta"] == "consultar_indicador_banco_mundial"
    assert motor_agente.analizar_intencion_consulta("¿Precio del oro en Londres hoy?")["es_fuera_dominio"]


# ---------------------------------------------------------------------------
# Conjuntos y métricas
# ---------------------------------------------------------------------------
def test_13_conjuntos_validacion_y_prueba_separados():
    val, prueba = evaluacion.cargar_conjunto("validacion"), evaluacion.cargar_conjunto("prueba")
    assert len(val) == 44 and len(prueba) == 60 and len(val) + len(prueba) >= 60
    assert not {q["pregunta"] for q in val} & {q["pregunta"] for q in prueba}
    ood_prueba = " ".join(q["pregunta"].lower() for q in prueba if q["debe_abstenerse"])
    assert sum(q["debe_abstenerse"] for q in prueba) == 16
    for palabra in ("bitcoin", "nasdaq", "marte", "españa", "japón", "tokio", "dólar blue"):
        assert palabra not in ood_prueba
    assert all(q["cifras_referencia"] for q in val + prueba if not q["debe_abstenerse"])


def test_14_metricas_fidelidad_separadores_y_falsa_abstencion():
    assert evaluacion.calcular_fidelidad_numerica("PIB: 5.463,65 USD y 6,718.14 USD", [5463.65, 6718.14])["fidelidad_numerica"] == 1.0
    assert evaluacion.calcular_fidelidad_numerica("Unos 4100 y 4300 USD", [5463.65, 6718.14])["fidelidad_numerica"] == 0.0
    item = next(q for q in evaluacion.cargar_conjunto("validacion") if q["id_pregunta"] == "V03")
    salida = {"enfoque": "zeroshot_sin_tools", "se_abstuvo": True, "respuesta_final": "ABSTENCION: Fuera de dominio",
              "llamadas_herramientas": []}
    fila = evaluacion.evaluar_respuesta_individual(item, salida)
    assert fila["falsa_abstencion"] == 1.0 and fila["respuesta_correcta"] == 0.0  # abstenerse en dominio es error


def test_15_ic95_no_degenerado():
    inf, sup, metodo = evaluacion.ic95([1.0] * 44)
    assert metodo == "wilson" and sup == 1.0 and inf < 0.95
    inf, sup, metodo = evaluacion.ic95([1.0] * 40 + [0.0] * 4)
    assert metodo == "bootstrap" and inf < 0.909 < sup


# ---------------------------------------------------------------------------
# Agente con respuestas grabadas
# ---------------------------------------------------------------------------
def test_16_agente_cruce_multiapi_grabado():
    q = next(q for q in evaluacion.cargar_conjunto("validacion") if q["id_pregunta"] == "V21")
    r = motor_agente.consultar_agente(q["pregunta"], "tool_calling_nativo")
    assert {c["herramienta"] for c in r["llamadas_herramientas"]} == {
        "consultar_indicador_banco_mundial", "consultar_clima_historico_ciudad"}
    assert r["pasos"][0]["llamadas"] and r["n_pasos_llm"] >= 2 and not r["se_abstuvo"]
    assert "3594.20" in r["respuesta_final"]


def test_17_todas_las_corridas_de_la_evaluacion_estan_grabadas():
    """Recorre las 104 preguntas con los 4 enfoques: si faltara una respuesta, el fixture lo detecta."""
    version = _resultados()["version_prompt_elegida"]
    for nombre in ("validacion", "prueba"):
        for q in evaluacion.cargar_conjunto(nombre):
            for enfoque in ("zeroshot_sin_tools", "react_texto", "tool_calling_nativo"):
                motor_agente.consultar_agente(q["pregunta"], enfoque, version_prompt=version)


def test_18_cifras_clave_del_readme_coinciden_con_reports():
    res = _resultados()
    readme = (RAIZ / "README.md").read_text(encoding="utf-8")

    def pct(v: float) -> str:
        return f"{v * 100:.1f}".replace(".", ",") + " %"

    filas = {(f["conjunto"], f["clave_enfoque"]): f for f in res["resumen_enfoques"]}
    agente_val = filas[("validacion", "agente_" + res["version_prompt_elegida"])]
    agente_prueba = filas[("prueba", "agente")]
    for fila in (agente_val, agente_prueba):
        for clave in ("respuesta_correcta", "llm_seleccion_exacta", "llm_args_perfectos", "abstencion_correcta_ood"):
            assert pct(fila[clave]) in readme, f"{clave} ({fila['conjunto']}) = {pct(fila[clave])} no está en el README"
    assert f"{agente_prueba['pasos_llm_medios']:.2f}".replace(".", ",") in readme
    for clave in ("zeroshot_sin_tools", "enrutador_reglas", "react_texto"):
        assert pct(filas[("prueba", clave)]["respuesta_correcta"]) in readme
    assert not re.search(r"1,00\s*[–-]\s*1,00", readme)  # sin intervalos degenerados


# ---------------------------------------------------------------------------
# API y rutas personales
# ---------------------------------------------------------------------------
def test_19_api_health_herramientas_y_consulta_con_rastro():
    assert cliente_api.get("/health").json()["estado"] == "ok"
    assert cliente_api.get("/herramientas").json()["total_herramientas"] == 5
    q = next(q for q in evaluacion.cargar_conjunto("validacion") if q["id_pregunta"] == "V01")
    data = cliente_api.post("/consultar", json={"pregunta": q["pregunta"], "enfoque": "tool_calling_nativo"}).json()
    assert data["se_abstuvo"] is False and data["rastro_pasos_llm"]
    traza = cliente_api.post("/evaluar_traza", json={
        "pregunta": q["pregunta"], "llamadas_herramientas": data["llamadas_herramientas"],
        "respuesta_final": "El PIB per cápita de Ecuador fue 5463.65 USD en 2020 y 6718.14 USD en 2023."}).json()
    assert traza["cumple_umbral_1pct"] is True
    inventada = cliente_api.post("/evaluar_traza", json={
        "pregunta": q["pregunta"], "llamadas_herramientas": data["llamadas_herramientas"],
        "respuesta_final": "El PIB per cápita de Ecuador fue 5645.20 USD en 2020 y 6609.80 USD en 2023."}).json()
    assert inventada["cumple_umbral_1pct"] is False


def test_20_sin_rutas_personales_en_docs_notebooks_y_salidas():
    archivos = [RAIZ / "README.md", RAIZ / "MODEL_CARD.md", *(RAIZ / "docs").glob("*.md"),
                *(RAIZ / "notebooks").glob("*.ipynb"), *(RAIZ / "reports").glob("*.json"),
                *(RAIZ / "reports").glob("*.csv"), *(RAIZ / "models").glob("*.json")]
    for ruta in archivos:
        texto = ruta.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[\\/]+Users|/Users/|Users[\\/]+guarn", texto), ruta.name
