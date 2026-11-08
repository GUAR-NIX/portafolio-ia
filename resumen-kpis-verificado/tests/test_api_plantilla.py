"""API, plantilla determinista y esquema estructurado (sin Ollama)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from kpis import config, generadores
from kpis.esquema import Informe

cliente = TestClient(app)


def test_api_salud_y_kpis():
    assert cliente.get("/salud").json()["paises"] == 18
    r = cliente.get("/kpis/ECU/2024").json()
    assert any(h["id"] == "crecimiento|ECU|2024|valor" for h in r["hechos"])
    assert cliente.get("/kpis/XXX/2024").status_code == 404


def test_api_verificar_texto_libre():
    r = cliente.post("/verificar", json={"texto": "La inflación de Ecuador fue de 9,9 % en 2024.", "pais": "ECU",
                                         "anio": 2024}).json()
    assert r["n_incorrecta"] == 1 and r["tasa_alucinacion"] == 1.0


def test_api_informe_plantilla():
    r = cliente.get("/informe/ECU/2024", params={"enfoque": "plantilla"}).json()
    assert r["verificacion"]["aprobado"]


@pytest.mark.parametrize("pais,anio", config.VALIDACION + config.PRUEBA)
def test_plantilla_no_alucina(pais, anio):
    """Control de cordura del verificador: la plantilla solo copia cifras de la capa, así que todo debe salir verde."""
    r = generadores.generar_plantilla(pais, anio)
    v = r["verificacion"]
    assert v["n_cifras"] > 10
    assert v["n_incorrecta"] == 0 and v["n_sin_fuente"] == 0
    assert v["n_referencias_ok"] == v["n_referencias"]
    Informe.model_validate(r["informe"])


def test_esquema_exige_entre_3_y_5_hallazgos():
    a = {"texto": "x", "referencias": []}
    with pytest.raises(Exception):
        Informe.model_validate({"titular": a, "hallazgos": [a, a], "riesgos": [a], "recomendaciones": [a]})
    Informe.model_validate({"titular": a, "hallazgos": [a, a, a], "riesgos": [a], "recomendaciones": [a]})
