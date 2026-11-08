import json

import pytest

from rag_laboral import config
from rag_laboral.evaluar_recuperacion import cargar_preguntas
from rag_laboral.generar import NO_CUBIERTO, Asistente, SalidaLLM
from rag_laboral.llm import LLM, Uso, extraer_json
from rag_laboral.recuperar import Buscador


@pytest.fixture(scope="module")
def buscador():
    return Buscador()


class LLMFalso(LLM):
    """Devuelve una salida fija para probar las reglas que se aplican en código."""

    def __init__(self, salida: dict):
        super().__init__(proveedor="falso")
        self.salida = salida
        self.uso = Uso()

    def chat(self, sistema, mensajes, json_schema=None, max_tokens=700):
        self.ultimo_mensaje = mensajes[-1]["content"]
        return {"texto": json.dumps(self.salida), "segundos": 1.0}


P001 = "¿Cuántos días de vacaciones al año me corresponden?"


def test_cita_inventada_se_elimina_y_se_cuenta(buscador):
    llm = LLMFalso({"articulos_citados": ["CT-69", "CT-999"], "cubierto_por_la_ley": True,
                    "respuesta": "Según el art. 69, quince días.", "confianza": "alta"})
    r = Asistente(buscador, llm).responder(P001)
    assert r.articulos_citados == ["CT-69"] and r.citas_eliminadas == ["CT-999"]
    assert "[CT-69]" in llm.ultimo_mensaje      # el modelo vio el identificador que puede citar


def test_si_el_modelo_dice_que_no_esta_cubierto_se_rechaza(buscador):
    llm = LLMFalso({"articulos_citados": [], "cubierto_por_la_ley": False, "respuesta": "No sé", "confianza": "baja"})
    r = Asistente(buscador, llm).responder(P001)
    assert not r.cubierto_por_la_ley and r.respuesta == NO_CUBIERTO and r.articulos_citados == []


def test_cubierto_sin_citas_validas_tambien_se_rechaza(buscador):
    llm = LLMFalso({"articulos_citados": ["CT-999"], "cubierto_por_la_ley": True, "respuesta": "x", "confianza": "alta"})
    r = Asistente(buscador, llm).responder(P001)
    assert not r.cubierto_por_la_ley and r.citas_eliminadas == ["CT-999"]


def test_umbral_rechaza_sin_llamar_al_llm(buscador):
    llm = LLMFalso({})
    r = Asistente(buscador, llm, umbral_rerank=1.1).responder(P001)   # ningún puntaje supera 1.1
    assert r.rechazo_por_umbral and not hasattr(llm, "ultimo_mensaje")


def test_esquema_pydantic_en_orden_evidencia_primero():
    assert list(SalidaLLM.model_json_schema()["properties"]) == [
        "articulos_citados", "cubierto_por_la_ley", "respuesta", "confianza"]


def test_extraer_json_tolera_texto_alrededor():
    assert extraer_json('Respuesta: {"a": 1} fin') == {"a": 1}


def test_modo_grabado_reproduce_la_evaluacion_real(buscador):
    """La respuesta grabada de qwen2.5:7b para P001 se reproduce idéntica sin Ollama."""
    filas = [json.loads(l) for l in (config.REPORTS / "respuestas_llm.jsonl").read_text(encoding="utf-8").splitlines()]
    guardada = next(f for f in filas if f["id"] == "P001")
    r = Asistente(buscador, LLM(proveedor="grabado")).responder(P001)
    assert r.respuesta == guardada["respuesta"] and r.articulos_citados == guardada["articulos_citados"]
    assert r.desde_grabacion


def test_api_salud_articulo_y_preguntar():
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    assert c.get("/salud").json()["articulos"] >= 600
    a = c.get("/articulo/CT-69").json()
    assert a["epigrafe"] == "Vacaciones anuales"
    assert c.get("/articulo/CT-9999").status_code == 404
    trampa = cargar_preguntas().query("id == 'T003'").pregunta.iloc[0]
    for p in (P001, trampa):
        r = c.post("/preguntar", json={"pregunta": p})
        assert r.status_code == 200
        d = r.json()
        assert all(x["id"] in d["articulos_citados"] for x in d["citas"])
        assert "abogado" in d["aviso"]
    assert c.post("/preguntar", json={"pregunta": "pregunta nunca grabada xyz"}).status_code == 503
