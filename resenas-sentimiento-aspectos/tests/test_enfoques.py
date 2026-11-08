import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from resenas import config
from resenas.enfoque_clasico import ModeloClasico, aspectos_en, etiqueta_estrellas
from resenas.enfoque_embeddings import (CacheEmbeddings, ModeloEmbeddings, analizar_lote, clausula_tiene_cita,
                                        etiquetas_clausulas, preparar_textos)
from resenas.enfoque_llm import analizar
from resenas.esquema import AnalisisResena, esquema_ollama
from resenas.etiquetas import leer_jsonl
from resenas.llm import LLM, Grabadora


# --------------------------------------------------------------------------- clásico
@pytest.mark.parametrize("frase,aspecto", [
    ("The battery dies before lunch", "bateria"), ("Charges my phone really fast", "carga"),
    ("Doesn't fit my iPhone 12", "ajuste"), ("Arrived damaged in a torn box", "envio"),
    ("Customer service sent a replacement", "atencion"), ("Great value for the price", "precio"),
    ("Saved my phone from a drop", "proteccion"), ("Easy to install, no bubbles", "facilidad_uso"),
])
def test_diccionario_de_aspectos(frase, aspecto):
    assert aspecto in aspectos_en(frase)


def test_etiqueta_por_estrellas():
    assert [etiqueta_estrellas(e) for e in range(1, 6)] == ["negativo", "negativo", "neutro", "positivo", "positivo"]


@pytest.fixture(scope="module")
def modelo_clasico_pequeno():
    pos = ["Great case, love it", "Works perfectly, fast charging", "Excellent quality, very happy",
           "Love the color and the price is great", "Perfect fit, highly recommend"] * 8
    neg = ["Terrible, broke in a week", "Does not work, waste of money", "Cheap junk, stopped charging",
           "Awful quality, do not buy", "Cracked after one day, horrible"] * 8
    neu = ["It is ok, nothing special", "Average product, does the job", "Fine I guess, not great not bad"] * 8
    df = pd.DataFrame({"titulo_resena": "", "texto": pos + neg + neu,
                       "estrellas": [5] * len(pos) + [1] * len(neg) + [3] * len(neu)})
    m = ModeloClasico()
    m.vec_pal.min_df = m.vec_car.min_df = 1
    return m.entrenar(df)


def test_clasico_devuelve_el_mismo_esquema_que_el_llm(modelo_clasico_pequeno):
    r = modelo_clasico_pequeno.analizar("Love it", "The color is great but it broke in a week. Terrible.")
    AnalisisResena.model_validate(r)
    d = {m["aspecto"]: m["polaridad"] for m in r["aspectos"]}
    assert d.get("diseno") == "positivo" and d.get("calidad") == "negativo"


# --------------------------------------------------------------------------- embeddings
def _embedding_falso(textos):
    """Vector determinista por palabra (bolsa de palabras con hash): sirve para probar sin Ollama."""
    salida = []
    for t in textos:
        v = np.zeros(1024, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,!").encode()).hexdigest(), 16) % 1024] += 1
        salida.append(v / max(np.linalg.norm(v), 1e-9))
    return np.array(salida)


def test_clausula_corresponde_a_cita():
    assert clausula_tiene_cita("but it broke in a week.", "it broke in a week")
    assert clausula_tiene_cita("It broke", "but it broke in a week")
    assert not clausula_tiene_cita("Shipping was fast", "it broke in a week")


def test_cache_embeddings_guarda_y_reanuda(tmp_path):
    llamadas = []

    def funcion(textos):
        llamadas.append(len(textos))
        return _embedding_falso(textos)

    c = CacheEmbeddings(tmp_path / "e.npz", funcion)
    assert c.embeber_faltantes(["a b", "c d", "a b"]) == 2
    c2 = CacheEmbeddings(tmp_path / "e.npz", funcion)
    assert c2.embeber_faltantes(["a b", "e f"]) == 1          # solo lo nuevo
    assert llamadas == [2, 1]
    assert c2.matriz(["a b"]).shape == (1, 1024)


def test_modelo_embeddings_aprende_aspectos(tmp_path):
    resenas = [{"id": f"r{i}", "titulo_resena": "", "texto": t} for i, t in enumerate([
        "The battery lasts all day. The color is ugly.", "Battery dies fast. Looks beautiful.",
        "Great battery life. Pretty color.", "Battery drains quickly. The design is nice."] * 5)]
    etq = {}
    for r in resenas:
        a, b = r["texto"].split(". ")
        pol_a = "negativo" if any(w in a for w in ("dies", "drains")) else "positivo"
        pol_b = "negativo" if "ugly" in b else "positivo"
        etq[r["id"]] = {"aspectos": [{"aspecto": "bateria", "polaridad": pol_a, "cita": a},
                                     {"aspecto": "diseno", "polaridad": pol_b, "cita": b}]}
    cache = CacheEmbeddings(tmp_path / "e.npz", _embedding_falso)
    cls, completos = preparar_textos(resenas)
    cache.embeber_faltantes([c for cs in cls for c in cs] + completos)
    textos, Y, pols = etiquetas_clausulas(resenas, etq)
    X = cache.matriz(textos)
    m = ModeloEmbeddings(umbral=0.5).entrenar(X, Y, pols, X[:4], ["positivo", "negativo"] * 2,
                                              cache.matriz(completos), ["positivo", "negativo"] * 10)
    res = analizar_lote(m, cache, resenas[:2])
    assert {x["aspecto"] for x in res[0]["aspectos"]} == {"bateria", "diseno"}
    AnalisisResena.model_validate(res[0])


# --------------------------------------------------------------------------- LLM
def test_llm_salida_estructurada_valida(llm_falso):
    llm = llm_falso([{"aspectos": [{"cita": "broke in a week", "aspecto": "calidad", "polaridad": "negativo"},
                                   {"cita": "broke", "aspecto": "calidad", "polaridad": "positivo"},
                                   {"cita": "ships from mars", "aspecto": "envio", "polaridad": "positivo"}],
                      "sentimiento_global": "negativo"}])
    r = analizar("Bad", "It broke in a week. Shipping was fast.", llm)
    assert [m["aspecto"] for m in r["aspectos"]] == ["calidad", "envio"]     # el repetido se descarta
    assert r["citas_no_literales"] == 1                                      # "ships from mars" no está
    assert r["error"] is None and r["segundos"] == 1.5
    enviado = llm.recibidos[0]
    assert enviado["esquema"] == esquema_ollama()
    assert "It broke in a week" in enviado["mensajes"][0]["content"]


def test_llm_respuesta_invalida_no_rompe(llm_falso):
    r = analizar("x", "y", llm_falso(['{"aspectos": [{"aspecto": "camara", "polaridad": "positivo"}]}']))
    assert r["error"] and r["aspectos"] == [] and r["sentimiento_global"] == "neutro"


def test_esquema_ollama_usa_el_catalogo():
    e = esquema_ollama()
    item = e["properties"]["aspectos"]["items"]
    assert item["properties"]["aspecto"]["enum"] == list(config.ASPECTOS)
    assert list(item["properties"])[0] == "cita"     # la cita se genera antes que la etiqueta


def test_modo_grabado_reproduce_la_evaluacion(muestra_pequena):
    """Las predicciones del LLM en reports/ se reproducen desde data/grabacion_llm.jsonl, sin Ollama (con 5 reseñas
    de tests/datos: el resto del texto no se versiona)."""
    preds = leer_jsonl(config.DIR_REPORTES / "predicciones_llm.jsonl")[:5]
    muestra = muestra_pequena
    Grabadora._cache = None
    llm = LLM(proveedor="grabado")
    for p in preds:
        r = analizar(muestra[p["id"]]["titulo_resena"], muestra[p["id"]]["texto"], llm)
        assert r["aspectos"] == p["aspectos"] and r["sentimiento_global"] == p["sentimiento_global"]
    assert llm.uso.desde_grabacion == len(preds)


def test_modo_grabado_avisa_si_no_hay_grabacion():
    from resenas.llm import ErrorLLM
    with pytest.raises(ErrorLLM, match="Modo demo"):
        analizar("Reseña nueva", "Nunca vista por el modelo " + json.dumps(list(range(5))), LLM(proveedor="grabado"))
