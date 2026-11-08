import pytest

from rag_laboral.evaluar_recuperacion import cargar_preguntas
from rag_laboral.recuperar import Buscador, Resultado
from rag_laboral.texto import expandir, tokens, ventana_relevante


@pytest.fixture(scope="module")
def buscador():
    return Buscador()


def test_tokens_normaliza_tildes_plurales_y_vacias():
    assert tokens("Las Remuneraciones del trabajador") == ["remuneracion", "trabajador"]
    assert "vacacion" in tokens("¿Cuántas vacaciones tengo?")


def test_sinonimos_coloquial_a_legal():
    assert "decimotercera remuneración" in expandir("¿Cuándo pagan el décimo tercero?")
    assert "despido intempestivo" in expandir("Me botaron del trabajo")
    assert expandir("¿Qué dice la ley?") == "¿Qué dice la ley?"


def test_sinonimos_v2_corrige_falsos_positivos_de_v1():
    # «renunciar a mis derechos» no es renunciar al trabajo (error encontrado en el análisis)
    assert "desahucio" in expandir("Puedo renunciar a mis derechos?", "v1")
    assert "desahucio" not in expandir("Puedo renunciar a mis derechos?", "v2")
    assert "décima tercera" in expandir("¿Cuándo pagan el décimo tercero?", "v2")


def test_ventana_relevante_de_articulo_largo():
    texto = "Son obligaciones del empleador:\n" + "\n".join(f"{i}. Obligación genérica número {i};" for i in range(1, 40))
    texto += "\n40. Inscribir a los trabajadores en el Instituto Ecuatoriano de Seguridad Social desde el primer día;"
    v = ventana_relevante(texto, "¿Desde cuándo me afilian al Seguro Social?", 300)
    assert v.startswith("Son obligaciones") and "40. Inscribir" in v and len(v) < 400


def test_rrf_premia_lo_que_aparece_en_ambas_listas():
    a = [Resultado("A", 9, 1), Resultado("B", 8, 2), Resultado("C", 7, 3)]
    b = [Resultado("C", 0.9, 1), Resultado("D", 0.8, 2), Resultado("A", 0.7, 3)]
    fusion = [r.id for r in Buscador.rrf([a, b], 4)]
    assert fusion[:2] == ["A", "C"] and set(fusion) == {"A", "B", "C", "D"}


def test_bm25_encuentra_vacaciones(buscador):
    ids = [h.id for h in buscador.buscar("vacaciones anuales quince días", "bm25", 5)]
    assert "CT-69" in ids


def test_las_cuatro_variantes_corren_sin_ollama_con_lo_grabado(buscador):
    """Los embeddings de las consultas y los puntajes del re-ranker están en data/indice/."""
    p = cargar_preguntas().query("id == 'P001'").pregunta.iloc[0]
    for metodo in ("bm25", "vectorial", "hibrido", "rerank"):
        hits = buscador.buscar(p, metodo, 5)
        assert len(hits) == 5 and all(h.id in buscador.por_id for h in hits)
    assert buscador.buscar(p, "rerank", 5)[0].id == "CT-69"


def test_consulta_no_grabada_falla_con_mensaje_claro(buscador):
    with pytest.raises(RuntimeError, match="grabado"):
        buscador.buscar("pregunta que nadie grabó jamás 123", "vectorial", 5)
