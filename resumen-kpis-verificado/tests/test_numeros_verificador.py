"""Extractor y verificador de cifras: casos difíciles de formato, unidades y contexto."""
import pytest

from kpis import capa_kpis
from kpis.esquema import Referencia
from kpis.numeros import extraer, interpretar_numero
from kpis.verificador import enmascarar_definicionales, verificar_texto


def _uno(texto):
    cifras = extraer(texto)
    assert len(cifras) == 1, [c.texto for c in cifras]
    return cifras[0]


@pytest.mark.parametrize("texto,valor,unidad,dec", [
    ("la inflación fue de 3,5 %", 3.5, "%", 1),
    ("una contracción del 3.5%", 3.5, "%", 1),
    ("bajó −1,9 % en el año", -1.9, "%", 1),
    ("subió 2,6 puntos porcentuales", 2.6, "pp", 1),
    ("subió 2,6 p. p.", 2.6, "pp", 1),
    ("un PIB per cápita de 6.827 USD", 6827.0, "USD", 0),
    ("un PIB per cápita de USD 6.827", 6827.0, "USD", 0),
    ("6,718.14 dólares", 6718.14, "USD", 2),
    ("1.234,5 USD", 1234.5, "USD", 1),
    ("123,8 mil millones de USD", 123.8e9, "USD", 1),
    ("121.000 millones de dólares", 121e9, "USD", 0),
    ("18,1 millones de habitantes", 18.1e6, "personas", 1),
    ("ocupa el puesto 3 de 18", 3.0, "puesto", 0),
    ("el 2º puesto", 2.0, "puesto", 0),
])
def test_extraer_formatos(texto, valor, unidad, dec):
    c = _uno(texto)
    assert c.valor == pytest.approx(valor)
    assert c.unidad == unidad
    assert c.decimales == dec


def test_ambiguedades_devuelven_candidatos():
    assert interpretar_numero("6,718") == ([6.718, 6718.0], 3)   # español: decimal; inglés: miles
    assert interpretar_numero("6.718") == ([6718.0, 6.718], 0)   # español: miles; inglés: decimal
    c = _uno("un PIB de US$ 1,2 billones")
    assert c.candidatos == pytest.approx([1.2e12, 1.2e9])         # billón español o anglicismo
    assert "escala_ambigua" in c.marcas


def test_ignora_anios_listas_y_totales_de_ranking():
    assert extraer("En 2024 y en 2023 no hubo cambios.") == []
    assert [c.texto for c in extraer("1. Hallazgo: desempleo de 3,5 %")] == ["3,5"]
    assert [c.valor for c in extraer("puesto 18 de 18")] == [18.0]


def test_ordinales():
    c = _uno("Ecuador ocupa el tercer lugar en remesas")
    assert (c.valor, c.unidad) == (3.0, "puesto")
    # Casos reales del LLM en validación: «18ª posición» y «4º mayor valor»
    assert [(c.valor, c.unidad) for c in extraer("mejorando su posición a la 18ª posición")] == [(18.0, "puesto")]
    assert [(c.valor, c.unidad) for c in extraer("es el 4º mayor valor en la región")] == [(4.0, "puesto")]


def test_cifras_definicionales_no_son_afirmaciones():
    texto = "Pobreza extrema (USD 3,00 al día, PPA 2021) y Gini (0–100)"
    assert extraer(enmascarar_definicionales(texto)) == []


def _estados(texto, pais="ECU", anio=2024, refs=None):
    return [(c.texto, c.estado) for c in verificar_texto(texto, pais, anio, refs)]


def test_verifica_cifras_correctas_con_redondeo_y_signo():
    est = _estados("En 2024 la economía de Ecuador se contrajo 1,9 %, frente a una mediana regional de 3,4 %.")
    assert est == [("1,9", "correcta"), ("3,4", "correcta")]


def test_detecta_cifra_incorrecta_con_valor_real():
    r = verificar_texto("La inflación fue de 2,2 % en 2024.", "ECU", 2024)
    assert r[0].estado == "incorrecta"
    assert r[0].real_formateado == "1,5 %"


def test_cifra_sin_fuente():
    assert _estados("Se espera un repunte de 4,2 % el próximo año.") == [("4,2", "sin_fuente")]


def test_puntos_porcentuales_y_porcentaje():
    assert _estados("La inflación bajó a 1,5 %, 0,7 puntos porcentuales menos que en 2023.") == \
        [("1,5", "correcta"), ("0,7", "correcta")]
    r = verificar_texto("Las remesas crecieron 0,8 % en 2024.", "ECU", 2024)
    assert r[0].estado == "correcta" and "pp_escrito_como_%" in r[0].marcas


def test_miles_y_millones():
    assert _estados("El PIB alcanzó 123,8 mil millones de USD y el PIB per cápita 6.827 USD.") == \
        [("123,8 mil millones", "correcta"), ("6.827", "correcta")]
    assert _estados("La población llegó a 18,1 millones de habitantes.")[0][1] == "correcta"
    # 1.238 mil millones no es 123,8 mil millones (error de escala)
    assert _estados("El PIB alcanzó 1.238 mil millones de USD.")[0][1] == "incorrecta"


def test_coma_inglesa_de_miles():
    """Caso real del LLM en prueba: «5,503 USD» (PIB per cápita de El Salvador, 5.503 USD)."""
    assert _estados("El PIB per cápita de El Salvador llegó a 5,503 USD.", "SLV", 2024) == [("5,503", "correcta")]


def test_ranking_en_ambos_sentidos():
    assert _estados("Ecuador tuvo el crecimiento más bajo de la región (puesto 18 de 18).")[0][1] == "correcta"
    assert _estados("La IED fue de 0,4 % del PIB, el tercer puesto más bajo.") == \
        [("0,4", "correcta"), ("tercer puesto", "correcta")]


def test_referencias_estructuradas_mandan():
    ref = Referencia(indicador="desempleo", pais="ECU", anio=2024, metrica="valor", valor=3.5)
    assert _estados("El indicador laboral quedó en 3,5 %.", refs=[ref]) == [("3,5", "correcta")]


def test_hechos_de_la_capa():
    h = capa_kpis.buscar_hecho("crecimiento", "ECU", 2024, "valor")
    assert h.valor == pytest.approx(-1.944, abs=1e-3)
    rk = capa_kpis.buscar_hecho("crecimiento", "ECU", 2024, "ranking")
    assert (rk.valor, rk.n) == (18.0, 18)
