"""Chequeo de dirección de comparaciones: cifras correctas con lectura engañosa (casos reales del LLM)."""
from kpis.comparaciones import revisar


def _veredictos(texto, pais, anio=2024):
    return [(c["tipo"], c["indicador"], c["correcta"]) for c in revisar(texto, pais, anio)]


def test_cifra_correcta_pero_comparacion_al_reves():
    # Honduras 2024: inflación 4,6 % y mediana 4,3 %: no está «por debajo»
    texto = "La inflación en Honduras se redujo a 4,6 %, y se mantuvo por debajo de la mediana regional (4,3 %)."
    v = _veredictos(texto, "HND")
    assert ("frente_a_region", "inflacion", False) in v
    assert ("frente_al_anio_anterior", "inflacion", True) in v


def test_indicador_de_la_oracion_previa_a_la_coma():
    texto = "Las remesas representan un 24,3 % del PIB, superando la mediana regional (2,7 %)."
    assert _veredictos(texto, "SLV") == [("frente_a_region", "remesas", True)]


def test_deficit_que_se_amplia():
    # El Salvador 2024: saldo de cuenta corriente -2,8 %, cae 1,0 pp: el déficit sí se amplió
    assert _veredictos("El déficit en la cuenta corriente se amplió a -2,8 %.", "SLV") == \
        [("frente_al_anio_anterior", "cuenta_corriente", True)]


def test_pib_en_dolares_presentado_como_crecimiento():
    # Chile 2023: el PIB en USD corrientes subió 11,5 %, pero el PIB real creció 0,7 %
    v = revisar("El PIB de Chile creció un 11,5 % en 2023.", "CHL", 2023)
    assert [(c["tipo"], c["correcta"]) for c in v] == [("pib_nominal_como_crecimiento", False)]
    assert revisar("El PIB en dólares corrientes creció un 11,5 % en 2023.", "CHL", 2023) == []


def test_no_opina_sin_indicador_claro():
    assert revisar("La situación mejoró frente a la región.", "ECU", 2024) == []
