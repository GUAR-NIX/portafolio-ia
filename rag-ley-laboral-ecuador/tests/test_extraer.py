import collections

from rag_laboral.extraer import RE_ART, limpiar_lineas, reparar_espacios, trocear, unir_parrafo

PAGINAS = [
    """Página 4 de 159
TÍTULO I
DEL CONTRATO INDIVIDUAL DE TRABAJO
Capítulo V
De la duración máxima de la jornada
Parágrafo 3ro.
De las vacaciones
Art. 69
Vacaciones anuales. - Todo trabajador tendrá derecho a gozar anualmente de un período
ininterrumpido de quince días de descanso.
Nota:
Artículo sustituido por artículo 9 de Ley No. 0, publicada en Registro Oficial 483 .
JURISPRUDENCIA:
Gaceta Judicial, VACACIONES, 12-may-1990""",
    """Página 5 de 159
Art. 42
Obligaciones del empleador. - Son obligaciones del empleador:
1. Pagar las cantidades que correspondan;
Nota:
Numeral 2 sustituido por Ley No. 0, publicada en Registro Oficial 483 .
2. Instalar las fábricas conforme a las disposiciones legales;
Art. (...)
Del teletrabajo. - El teletrabajo es una forma de organización laboral.""",
]


def test_patron_de_articulo():
    assert RE_ART.match("Art. 69").group(1) == "69"
    assert RE_ART.match("Art. 16.1").group(1) == "16.1"
    assert RE_ART.match("Art. 33.- El trabajo es un derecho").group(2) == "El trabajo es un derecho"
    assert RE_ART.match("Artículo sustituido por Ley 39") is None


def test_troceo_por_articulo_con_ubicacion_y_notas_aparte():
    arts = trocear(limpiar_lineas(PAGINAS), "Código del Trabajo", "CT")
    ids = [a["id"] for a in arts]
    assert ids == ["CT-69", "CT-42", "CT-42.inn1"]
    a69 = arts[0]
    assert a69["epigrafe"] == "Vacaciones anuales"
    assert "quince días de descanso" in a69["texto"]
    assert "Página" not in a69["texto"] and "Gaceta" not in a69["texto"] and "sustituido" not in a69["texto"]
    assert "JURISPRUDENCIA" in a69["notas"]
    assert a69["seccion"] == "Parágrafo 3ro.: De las vacaciones"
    assert a69["titulo"].startswith("TÍTULO I: DEL CONTRATO")


def test_nota_en_medio_de_una_lista_no_se_traga_el_resto():
    a42 = trocear(limpiar_lineas(PAGINAS), "Código del Trabajo", "CT")[1]
    assert "1. Pagar" in a42["texto"] and "2. Instalar las fábricas" in a42["texto"]
    assert "Numeral 2 sustituido" in a42["notas"]


def test_reparar_palabras_partidas_por_pypdf():
    frec = collections.Counter({"trabajador": 50, "traba": 0, "jador": 1, "la": 900, "a": 800, "l": 1, "el": 700})
    assert reparar_espacios("pagado al traba jador", frec) == "pagado al trabajador"
    assert reparar_espacios("mediante l a presentación", frec) == "mediante la presentación"
    assert reparar_espacios("a la empresa", frec) == "a la empresa"  # dos palabras frecuentes: no se unen


def test_unir_parrafo_respeta_numerales():
    t = unir_parrafo(["Son obligaciones:", "1. Pagar a tiempo y", "en dinero;", "2. Respetar"])
    assert t == "Son obligaciones:\n1. Pagar a tiempo y en dinero;\n2. Respetar"
