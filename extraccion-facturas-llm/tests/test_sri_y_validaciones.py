import random
from datetime import date

from facturas import sri
from facturas.esquema import Factura
from facturas.generar import generar_verdad
from facturas.validaciones import errores, validar


def test_cedula_y_ruc_digito_verificador():
    rng = random.Random(1)
    for _ in range(200):
        ced = sri.generar_cedula(rng)
        assert sri.cedula_valida(ced)
        malo = ced[:9] + str((int(ced[9]) + 1) % 10)
        assert not sri.cedula_valida(malo)
    for tipo in ("natural", "privada", "publica"):
        for _ in range(50):
            ruc = sri.generar_ruc(rng, tipo)
            assert sri.tipo_ruc(ruc) == tipo, ruc
    assert sri.ruc_valido("1790011674001")          # forma de un RUC de sociedad (dígito 4 por módulo 11)
    assert not sri.ruc_valido("1790011675001")
    assert sri.identificacion_valida(sri.CONSUMIDOR_FINAL)


def test_clave_de_acceso_se_arma_y_desarma():
    clave = sri.armar_clave(date(2024, 5, 17), "0992345678001", "001-002-000000123", "12345678")
    assert len(clave) == 49 and sri.clave_valida(clave)
    partes = sri.desarmar_clave(clave)
    assert partes["fecha"] == "2024-05-17" and partes["ruc"] == "0992345678001"
    assert partes["numero_factura"] == "001-002-000000123" and partes["tipo"] == "01"
    otra = clave[:20] + str((int(clave[20]) + 1) % 10) + clave[21:]
    assert not sri.clave_valida(otra)


def test_tasa_iva_cambia_en_abril_2024():
    assert sri.tasa_iva(date(2024, 3, 31)) == 0.12
    assert sri.tasa_iva(date(2024, 4, 1)) == 0.15


def test_las_facturas_generadas_cuadran():
    rng = random.Random(7)
    for i in range(300):
        f, _ = generar_verdad(rng, i)
        assert not errores(validar(Factura.model_validate(f))), f


def test_validaciones_detectan_cada_error(factura_ok):
    assert not errores(validar(Factura.model_validate(factura_ok)))

    def reglas(cambios):
        return {p.regla for p in errores(validar(Factura.model_validate({**factura_ok, **cambios})))}

    assert "ruc_digito" in reglas({"ruc_emisor": factura_ok["ruc_emisor"][:-4] + "9" + factura_ok["ruc_emisor"][-3:]})
    assert "total_cuadra" in reglas({"total": factura_ok["total"] + 1})
    assert "iva_tasa" in reglas({"iva": round(factura_ok["iva"] * 1.1, 2), "total": round(
        factura_ok["total"] + factura_ok["iva"] * 0.1, 2)})
    assert "items_suman" in reglas({"items": factura_ok["items"][:-1]})
    assert "autorizacion_digito" in reglas({"numero_autorizacion": factura_ok["numero_autorizacion"][:-1]})
    assert "autorizacion_fecha" in reglas({"fecha_emision": "2023-01-02"})
    assert "numero_formato" in reglas({"numero_factura": "1-1"})
    linea = dict(factura_ok["items"][0], total=factura_ok["items"][0]["total"] + 5)
    assert "item_linea" in reglas({"items": [linea] + factura_ok["items"][1:]})


def test_faltantes_son_advertencia_no_error(factura_ok):
    p = validar(Factura.model_validate({**factura_ok, "numero_autorizacion": None, "cliente_identificacion": None}))
    assert not errores(p)
    assert [x.regla for x in p] == ["autorizacion_presente"]


def test_esquema_normaliza(factura_ok):
    f = Factura.model_validate({**factura_ok, "descuento": -3.14, "numero_factura": "1-2-345",
                                "ruc_emisor": "R.U.C. " + factura_ok["ruc_emisor"]})
    assert f.descuento == 3.14
    assert f.numero_factura == "001-002-000000345"
    assert f.ruc_emisor == factura_ok["ruc_emisor"]
