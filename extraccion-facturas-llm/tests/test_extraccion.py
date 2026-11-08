import json
import os

import pytest

from facturas import config
from facturas.extractor import extraer_de_texto, extraer_pdf
from facturas.llm import LLM, Grabadora
from facturas.metricas import comparar, comparar_items
from facturas.regex_base import extraer_regex
from facturas.reparaciones import limpiar_descripcion, reparar_dict
from facturas.texto import extraer_texto, ocr_disponible


def test_pypdf_lee_las_facturas_digitales(muestras):
    for m in muestras:
        if m["condicion"] == "digital":
            t = extraer_texto(m["pdf"])
            assert t.metodo == "pypdf"
            assert m["verdad"]["ruc_emisor"] in t.texto
            assert m["verdad"]["numero_factura"] in t.texto


@pytest.mark.skipif(not ocr_disponible(), reason="OCR de Windows no disponible en este sistema")
def test_ocr_lee_escaneadas_y_endereza_giradas(muestras):
    escaneadas = [m for m in muestras if m["condicion"] != "digital"]
    assert escaneadas
    for m in escaneadas:
        t = extraer_texto(m["pdf"])
        assert t.metodo == "ocr"
        assert t.rotacion == (360 - int(m["rotacion"])) % 360      # se gira lo que falta para enderezar
        assert m["verdad"]["ruc_emisor"] in t.texto.replace(" ", "")


def test_regex_perfecta_en_digitales_de_plantillas_vistas(muestras):
    for m in muestras:
        if m["condicion"] == "digital" and not m["plantilla"].startswith("E"):
            fila = comparar(extraer_regex(extraer_texto(m["pdf"]).texto), m["verdad"])
            assert fila["factura_perfecta"], (m["id"], fila)


def test_regex_falla_en_la_plantilla_nunca_vista(muestras):
    m = next(m for m in muestras if m["plantilla"] == "E_compacta" and m["condicion"] == "digital")
    fila = comparar(extraer_regex(extraer_texto(m["pdf"]).texto), m["verdad"])
    assert not fila["factura_perfecta"] and fila["items_f1"] == 0


def test_reintento_corrige_con_los_errores_de_validacion(llm_falso, factura_ok):
    mala = {**factura_ok, "total": factura_ok["total"] + 10}
    llm = llm_falso([mala, factura_ok])
    r = extraer_de_texto("texto", llm, max_reintentos=1)
    assert r.estado == "valida" and r.reintentos == 1
    correccion = llm.recibidos[1][-1]["content"]
    assert "no pasa estas validaciones" in correccion and "el total es" in correccion
    assert llm.recibidos[1][-2]["role"] == "assistant"


def test_sin_arreglo_queda_para_revision_y_se_guarda_el_mejor(llm_falso, factura_ok):
    peor = {**factura_ok, "total": factura_ok["total"] + 10, "ruc_emisor": "123"}
    mala = {**factura_ok, "total": factura_ok["total"] + 10}
    r = extraer_de_texto("texto", llm_falso([mala, peor]), max_reintentos=1)
    assert r.estado == "revisar"
    assert r.factura.ruc_emisor == factura_ok["ruc_emisor"]          # se queda el intento con menos errores


def test_json_roto_no_revienta(llm_falso):
    r = extraer_de_texto("texto", llm_falso(["esto no es json"]), max_reintentos=1)
    assert r.estado == "error" and r.factura is None and len(r.intentos) == 2


def test_items_f1():
    v = [{"descripcion": "Cemento gris saco 50 kg", "cantidad": 2, "precio_unitario": 8, "total": 16},
         {"descripcion": "Clavos", "cantidad": 1, "precio_unitario": 2, "total": 2}]
    p = [{"descripcion": "CEMENTO GRIS SACO 50 KG", "cantidad": 2, "precio_unitario": 8, "total": 16},
         {"descripcion": "Clavos", "cantidad": 1, "precio_unitario": 2.5, "total": 2.5},
         {"descripcion": "Inventado", "cantidad": 1, "precio_unitario": 1, "total": 1}]
    r = comparar_items(p, v)
    assert r["items_ok"] == 1 and r["items_precision"] == pytest.approx(1 / 3) and r["items_recall"] == 0.5


def test_null_correcto_solo_si_el_dato_no_existe(factura_ok):
    sin_aut = {**factura_ok, "numero_autorizacion": None}
    assert comparar(sin_aut, sin_aut)["numero_autorizacion"]
    assert not comparar(factura_ok, sin_aut)["numero_autorizacion"]      # inventar un dato ausente es error
    assert not comparar(sin_aut, factura_ok)["numero_autorizacion"]


@pytest.mark.skipif(not config.GRABACION.exists(), reason="sin data/grabacion_llm.jsonl")
def test_modo_grabado_reproduce_la_evaluacion_sin_ollama(muestras, monkeypatch):
    """Las 10 muestras están en la muestra de evaluación: sus respuestas del LLM están grabadas."""
    monkeypatch.setenv("FACTURAS_GRABAR", "0")
    Grabadora._cache = None
    pred = {}
    ruta = config.REPORTES / "predicciones_llm.jsonl"
    if ruta.exists():
        pred = {json.loads(ln)["id"]: json.loads(ln) for ln in ruta.read_text(encoding="utf-8").splitlines()}
    hechas = [m for m in muestras if m["id"] in pred]
    if not hechas or (not ocr_disponible() and any(m["condicion"] != "digital" for m in hechas)):
        pytest.skip("muestras sin grabar o sin OCR")
    for m in hechas:
        r = extraer_pdf(m["pdf"], LLM(proveedor="grabado"))
        assert r.estado == pred[m["id"]]["estado"], m["id"]
        assert json.loads(r.factura.model_dump_json()) == reparar_dict(pred[m["id"]]["factura"])
    os.environ.pop("FACTURAS_GRABAR", None)


def test_reparaciones(factura_ok):
    assert limpiar_descripcion("Humita— 1 x $1.40") == "Humita"
    assert limpiar_descripcion("LLAVE DE PASO 1/2 PULGADA 25 x 6.84") == "LLAVE DE PASO 1/2 PULGADA"
    assert limpiar_descripcion("Varilla corrugada 12 mm x 12 m") == "Varilla corrugada 12 mm x 12 m"
    assert limpiar_descripcion("Tubo PVC 1/2 pulgada x 6 m") == "Tubo PVC 1/2 pulgada x 6 m"
    cf = reparar_dict({**factura_ok, "cliente_nombre": None, "cliente_identificacion": "9999999999999"})
    assert cf["cliente_nombre"] == "CONSUMIDOR FINAL"


def _fila(campos_ok, ok, pred, real, perfecta):
    return {"campos_ok": campos_ok, "items_ok": ok, "items_pred": pred, "items_real": real, "factura_perfecta": perfecta}


def test_ic95_contiene_el_valor_y_wilson_no_colapsa_en_cero():
    from facturas.esquema import CAMPOS
    from facturas.metricas import ic95, wilson
    filas = [_fila(len(CAMPOS), 3, 3, 3, True)] * 7 + [_fila(len(CAMPOS) - 2, 1, 2, 3, False)] * 3
    ic = ic95(filas)
    exactitud = sum(f["campos_ok"] for f in filas) / (len(filas) * len(CAMPOS))
    assert ic["exactitud_campos"][0] <= exactitud <= ic["exactitud_campos"][1]
    assert ic == ic95(filas)                                     # semilla fija
    assert ic95([dict(f, factura_perfecta="True" if f["factura_perfecta"] else "False") for f in filas]) == ic
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35                           # 0 de 10: el techo no es 0
    assert wilson(7, 10)[0] < 0.7 < wilson(7, 10)[1]


def test_evaluacion_trae_ic_y_el_readme_los_cita():
    informe = json.loads((config.REPORTES / "evaluacion.json").read_text(encoding="utf-8"))
    for m, bloque in informe["muestra_50"].items():
        g = bloque["global"]
        for k in ("exactitud_campos", "items_f1", "facturas_perfectas"):
            lo, hi = g["ic95"][k]
            assert lo <= g[k] <= hi, (m, k)
    llm = informe["muestra_50"]["llm"]["global"]
    readme = (config.RAIZ / "README.md").read_text(encoding="utf-8")
    lo, hi = llm["ic95"]["exactitud_campos"]
    assert f"[{100 * lo:.1f}–{100 * hi:.1f}]".replace(".", ",") in readme
    # la cascada llama al LLM en el 46 % de la muestra y en el 30 % de las 200 (no confundirlos)
    assert informe["costo"]["cascada_usa_llm_pct_muestra"] == 0.46 and "46 %" in readme
    assert round(informe["costo"]["cascada_usa_llm_pct_200"], 2) == 0.3
