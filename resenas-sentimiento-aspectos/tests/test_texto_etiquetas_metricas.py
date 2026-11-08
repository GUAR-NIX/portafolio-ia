import json

import pandas as pd
import pytest

from resenas import config
from resenas.muestra import CAMPOS_INDICE, huella, reconstruir
from resenas.esquema import AnalisisResena
from resenas.etiquetas import como_dict, leer_jsonl, parsear_linea
from resenas.metricas import acuerdo, evaluar, kappa
from resenas.texto import cita_en_texto, clausulas, texto_completo


# --------------------------------------------------------------------------- texto
def test_clausulas_separa_oraciones_y_contrastes():
    cs = clausulas("Love the color, but it broke in a week. Shipping was fast!\nGreat price")
    assert cs == ["Love the color", "but it broke in a week.", "Shipping was fast!", "Great price"]


def test_texto_completo_une_titulo_y_cuerpo():
    assert texto_completo("Great case", "Fits well.") == "Great case.\nFits well."
    assert texto_completo("", "Solo cuerpo") == "Solo cuerpo"


def test_cita_tolera_mayusculas_espacios_y_comillas():
    texto = "It doesn’t   fit my phone.  Arrived FAST"
    assert cita_en_texto("it doesn't fit my phone", texto)
    assert cita_en_texto("arrived fast.", texto)
    assert not cita_en_texto("arrived late", texto)


# --------------------------------------------------------------------------- etiquetas
def test_formato_compacto():
    d = parsear_linea("R000001 | - | calidad-: broke in a week ; envio+: arrived fast")
    assert d["sentimiento_global"] == "negativo"
    assert d["aspectos"] == [{"aspecto": "calidad", "polaridad": "negativo", "cita": "broke in a week"},
                             {"aspecto": "envio", "polaridad": "positivo", "cita": "arrived fast"}]
    assert parsear_linea("R000002 | + | ")["aspectos"] == []


@pytest.mark.parametrize("linea", ["R1 | + | bateriaX+: algo", "R1 | + | bateria*: algo",
                                   "R1 | + | carga+: a ; carga-: b"])
def test_formato_compacto_rechaza_errores(linea):
    with pytest.raises(ValueError):
        parsear_linea(linea)


RUTAS_ETIQUETAS = [(config.ETIQUETAS_DEV, 300), (config.ETIQUETAS_TEST, 320), (config.ETIQUETAS_TEST_PASADA2, 160),
                   (config.ETIQUETAS_ES, 80)]


@pytest.mark.parametrize("ruta,n", RUTAS_ETIQUETAS)
def test_etiquetas_versionadas_validas_y_estan_en_el_indice(ruta, n):
    indice = {d["id"] for d in leer_jsonl(config.INDICE_MUESTRA)}
    etiquetas = leer_jsonl(ruta)
    assert len(etiquetas) == n
    for d in etiquetas:
        AnalisisResena.model_validate(d)
        assert d["id"] in indice


def test_citas_literales_en_la_muestra_pequena(muestra_pequena):
    """Sin descargar nada: las 8 reseñas de tests/datos tienen sus citas literales en el texto."""
    etiquetas = {d["id"]: d for ruta, _ in RUTAS_ETIQUETAS for d in leer_jsonl(ruta)}
    for id_, r in muestra_pequena.items():
        assert etiquetas[id_]["aspectos"] or r["conjunto"] == "dev"
        for m in etiquetas[id_]["aspectos"]:
            assert cita_en_texto(m["cita"], texto_completo(r["titulo_resena"], r["texto"])), (id_, m)


@pytest.mark.skipif(not config.MUESTRA_ETIQUETADO.exists(),
                    reason="falta el texto reconstruido: python descargar_datos.py && python -m resenas.muestra")
@pytest.mark.parametrize("ruta,n", RUTAS_ETIQUETAS)
def test_citas_literales_en_toda_la_muestra(ruta, n):
    muestra = {d["id"]: d for d in leer_jsonl(config.MUESTRA_ETIQUETADO) + leer_jsonl(config.MUESTRA_ES)}
    for d in leer_jsonl(ruta):
        r = muestra[d["id"]]
        for m in d["aspectos"]:
            assert cita_en_texto(m["cita"], texto_completo(r["titulo_resena"], r["texto"])), (d["id"], m)


def test_el_texto_de_las_resenas_no_se_versiona():
    indice = leer_jsonl(config.INDICE_MUESTRA)
    assert len(indice) == 700
    assert {d["conjunto"] for d in indice} == {"dev", "test", "es"}
    assert all(set(d) == set(CAMPOS_INDICE) for d in indice)          # sin titulo_resena ni texto
    assert len({d["id"] for d in indice}) == 700
    assert not (config.DIR_ETIQUETAS / "muestra.jsonl").exists()
    assert not (config.DIR_ETIQUETAS / "muestra_es.jsonl").exists()


def test_muestra_pequena_coincide_con_las_huellas_del_indice(muestra_pequena):
    indice = {d["id"]: d for d in leer_jsonl(config.INDICE_MUESTRA)}
    for id_, r in muestra_pequena.items():
        assert huella(r["titulo_resena"], r["texto"]) == indice[id_]["huella"]


def test_reconstruir_verifica_las_huellas(tmp_path, monkeypatch, muestra_pequena):
    """Con una 'descarga' falsa que contiene las 8 reseñas: reconstruye; si el texto cambia, se niega a etiquetar."""
    filas = list(muestra_pequena.values())
    indice = [{**{k: r[k] for k in CAMPOS_INDICE if k != "huella"}, "huella": huella(r["titulo_resena"], r["texto"])}
              for r in filas]
    ruta_indice = tmp_path / "indice.jsonl"
    ruta_indice.write_text("\n".join(json.dumps(d) for d in indice), encoding="utf-8")
    fuente = pd.DataFrame(filas)[["id", "titulo_resena", "texto"]]
    fuente.to_parquet(tmp_path / "resenas.parquet")
    monkeypatch.setattr(config, "INDICE_MUESTRA", ruta_indice)
    monkeypatch.setattr(config, "RESENAS_RAW", tmp_path / "resenas.parquet")
    monkeypatch.setattr(config, "MUESTRA_ETIQUETADO", tmp_path / "m.jsonl")
    monkeypatch.setattr(config, "MUESTRA_ES", tmp_path / "m_es.jsonl")
    assert len(reconstruir(verbose=False)) == 8
    assert len(leer_jsonl(tmp_path / "m_es.jsonl")) == 1
    fuente.loc[0, "texto"] = "otro texto"
    fuente.to_parquet(tmp_path / "resenas.parquet")
    with pytest.raises(ValueError, match="no coincide"):
        reconstruir(verbose=False)
    (tmp_path / "resenas.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="descargar_datos"):
        reconstruir(verbose=False)


def test_dev_y_test_no_se_solapan():
    dev = {d["id"] for d in leer_jsonl(config.ETIQUETAS_DEV)}
    test = {d["id"] for d in leer_jsonl(config.ETIQUETAS_TEST)}
    p2 = {d["id"] for d in leer_jsonl(config.ETIQUETAS_TEST_PASADA2)}
    assert not dev & test
    assert p2 <= test


# --------------------------------------------------------------------------- métricas
def _r(glob, **asp):
    return {"global": glob, "aspectos": asp}


def test_metricas_a_mano():
    verdad = {"a": _r("negativo", calidad="negativo", envio="positivo"), "b": _r("positivo", precio="positivo")}
    pred = {"a": _r("negativo", calidad="positivo", envio="positivo", diseno="positivo"), "b": _r("neutro")}
    r = evaluar(verdad, pred, bootstrap=0)
    # pares: tp = calidad, envio (2); fp = diseno (1); fn = precio (1) → P = R = F1 = 2/3
    assert r["f1_aspectos"] == pytest.approx(2 / 3)
    # tríos: tp = envio (1); fp = calidad+, envio?, diseno → 3 predichas, 1 bien; 3 en la verdad
    assert r["f1_aspecto_polaridad"] == pytest.approx(1 / 3)
    assert r["exactitud_polaridad"] == pytest.approx(1 / 2)
    assert r["exactitud_global"] == pytest.approx(1 / 2)
    assert r["por_aspecto"]["precio"]["exhaustividad"] == 0


def test_bootstrap_da_intervalo_que_contiene_el_valor():
    verdad = {str(i): _r("positivo", calidad="positivo" if i % 2 else "negativo") for i in range(40)}
    pred = {str(i): _r("positivo", calidad="positivo") if i % 3 else _r("negativo") for i in range(40)}
    r = evaluar(verdad, pred, bootstrap=200)
    lo, hi = r["ic95"]["f1_aspectos"]
    assert lo <= r["f1_aspectos"] <= hi


def test_kappa_y_acuerdo():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    p = {str(i): _r("positivo", calidad="negativo", precio="positivo") for i in range(10)}
    a = acuerdo(p, p)
    assert a["f1_aspectos_entre_pasadas"] == 1.0 and a["acuerdo_polaridad"] == 1.0


def test_como_dict():
    d = {"sentimiento_global": "neutro", "aspectos": [{"aspecto": "carga", "polaridad": "negativo", "cita": "x"}]}
    assert como_dict(d) == {"global": "neutro", "aspectos": {"carga": "negativo"}}
