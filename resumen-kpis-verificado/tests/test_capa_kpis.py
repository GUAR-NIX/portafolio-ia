"""Capa de KPIs: cálculos deterministas contra la instantánea."""
import numpy as np
import pandas as pd
import pytest

from kpis import capa_kpis, config


@pytest.fixture(scope="module")
def t():
    return capa_kpis.tabla_larga()


def test_instantanea_completa():
    df = capa_kpis.cargar()
    assert set(df["iso3"]) == set(config.PAISES)
    assert set(df["indicador"]) == {i.clave for i in config.INDICADORES}
    assert df["anio"].min() == 2000
    assert not df.duplicated(["iso3", "indicador", "anio"]).any()


def test_variacion_en_pp_para_tasas_y_en_pct_para_niveles(t):
    inf = t[(t.indicador == "inflacion") & (t.iso3 == "ECU") & (t.anio == 2024)].iloc[0]
    assert inf["unidad_variacion"] == "pp"
    assert inf["variacion"] == pytest.approx(inf["valor"] - inf["valor_previo"])
    pib = t[(t.indicador == "pib_usd") & (t.iso3 == "ECU") & (t.anio == 2024)].iloc[0]
    assert pib["unidad_variacion"] == "%"
    assert pib["variacion"] == pytest.approx((pib["valor"] / pib["valor_previo"] - 1) * 100)


def test_variacion_solo_entre_anios_consecutivos(t):
    # Pobreza, Gini y deuda tienen años sin dato: no se calcula variación a través de un hueco
    tras_hueco = t[t["anio_previo"].notna() & (t["anio"] - t["anio_previo"] > 1)]
    assert len(tras_hueco) > 0
    assert tras_hueco["variacion"].isna().all()


def test_mediana_y_ranking(t):
    sub = t[(t.indicador == "crecimiento") & (t.anio == 2024)]
    assert sub["mediana_region"].iloc[0] == pytest.approx(sub["valor"].median())
    assert sub.loc[sub["valor"].idxmax(), "ranking"] == 1
    assert sub.loc[sub["valor"].idxmin(), "ranking"] == len(sub)
    r = capa_kpis.ranking_regional("crecimiento", 2024)
    assert list(r["ranking"]) == sorted(r["ranking"])


def test_brecha_con_region(t):
    f = t[(t.indicador == "desempleo") & (t.iso3 == "ECU") & (t.anio == 2024)].iloc[0]
    assert f["brecha_region"] == pytest.approx(f["valor"] - f["mediana_region"])


def test_anomalia_no_mira_el_futuro():
    """El z de un año se calcula solo con años anteriores: recortar el futuro no cambia el pasado."""
    t = capa_kpis.tabla_larga()
    a = t[(t.indicador == "desempleo") & (t.iso3 == "PAN") & (t.anio == 2024)].iloc[0]
    assert a["anomalia"] and a["z_anomalia"] > capa_kpis.UMBRAL_ANOMALIA
    hist = t[(t.indicador == "desempleo") & (t.iso3 == "PAN") & (t.anio < 2024)]["variacion"].dropna().to_numpy()
    med = np.median(hist)
    escala = max(1.4826 * np.median(np.abs(hist - med)), capa_kpis.PISO_ESCALA["pp"])
    assert a["z_anomalia"] == pytest.approx((a["variacion"] - med) / escala)


def test_poblacion_no_genera_falsas_anomalias(t):
    assert not t[(t.indicador == "poblacion") & (t.anio >= 2015)]["anomalia"].any()


def test_hechos_y_tabla_para_prompt():
    hs = capa_kpis.hechos("ECU", 2024)
    ids = {h.id for h in hs}
    assert "crecimiento|ECU|2024|valor" in ids and "crecimiento|LAT|2024|mediana_region" in ids
    tabla = capa_kpis.tabla_para_prompt("ECU", 2024)
    assert "-1,9 %" in tabla and "puesto 18 de 18" in tabla


def test_comparacion_ecuador():
    c = capa_kpis.comparacion_con_region("ECU", 2024)
    assert len(c) == len(config.INDICADORES)
    assert isinstance(c, pd.DataFrame)


@pytest.mark.parametrize("x,dec,esperado", [(1234.5, 1, "1.234,5"), (-0.04, 1, "0,0"), (-1.94, 1, "-1,9")])
def test_formato_espanol(x, dec, esperado):
    assert capa_kpis.fmt_num(x, dec) == esperado
