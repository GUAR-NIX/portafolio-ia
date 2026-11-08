"""Pruebas de limpieza de máscaras XXXX, truncado y carga de datos CFPB y español."""
from __future__ import annotations

import pytest

from quejas import config, datos


def test_contar_y_limpiar_mascaras_anonimizacion():
    texto = "On XX/XX/2017 I paid {$450.00} to XXXX XXXX for account XXXX and {$20.00} fee."
    n_xxxx, n_montos = datos.contar_mascaras_anonimizacion(texto)
    assert n_xxxx == 4
    assert n_montos == 2

    limpio = datos.limpiar_texto_para_tfidf(texto)
    assert "XXXX" not in limpio
    assert "XX/XX" not in limpio
    assert "monto_usd" in limpio


def test_preparar_texto_para_modelo_colapsa_redacciones_y_trunca():
    texto = "XXXX XXXX XXXX charged my credit card " + " ".join([f"word{i}" for i in range(150)])
    corto = datos.preparar_texto_para_modelo(texto, max_palabras=30)
    palabras = corto.split()
    assert len(palabras) == 30
    assert corto.startswith("[REDACTED] charged my credit card")


def test_quejas_espanol_son_traducciones_de_quejas_reales_del_conjunto_comun():
    df_es = datos.cargar_quejas_espanol()
    assert len(df_es) == 60
    assert (df_es["categoria"].value_counts() == 10).all()
    assert df_es["origen"].str.contains("REAL del CFPB").all()
    assert sorted(df_es["orden_comun"]) == list(range(60))


@pytest.mark.skipif(not config.DATOS_CRUDOS.exists(), reason="Parquet CFPB no descargado")
def test_corpus_cfpb_particion_temporal_sin_fugas():
    df = datos.cargar_corpus_cfpb()
    assert len(df) == 50568
    assert set(df["categoria"].unique()) == set(config.CATEGORIAS)
    fechas = df.groupby("particion")["fecha_recepcion"].agg(["min", "max"])
    assert fechas.loc["train", "max"] < fechas.loc["val", "min"]
    assert fechas.loc["val", "max"] < fechas.loc["test", "min"]
    assert fechas.loc["test_misma_epoca", "max"] < fechas.loc["val", "min"]
    assert 20000 <= (df["particion"] == "train").sum() <= 50000
    comun = df[df["orden_comun"] >= 0]
    assert (comun["particion"] == "test").all()
    assert (comun["categoria"].value_counts() == 80).all()
    # Las 6 primeras del orden intercalado cubren las 6 mesas
    assert set(comun.sort_values("orden_comun").head(6)["categoria"]) == set(config.CATEGORIAS)
    assert set(df.loc[df["urgente"], "tipo_urgencia"]) <= {"fraude", "ejecucion_hipotecaria", "reposesion", "amenaza_legal_embargo"}
