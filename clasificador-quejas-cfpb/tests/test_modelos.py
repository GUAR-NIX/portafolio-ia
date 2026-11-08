"""Pruebas de TF-IDF + LR, Zero-Shot sobre embeddings, clasificador sobre embeddings y curva Few-Shot."""
from __future__ import annotations

import numpy as np
import pytest

from quejas import config
from quejas.llm import CacheEmbeddings, GrabadoraLLM, huella
from quejas.modelos import (
    ModeloEmbeddingsSupervisado,
    ModeloTFIDF,
    ModeloZeroShotEmbeddings,
    ModeloLLM,
    ajustar_temperatura,
    curva_aprendizaje_few_shot,
)
from quejas.llm import probabilidades_desde_logprobs


def test_modelo_tfidf_suma_uno_y_separa_clases(corpus_sintetico):
    textos, y, _ = corpus_sintetico
    m = ModeloTFIDF(C=2.0).ajustar(textos, y)
    probs = m.predecir_proba(textos[:12])
    assert probs.shape == (12, len(config.CATEGORIAS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    preds = m.predecir(textos)
    assert np.mean(np.array(preds) == y) > 0.95


def test_modelo_embeddings_supervisado_separa_clases(corpus_sintetico):
    _, y, X_vec = corpus_sintetico
    m = ModeloEmbeddingsSupervisado(C=4.0).ajustar_vectores(X_vec, y)
    probs = m.predecir_proba_desde_vectores(X_vec)
    assert probs.shape == (len(y), len(config.CATEGORIAS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    preds = m.predecir_desde_vectores(X_vec)
    assert np.mean(np.array(preds) == y) > 0.98


def test_modelo_zeroshot_embeddings_con_cache_temporal(tmp_path, corpus_sintetico):
    _, y, X_vec = corpus_sintetico
    cache = CacheEmbeddings(ruta=tmp_path / "emb.npz")
    for i, cat in enumerate(config.CATEGORIAS):
        proto_vec = np.zeros(config.DIM_EMBEDDING, dtype=np.float32)
        proto_vec[i * 20 : (i + 1) * 20] = 1.0
        cache.vectores[huella(config.PROTOTIPOS_TEXTO[cat])] = proto_vec

    zs = ModeloZeroShotEmbeddings(temperatura_escala=12.0).ajustar_prototipos(cache, permitir_ollama=False)
    probs = zs.predecir_proba_desde_vectores(X_vec)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    preds = zs.predecir_desde_vectores(X_vec)
    assert np.mean(np.array(preds) == y) > 0.98


def test_curva_few_shot_crece_con_mas_ejemplos(corpus_sintetico):
    textos, y, X_vec = corpus_sintetico
    filas = curva_aprendizaje_few_shot(
        X_train_vec=X_vec,
        X_train_tfidf_textos=textos,
        y_train=y,
        X_eval_vec=X_vec,
        X_eval_tfidf_textos=textos,
        y_eval=y,
        ejemplos_por_clase=(2, 10),
        n_repeticiones=2,
    )
    assert len(filas) == 2
    assert filas[1]["f1_macro_bge_m3_media"] >= filas[0]["f1_macro_bge_m3_media"] - 1e-6


def test_modelo_llm_reproduce_desde_grabadora_por_modo(tmp_path):
    grab = GrabadoraLLM(ruta=tmp_path / "grab.jsonl")
    txt = "Equifax refuses to remove fraudulent inquiry from my credit report."
    probs = {c: 0.0 for c in config.CATEGORIAS}
    probs.update({"reporte_credito": 0.8, "cobranza_deudas": 0.2})
    grab.guardar(txt, {"categoria": "reporte_credito", "urgente": True, "probs_categoria": probs,
                       "p_urgente": 0.9, "segundos": 3.4}, modo="cero")
    grab.guardar(txt, {"categoria": "cobranza_deudas", "urgente": False, "probs_categoria": None,
                       "p_urgente": None, "segundos": 5.0}, modo="pocos")

    r0 = ModeloLLM(GrabadoraLLM(ruta=tmp_path / "grab.jsonl"), "cero").predecir_lote([txt])
    assert r0["preds"] == ["reporte_credito"]
    assert float(r0["probs"][0, 0]) == pytest.approx(0.8)
    assert bool(r0["urgente"][0]) is True
    # Sin logprobs se usa la respuesta como distribución casi determinista
    r1 = ModeloLLM(GrabadoraLLM(ruta=tmp_path / "grab.jsonl"), "pocos").predecir_lote([txt])
    assert r1["preds"] == ["cobranza_deudas"]
    assert float(r1["probs"][0].sum()) == pytest.approx(1.0)
    with pytest.raises(RuntimeError):
        ModeloLLM(grab, "pocos").predecir_lote(["texto que no está grabado en ningún modo"])


def test_probabilidades_desde_logprobs_asigna_fichas_a_mesas():
    contenido = '{"categoria": "tarjetas_credito", "urgente": true}'
    fichas = ['{"', "categoria", '":', ' "', "tar", "jet", "as", "_credit", "o", '",', ' "', "urg", "ente", '":', " true", "}"]
    logprobs = [{"token": t, "logprob": 0.0, "top_logprobs": [{"token": t, "logprob": 0.0}]} for t in fichas]
    logprobs[4]["top_logprobs"] = [{"token": "tar", "logprob": np.log(0.7)}, {"token": "cob", "logprob": np.log(0.2)},
                                   {"token": "classification", "logprob": np.log(0.1)}]
    logprobs[14]["top_logprobs"] = [{"token": " true", "logprob": np.log(0.6)}, {"token": " false", "logprob": np.log(0.3)}]
    probs, p_urg, masa = probabilidades_desde_logprobs(logprobs, contenido)
    assert probs["tarjetas_credito"] == pytest.approx(0.7 / 0.9)
    assert probs["cobranza_deudas"] == pytest.approx(0.2 / 0.9)
    assert masa == pytest.approx(0.9)
    assert p_urg == pytest.approx(0.6 / 0.9)


def test_escalado_de_temperatura_suaviza_logits_sobreconfiados():
    rng = np.random.default_rng(0)
    y = rng.choice(config.CATEGORIAS, size=300)
    logits = rng.normal(0, 1, size=(300, 6))
    # Logits muy grandes con 50 % de aciertos: el modelo está sobreconfiado y T debe ser > 1
    for i, c in enumerate(y):
        j = config.CATEGORIAS.index(c) if i % 2 == 0 else (config.CATEGORIAS.index(c) + 1) % 6
        logits[i, j] += 12.0
    assert ajustar_temperatura(logits, y) > 2.0
