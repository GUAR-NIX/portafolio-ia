"""Motor de inferencia: clasificación, abstención a triaje humano, cascada TF-IDF → LLM y urgencia."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config, datos
from .llm import CacheEmbeddings, GrabadoraLLM, asegurar_ollama_activo, clasificar_texto_llm, huella


@lru_cache(maxsize=1)
def cargar_artefacto(ruta: Path = config.MODELOS / "clasificador_quejas.joblib") -> dict[str, Any]:
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero: python -m quejas.entrenar"
        )
    return joblib.load(ruta)


@lru_cache(maxsize=1)
def _cache_emb() -> CacheEmbeddings:
    return CacheEmbeddings(config.CACHE_EMBEDDINGS)


@lru_cache(maxsize=1)
def _grabadora() -> GrabadoraLLM:
    return GrabadoraLLM(config.GRABACION_LLM)


def cargar_metadata(ruta: Path = config.MODELOS / "metadata.json") -> dict[str, Any]:
    if Path(ruta).exists():
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    return cargar_artefacto().get("metadata", {})


def detectar_urgencia(texto: str) -> dict[str, Any]:
    """Probabilidad de urgencia (fraude, ejecución hipotecaria, reposesión, amenaza legal) con TF-IDF + LR."""
    m = cargar_artefacto()["modelo_urgencia"]
    p = float(m.proba([datos.limpiar_texto_para_tfidf(texto)])[0])
    return {"prob_urgente": round(p, 4), "umbral_urgencia": round(m.umbral_, 3), "urgente": bool(p >= m.umbral_)}


def clasificar_queja(
    texto: str,
    enfoque: str = "auto",
    umbral_abstencion: float | None = None,
    permitir_ollama: bool = False,
) -> dict[str, Any]:
    """
    Clasifica una queja financiera y decide si enrutarla automáticamente a la mesa especialista
    o derivarla a triaje humano cuando la confianza (calibrada) es inferior a `umbral_abstencion`.
    """
    if not texto or len(texto.strip()) < 10:
        raise ValueError("El texto de la queja debe tener al menos 10 caracteres.")

    art = cargar_artefacto()
    umbral = float(
        umbral_abstencion
        if umbral_abstencion is not None
        else art.get("umbral_optimo", config.UMBRAL_ABSTENCION)
    )
    texto_corto = datos.preparar_texto_para_modelo(texto, max_palabras=110)
    texto_limpio = datos.limpiar_texto_para_tfidf(texto)
    n_xxxx, n_montos = datos.contar_mascaras_anonimizacion(texto)

    cache = _cache_emb()
    usar_emb = False
    if enfoque == "bge_m3":
        usar_emb = (huella(texto_corto) in cache.vectores) or permitir_ollama
    elif enfoque == "auto":
        usar_emb = huella(texto_corto) in cache.vectores

    if usar_emb:
        vec = cache.matriz([texto_corto], permitir_ollama=permitir_ollama)
        probs = art["modelo_emb_sup"].predecir_proba_desde_vectores(vec)[0]
        motor_usado = "bge_m3_lr"
    else:
        probs = art["modelo_tfidf"].predecir_proba([texto_limpio])[0]
        motor_usado = "tfidf_lr"

    orden = np.argsort(-probs)
    i1, i2 = int(orden[0]), int(orden[1])
    cat1, cat2 = config.CATEGORIAS[i1], config.CATEGORIAS[i2]
    conf1, conf2 = float(probs[i1]), float(probs[i2])

    enrutar_auto = bool(conf1 >= umbral)
    destino = (
        f"Mesa especialista: {config.ETIQUETAS_CATEGORIA[cat1]}"
        if enrutar_auto
        else f"Triaje humano (ambigua entre {config.ETIQUETAS_CATEGORIA[cat1]} y {config.ETIQUETAS_CATEGORIA[cat2]})"
    )

    return {
        "categoria_principal": cat1,
        "etiqueta_principal": config.ETIQUETAS_CATEGORIA[cat1],
        "confianza": round(conf1, 4),
        "categoria_secundaria": cat2,
        "etiqueta_secundaria": config.ETIQUETAS_CATEGORIA[cat2],
        "confianza_secundaria": round(conf2, 4),
        "margen_confianza": round(conf1 - conf2, 4),
        "umbral_abstencion": round(umbral, 3),
        "enrutar_automaticamente": enrutar_auto,
        "destino_operativo": destino,
        "motor_utilizado": motor_usado,
        "mascaras_xxxx_detectadas": n_xxxx,
        "montos_detectados": n_montos,
        "probabilidades": {
            c: round(float(probs[idx]), 4) for idx, c in enumerate(config.CATEGORIAS)
        },
        **detectar_urgencia(texto),
    }


def enrutar_con_cascada(texto: str, permitir_ollama: bool = False, modo_llm: str = "pocos") -> dict[str, Any]:
    """
    Ruta completa de una queja:
      1. TF-IDF + LR (calibrado) propone una mesa. Si su confianza >= umbral de cascada, decide él.
      2. Si no, la queja va al LLM (few-shot por defecto): se usa la respuesta grabada o, con
         `permitir_ollama=True`, se consulta Ollama. Sin LLM disponible, va a triaje humano.
      3. Urgencia: TF-IDF + LR de urgencia; si se consultó el LLM, también cuenta su bandera.
    """
    base = clasificar_queja(texto, enfoque="tfidf", umbral_abstencion=0.0)
    art = cargar_artefacto()
    umbral_c = float(art["umbral_cascada"])
    pasos = [f"TF-IDF + LR propone «{base['etiqueta_principal']}» con confianza calibrada {base['confianza']:.2f} "
             f"(umbral de cascada {umbral_c:.2f})."]
    resultado = {
        "clasificador_barato": base,
        "umbral_cascada": round(umbral_c, 4),
        "fue_al_llm": False,
        "llm": None,
        "categoria_final": base["categoria_principal"],
        "decidido_por": "TF-IDF + LR",
    }
    urgente = base["urgente"]
    if base["confianza"] >= umbral_c:
        pasos.append("Confianza suficiente: se enruta directo, sin LLM.")
    else:
        resultado["fue_al_llm"] = True
        texto_corto = datos.preparar_texto_para_modelo(texto, max_palabras=110)
        grab = _grabadora()
        disponible = grab.obtener(texto_corto, modo_llm) is not None or (permitir_ollama and asegurar_ollama_activo())
        if disponible:
            reg = clasificar_texto_llm(texto_corto, grab, modo=modo_llm, ejemplos=art.get("ejemplos_few_shot"),
                                       permitir_ollama=permitir_ollama)
            resultado["llm"] = reg
            resultado["categoria_final"] = reg["categoria"]
            resultado["decidido_por"] = f"LLM {'few-shot' if modo_llm == 'pocos' else 'zero-shot'} ({config.MODELO_LLM})"
            urgente = urgente or bool(reg.get("urgente"))
            acuerdo = reg["categoria"] == base["categoria_principal"]
            pasos.append(f"Confianza baja: se consulta el LLM, que responde «{config.ETIQUETAS_CATEGORIA[reg['categoria']]}» "
                         f"({'coincide' if acuerdo else 'no coincide'} con TF-IDF).")
        else:
            resultado["categoria_final"] = None
            resultado["decidido_por"] = "Triaje humano"
            pasos.append("Confianza baja y LLM no disponible (sin Ollama ni respuesta grabada): va a triaje humano.")
    cat = resultado["categoria_final"]
    resultado["destino_operativo"] = (
        f"Mesa especialista: {config.ETIQUETAS_CATEGORIA[cat]}" if cat else "Triaje humano"
    )
    resultado["urgente"] = bool(urgente)
    resultado["prioridad"] = "alta (posible fraude, ejecución, reposesión o amenaza legal)" if urgente else "normal"
    pasos.append(f"Urgencia: prob. {base['prob_urgente']:.2f} (umbral {base['umbral_urgencia']:.2f}) → prioridad {resultado['prioridad']}.")
    resultado["pasos"] = pasos
    return resultado


def enrutar_lote(
    quejas: list[dict[str, Any]],
    enfoque: str = "auto",
    umbral_abstencion: float | None = None,
) -> dict[str, Any]:
    resultados = []
    for idx, q in enumerate(quejas):
        res = clasificar_queja(
            texto=str(q["texto_queja"]),
            enfoque=enfoque,
            umbral_abstencion=umbral_abstencion,
            permitir_ollama=False,
        )
        res["id_queja"] = str(q.get("id_queja", f"Q_{idx + 1}"))
        resultados.append(res)

    n_auto = sum(1 for r in resultados if r["enrutar_automaticamente"])
    n_triaje = len(resultados) - n_auto
    return {
        "total_quejas": len(resultados),
        "enrutadas_automaticamente": n_auto,
        "derivadas_a_triaje_humano": n_triaje,
        "cobertura_automatizada_pct": round(n_auto / max(len(resultados), 1) * 100.0, 2),
        "resultados": resultados,
    }
