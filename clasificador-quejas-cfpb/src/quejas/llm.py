"""Cliente de Ollama para embeddings (bge-m3) y clasificación estructurada (qwen2.5:7b) en modo
zero-shot y few-shot, con grabación persistente en disco (data/embeddings_bge_m3.npz y
data/grabacion_llm.jsonl) para que la evaluación, la API y las pruebas corran sin Ollama abierto.

Cada respuesta del LLM se graba con:
  - la categoría y la bandera de urgencia que devolvió el JSON (decodificación restringida);
  - la distribución de probabilidad de la PRIMERA ficha (token) del valor de `categoria` y de
    `urgente`, obtenida de los logprobs de Ollama. Esa distribución es la señal de confianza del
    LLM: un primer diseño le pedía al modelo un campo `confianza` y respondía 0,99 casi siempre.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import httpx
import numpy as np

from . import config

_PROCESO_OLLAMA: subprocess.Popen | None = None


def huella(texto: str) -> str:
    return hashlib.sha1(texto.strip().encode("utf-8")).hexdigest()[:16]


def asegurar_ollama_activo(url: str = config.OLLAMA_URL) -> bool:
    """Verifica si Ollama responde en `url`; si no, intenta iniciar `ollama serve` localmente."""
    global _PROCESO_OLLAMA
    try:
        r = httpx.get(f"{url}/api/tags", timeout=2.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    exe = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"))
    if not exe.exists():
        return False

    _PROCESO_OLLAMA = subprocess.Popen(
        [str(exe), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(25):
        try:
            r = httpx.get(f"{url}/api/tags", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def embeber_ollama(
    textos: list[str],
    modelo: str = config.MODELO_EMB,
    url: str = config.OLLAMA_URL,
) -> np.ndarray:
    """Llama a /api/embed de Ollama y devuelve matriz (N, 1024) normalizada L2."""
    if not textos:
        return np.zeros((0, config.DIM_EMBEDDING), dtype=np.float32)
    asegurar_ollama_activo(url)
    with httpx.Client(timeout=600.0) as client:
        r = client.post(f"{url}/api/embed", json={"model": modelo, "input": textos})
        r.raise_for_status()
        vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
    normas = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(normas, 1e-12)


class CacheEmbeddings:
    """Almacena vectores bge-m3 por huella SHA-1 en un archivo comprimido .npz (float16)."""

    def __init__(self, ruta: Path = config.CACHE_EMBEDDINGS, funcion=embeber_ollama) -> None:
        self.ruta = Path(ruta)
        self.funcion = funcion
        self.vectores: dict[str, np.ndarray] = {}
        self.segundos_por_texto: float = 0.5
        if self.ruta.exists():
            z = np.load(self.ruta)
            for h, v in zip(z["huellas"], z["vectores"]):
                self.vectores[str(h)] = v.astype(np.float32)
            if "segundos_por_texto" in z:
                self.segundos_por_texto = float(z["segundos_por_texto"])

    def guardar(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        hs = list(self.vectores.keys())
        if not hs:
            return
        tmp = self.ruta.with_name(self.ruta.stem + "_tmp.npz")
        np.savez_compressed(
            tmp,
            huellas=np.array(hs),
            vectores=np.stack([self.vectores[h] for h in hs]).astype(np.float16),
            segundos_por_texto=np.array(self.segundos_por_texto, dtype=np.float32),
        )
        tmp.replace(self.ruta)

    def asegurar_textos(self, textos: list[str], lote: int = 16) -> int:
        faltan = list(dict.fromkeys(t for t in textos if huella(t) not in self.vectores))
        if not faltan:
            return 0
        t0 = time.perf_counter()
        hechos = 0
        for i in range(0, len(faltan), lote):
            parte = faltan[i : i + lote]
            vecs = self.funcion(parte)
            for t, v in zip(parte, vecs):
                self.vectores[huella(t)] = v
            hechos += len(parte)
            if hechos % 128 == 0 or hechos == len(faltan):
                self.segundos_por_texto = (time.perf_counter() - t0) / max(hechos, 1)
                self.guardar()
                print(
                    f"      [bge-m3] {hechos}/{len(faltan)} textos embebidos "
                    f"({self.segundos_por_texto * 1000:.0f} ms/texto)",
                    flush=True,
                )
        return len(faltan)

    def matriz(self, textos: list[str], permitir_ollama: bool = False) -> np.ndarray:
        if not textos:
            return np.zeros((0, config.DIM_EMBEDDING), dtype=np.float32)
        faltan = [t for t in textos if huella(t) not in self.vectores]
        if faltan:
            if permitir_ollama:
                self.asegurar_textos(faltan)
            else:
                raise KeyError(
                    f"Faltan {len(faltan)} textos en {self.ruta.name}. "
                    "Ejecuta con Ollama activo para poblar la caché."
                )
        mat = np.stack([self.vectores[huella(t)] for t in textos]).astype(np.float32)
        normas = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(normas, 1e-12)


# ---------------------------------------------------------------------------
# Prompt y esquema del LLM
# ---------------------------------------------------------------------------
ESQUEMA_JSON_LLM = {
    "type": "object",
    "properties": {
        "categoria": {"type": "string", "enum": config.CATEGORIAS},
        "urgente": {"type": "boolean"},
    },
    "required": ["categoria", "urgente"],
}

PROMPT_SISTEMA_LLM = (
    "You are a financial complaint routing specialist. "
    "Classify the customer complaint into EXACTLY ONE of these 6 product desks:\n"
    "- reporte_credito : Credit reporting bureaus (Equifax, Experian, TransUnion), credit report errors, credit score, unauthorized inquiries, credit repair or monitoring services.\n"
    "- cobranza_deudas : Debt collection (by collection agencies or creditors collecting a debt), harassing collection calls, disputed or medical debt collection.\n"
    "- hipotecas : Home mortgage loans, escrow, refinancing, loan modification, foreclosure.\n"
    "- tarjetas_credito : Credit cards or prepaid cards, billing disputes, card fees, APR, rewards.\n"
    "- cuentas_bancarias : Checking/savings bank accounts, overdraft fees, ATM, deposits, wire/Zelle money transfers, money services.\n"
    "- prestamos_consumo : Student loans, auto/vehicle loans or leases, personal, payday or title installment loans.\n"
    "Also set 'urgente' to true ONLY if the complaint is about: fraud or identity theft "
    "(accounts, debts, cards or loans opened fraudulently, scams); risk of losing the home "
    "(foreclosure, loan modification, struggling to pay the mortgage); vehicle repossession; or a "
    "debt collector threatening arrest/jail, seizing property or exempt funds, or suing without notice. "
    "Otherwise 'urgente' is false.\n"
    "Respond strictly in JSON with keys 'categoria' and 'urgente'."
)


def mensajes_llm(texto: str, modo: str, ejemplos: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    """Construye los mensajes de chat. En few-shot, cada ejemplo es un turno usuario/asistente."""
    msgs = [{"role": "system", "content": PROMPT_SISTEMA_LLM}]
    if modo == "pocos":
        for ej in ejemplos or []:
            msgs.append({"role": "user", "content": f"Complaint: {ej['texto_corto']}"})
            msgs.append({
                "role": "assistant",
                "content": json.dumps({"categoria": ej["categoria"], "urgente": bool(ej["urgente"])}),
            })
    msgs.append({"role": "user", "content": f"Complaint: {texto}"})
    return msgs


def _ficha_del_valor(logprobs: list[dict[str, Any]], contenido: str, clave: str) -> dict[str, Any] | None:
    """Devuelve la entrada de logprobs de la primera ficha que cubre el valor de `clave` en el JSON."""
    pos_clave = contenido.find(f'"{clave}"')
    if pos_clave < 0:
        return None
    pos = contenido.find(":", pos_clave) + 1
    while pos < len(contenido) and contenido[pos] in ' "':
        pos += 1
    desplazamiento = 0
    for lp in logprobs:
        fin = desplazamiento + len(lp.get("token", ""))
        if fin > pos:
            return lp
        desplazamiento = fin
    return None


def probabilidades_desde_logprobs(
    logprobs: list[dict[str, Any]] | None, contenido: str
) -> tuple[dict[str, float] | None, float | None, float | None]:
    """
    Convierte los top-logprobs de la primera ficha del valor de `categoria` en una distribución
    sobre las 6 mesas (cada ficha se asigna a las mesas cuyo nombre empieza por ella) y calcula
    P(urgente = true) con la ficha del valor de `urgente`. Devuelve también la masa cubierta.
    """
    if not logprobs:
        return None, None, None
    probs_cat = None
    masa = None
    lp_cat = _ficha_del_valor(logprobs, contenido, "categoria")
    if lp_cat is not None:
        acum = {c: 0.0 for c in config.CATEGORIAS}
        for alt in lp_cat.get("top_logprobs", []):
            tok = alt["token"].strip().strip('"').strip().lower()
            if not tok:
                continue
            destino = [c for c in config.CATEGORIAS if c.startswith(tok) or tok.startswith(c)]
            for c in destino:
                acum[c] += math.exp(alt["logprob"]) / len(destino)
        masa = sum(acum.values())
        if masa > 0:
            probs_cat = {c: v / masa for c, v in acum.items()}
    p_urg = None
    lp_urg = _ficha_del_valor(logprobs, contenido, "urgente")
    if lp_urg is not None:
        p_true = p_false = 0.0
        for alt in lp_urg.get("top_logprobs", []):
            tok = alt["token"].strip().lower()
            if tok and "true".startswith(tok):
                p_true += math.exp(alt["logprob"])
            elif tok and "false".startswith(tok):
                p_false += math.exp(alt["logprob"])
        if p_true + p_false > 0:
            p_urg = p_true / (p_true + p_false)
    return probs_cat, p_urg, masa


class GrabadoraLLM:
    """Almacena en data/grabacion_llm.jsonl las respuestas reales de qwen2.5:7b, por (modo, texto)."""

    def __init__(self, ruta: Path = config.GRABACION_LLM) -> None:
        self.ruta = Path(ruta)
        self.registros: dict[tuple[str, str], dict[str, Any]] = {}
        if self.ruta.exists():
            for ln in self.ruta.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.registros[(d.get("modo", "cero"), d["huella"])] = d

    def obtener(self, texto: str, modo: str = "cero") -> dict[str, Any] | None:
        return self.registros.get((modo, huella(texto)))

    def guardar(self, texto: str, registro: dict[str, Any], modo: str = "cero") -> None:
        h = huella(texto)
        registro["huella"] = h
        registro["modo"] = modo
        self.registros[(modo, h)] = registro
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def clasificar_texto_llm(
    texto: str,
    grabadora: GrabadoraLLM,
    modo: str = "cero",
    ejemplos: list[dict[str, Any]] | None = None,
    permitir_ollama: bool = True,
    modelo: str = config.MODELO_LLM,
    url: str = config.OLLAMA_URL,
) -> dict[str, Any]:
    previo = grabadora.obtener(texto, modo)
    if previo is not None:
        return previo
    if not permitir_ollama:
        raise RuntimeError(
            f"La queja no está en {grabadora.ruta.name} (modo {modo}) y permitir_ollama=False."
        )

    asegurar_ollama_activo(url)
    t0 = time.perf_counter()
    payload = {
        "model": modelo,
        "messages": mensajes_llm(texto, modo, ejemplos),
        "stream": False,
        "format": ESQUEMA_JSON_LLM,
        "logprobs": True,
        "top_logprobs": 10,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 32, "num_ctx": 2048},
    }
    with httpx.Client(timeout=900.0) as client:
        r = client.post(f"{url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    duracion = time.perf_counter() - t0
    contenido = data["message"]["content"]
    try:
        parsed = json.loads(contenido)
        cat = parsed.get("categoria")
        urg = bool(parsed.get("urgente", False))
        valido = cat in config.CATEGORIAS
    except Exception:
        cat, urg, valido = None, False, False
    probs_cat, p_urg, masa = probabilidades_desde_logprobs(data.get("logprobs"), contenido)

    reg = {
        "categoria": cat if valido else config.CATEGORIAS[0],
        "respuesta_valida": valido,
        "urgente": urg,
        "probs_categoria": {k: round(v, 5) for k, v in probs_cat.items()} if probs_cat else None,
        "masa_logprobs": round(masa, 5) if masa is not None else None,
        "p_urgente": round(p_urg, 5) if p_urg is not None else None,
        "segundos": round(duracion, 3),
        "tokens_entrada": int(data.get("prompt_eval_count", 0)),
        "tokens_salida": int(data.get("eval_count", 0)),
    }
    grabadora.guardar(texto, reg, modo)
    return reg
