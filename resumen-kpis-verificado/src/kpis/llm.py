"""Cliente de Ollama con grabación: cada respuesta se guarda al instante en data/grabacion_llm.jsonl,
indexada por una huella de (modelo, mensajes, formato, opciones). Las corridas son reanudables y pytest,
la app y la API reproducen todo sin Ollama (KPIS_PROVEEDOR=grabado, el valor por defecto)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import httpx

from . import config

_bloqueo = threading.Lock()
REINTENTOS_RED = 4
_cache: dict[str, dict[str, Any]] | None = None


class SinGrabacion(RuntimeError):
    """La llamada no está grabada y el proveedor es «grabado» (o Ollama no responde)."""


def _cargar() -> dict[str, dict[str, Any]]:
    global _cache
    if _cache is None:
        _cache = {}
        if config.GRABACION_LLM.exists():
            for linea in config.GRABACION_LLM.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    reg = json.loads(linea)
                    _cache[reg["huella"]] = reg
    return _cache


def huella(mensajes: list[dict], formato: Any, opciones: dict) -> str:
    clave = json.dumps({"modelo": config.MODELO, "mensajes": mensajes, "formato": formato,
                        "opciones": {k: v for k, v in opciones.items() if k != "num_thread"}},
                       ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(clave.encode("utf-8")).hexdigest()


def chat(mensajes: list[dict], formato: Any = None, etiqueta: str = "",
         proveedor: str | None = None) -> dict[str, Any]:
    """Devuelve {"contenido": str, "segundos": float, "tokens_salida": int, "grabado": bool}."""
    opciones = dict(config.OPCIONES_LLM)
    h = huella(mensajes, formato, opciones)
    cache = _cargar()
    if h in cache:
        reg = cache[h]
        return {"contenido": reg["contenido"], "segundos": reg["segundos"],
                "tokens_salida": reg.get("tokens_salida", 0), "grabado": True}
    if (proveedor or config.PROVEEDOR) != "ollama":
        raise SinGrabacion(f"Llamada no grabada ({etiqueta}). Arranca `ollama serve` y usa KPIS_PROVEEDOR=ollama.")
    cuerpo = {"model": config.MODELO, "messages": mensajes, "stream": False, "options": opciones,
              "keep_alive": "30m"}
    if formato is not None:
        cuerpo["format"] = formato
    # Ollama en CPU a veces corta la conexión: se reintenta con espera creciente. El tiempo grabado
    # es el del intento que respondió.
    for intento in range(REINTENTOS_RED + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=cuerpo, timeout=900)
            r.raise_for_status()
            break
        except httpx.HTTPError as e:
            if intento == REINTENTOS_RED:
                raise SinGrabacion(f"Ollama no responde en {config.OLLAMA_URL}: {e}") from e
            espera = 15 * 2 ** intento
            print(f"  Ollama cortó la conexión ({e}); reintento en {espera} s", flush=True)
            time.sleep(espera)
    datos = r.json()
    seg = time.perf_counter() - t0
    reg = {"huella": h, "etiqueta": etiqueta, "modelo": config.MODELO, "contenido": datos["message"]["content"],
           "segundos": round(seg, 2), "tokens_entrada": datos.get("prompt_eval_count", 0),
           "tokens_salida": datos.get("eval_count", 0)}
    with _bloqueo:
        cache[h] = reg
        with config.GRABACION_LLM.open("a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    return {"contenido": reg["contenido"], "segundos": reg["segundos"], "tokens_salida": reg["tokens_salida"],
            "grabado": False}
