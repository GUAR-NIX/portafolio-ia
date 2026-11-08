"""
Cliente mínimo de Ollama (solo httpx), con el mismo patrón en toda la serie de proyectos:

* Ollama (por defecto, local y gratis): OLLAMA_URL, RAG_MODELO=qwen2.5:7b, RAG_MODELO_EMB=bge-m3
* Grabado (demo y pruebas sin Ollama): RAG_PROVEEDOR=grabado → reproduce data/grabacion_llm.jsonl

Con RAG_GRABAR=1 cada llamada real se guarda por huella del contenido (modelo + prompt + esquema): la evaluación se
reanuda si se corta y cualquiera puede reproducirla sin Ollama. Los embeddings se guardan igual en data/cache_embeddings.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import config


def huella(*partes) -> str:
    return hashlib.sha1(json.dumps(partes, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Almacen:
    """Diccionario clave → valor que se agrega a un .jsonl (un renglón por llamada: sobrevive a un apagón)."""

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self._datos: dict | None = None

    def cargar(self) -> dict:
        if self._datos is None:
            self._datos = {}
            if self.ruta.exists():
                for ln in self.ruta.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        try:
                            d = json.loads(ln)
                        except json.JSONDecodeError:  # último renglón cortado por un apagón
                            continue
                        self._datos[d["clave"]] = d["valor"]
        return self._datos

    def get(self, clave: str):
        return self.cargar().get(clave)

    def guardar(self, clave: str, valor) -> None:
        datos = self.cargar()
        if clave in datos:
            return
        datos[clave] = valor
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ruta, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clave": clave, "valor": valor}, ensure_ascii=False) + "\n")


GRABADORA = Almacen(config.GRABACION)
CACHE_EMB = Almacen(config.EMB_CACHE)


def grabar() -> bool:
    return os.environ.get("RAG_GRABAR", "0") == "1"


class ErrorLLM(RuntimeError):
    pass


@dataclass
class Uso:
    llamadas: int = 0
    segundos: float = 0.0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    desde_grabacion: int = 0


@dataclass
class LLM:
    proveedor: str = config.PROVEEDOR
    modelo: str = config.MODELO
    url: str = config.OLLAMA_URL
    temperatura: float = config.TEMPERATURA
    uso: Uso = field(default_factory=Uso)

    def chat(self, sistema: str, mensajes: list[dict], json_schema: dict | None = None,
             max_tokens: int = 700) -> dict:
        """Devuelve {"texto", "segundos", "tokens_entrada", "tokens_salida"}; usa la grabación si existe."""
        clave = huella(self.modelo, sistema, mensajes, json_schema, max_tokens)
        if self.proveedor == "grabado" or grabar():
            previa = GRABADORA.get(clave)
            if previa is not None:
                self._sumar(previa, grabada=True)
                return previa
            if self.proveedor == "grabado":
                raise ErrorLLM("Modo demo: esta pregunta no está grabada. Abre Ollama para preguntas nuevas.")
        t0 = time.perf_counter()
        try:
            if self.proveedor != "ollama":
                raise ErrorLLM(f"Proveedor desconocido: {self.proveedor}")
            cuerpo = {
                "model": self.modelo,
                "messages": [{"role": "system", "content": sistema}, *mensajes],
                "stream": False,
                "options": {"temperature": self.temperatura, "num_predict": max_tokens, "num_ctx": 8192},
                "keep_alive": "30m",
            }
            if json_schema:
                cuerpo["format"] = json_schema
            r = httpx.post(f"{self.url}/api/chat", json=cuerpo, timeout=config.TIMEOUT_LLM)
            r.raise_for_status()
            d = r.json()
        except httpx.ConnectError as e:
            raise ErrorLLM(f"No hay conexión con Ollama ({self.url}). ¿Está abierto?") from e
        except httpx.HTTPStatusError as e:
            raise ErrorLLM(f"Ollama respondió {e.response.status_code}: {e.response.text[:300]}") from e
        salida = {"texto": d["message"]["content"], "segundos": round(time.perf_counter() - t0, 2),
                  "tokens_entrada": d.get("prompt_eval_count", 0), "tokens_salida": d.get("eval_count", 0)}
        self._sumar(salida, grabada=False)
        if grabar():
            GRABADORA.guardar(clave, salida)
        return salida

    def _sumar(self, s: dict, grabada: bool) -> None:
        self.uso.llamadas += 1
        self.uso.desde_grabacion += int(grabada)
        self.uso.segundos += s.get("segundos", 0)
        self.uso.tokens_entrada += s.get("tokens_entrada", 0)
        self.uso.tokens_salida += s.get("tokens_salida", 0)

    def disponible(self) -> dict:
        if self.proveedor == "grabado":
            return {"proveedor": "grabado", "ok": True, "grabadas": len(GRABADORA.cargar())}
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=5)
            modelos = [m["name"] for m in r.json().get("models", [])]
            return {"proveedor": "ollama", "ok": self.modelo in modelos, "modelos": modelos}
        except Exception as e:  # noqa: BLE001
            return {"proveedor": "ollama", "ok": False, "error": str(e)}


def embeber(textos: list[str], modelo: str = config.MODELO_EMB, url: str = config.OLLAMA_URL,
            lote: int = 16) -> list[list[float]]:
    """Embeddings con caché en disco por huella (modelo + texto): reanudable y reproducible sin Ollama."""
    claves = [huella(modelo, t) for t in textos]
    faltan = [i for i, c in enumerate(claves) if CACHE_EMB.get(c) is None]
    for ini in range(0, len(faltan), lote):
        idx = faltan[ini:ini + lote]
        try:
            r = httpx.post(f"{url}/api/embed", json={"model": modelo, "input": [textos[i] for i in idx],
                                                     "keep_alive": "30m", "truncate": True}, timeout=600)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise ErrorLLM(f"No hay conexión con Ollama ({url}) para calcular embeddings.") from e
        for i, v in zip(idx, r.json()["embeddings"]):
            CACHE_EMB.guardar(claves[i], [round(x, 5) for x in v])
    return [CACHE_EMB.get(c) for c in claves]


def extraer_json(texto: str) -> dict:
    """Tolera texto alrededor del JSON (no debería pasar con salida estructurada, pero un 7B a veces se corta)."""
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        raise ErrorLLM(f"El modelo no devolvió JSON: {texto[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ErrorLLM(f"JSON inválido del modelo: {texto[:200]}") from e
