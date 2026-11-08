"""
Cliente LLM mínimo (solo httpx), con el mismo patrón del proyecto IA/:

* Ollama (por defecto, local y gratis): OLLAMA_URL, FACTURAS_MODELO=qwen2.5:7b
* Grabado (demo y pruebas sin Ollama):  FACTURAS_PROVEEDOR=grabado → reproduce data/grabacion_llm.jsonl

Con FACTURAS_GRABAR=1 cada llamada real se guarda por huella del contenido (modelo + prompt + esquema): la evaluación
se puede reanudar si se corta y cualquiera puede reproducirla sin Ollama.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from . import config


def _clave(*partes) -> str:
    return hashlib.sha1(json.dumps(partes, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Grabadora:
    _cache: dict | None = None

    @classmethod
    def cargar(cls) -> dict:
        if cls._cache is None:
            cls._cache = {}
            if config.GRABACION.exists():
                for ln in config.GRABACION.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        d = json.loads(ln)
                        cls._cache[d["clave"]] = d["salida"]
        return cls._cache

    @classmethod
    def guardar(cls, clave: str, salida: dict) -> None:
        cache = cls.cargar()
        if clave in cache:
            return
        cache[clave] = salida
        config.GRABACION.parent.mkdir(parents=True, exist_ok=True)
        with open(config.GRABACION, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clave": clave, "salida": salida}, ensure_ascii=False) + "\n")


def _grabar() -> bool:
    return os.environ.get("FACTURAS_GRABAR", "0") == "1"


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
             max_tokens: int = 1500) -> str:
        """Devuelve el texto de la respuesta. Las respuestas grabadas guardan también segundos y tokens."""
        clave = _clave(self.modelo, sistema, mensajes, json_schema, max_tokens)
        if self.proveedor == "grabado" or _grabar():
            previa = Grabadora.cargar().get(clave)
            if previa is not None:
                self.uso.llamadas += 1
                self.uso.desde_grabacion += 1
                self.uso.segundos += previa.get("segundos", 0)
                self.uso.tokens_entrada += previa.get("tokens_entrada", 0)
                self.uso.tokens_salida += previa.get("tokens_salida", 0)
                return previa["texto"]
            if self.proveedor == "grabado":
                raise ErrorLLM("Modo demo: esta factura no está grabada. Abre Ollama para procesar facturas nuevas.")
        t0 = time.perf_counter()
        try:
            if self.proveedor != "ollama":
                raise ErrorLLM(f"Proveedor desconocido: {self.proveedor}")
            texto, te, ts = self._ollama(sistema, mensajes, json_schema, max_tokens)
        except httpx.ConnectError as e:
            raise ErrorLLM(f"No hay conexión con Ollama ({self.url}). ¿Está abierto?") from e
        except httpx.HTTPStatusError as e:
            raise ErrorLLM(f"Ollama respondió {e.response.status_code}: {e.response.text[:300]}") from e
        seg = time.perf_counter() - t0
        self.uso.llamadas += 1
        self.uso.segundos += seg
        self.uso.tokens_entrada += te
        self.uso.tokens_salida += ts
        if _grabar():
            Grabadora.guardar(clave, {"texto": texto, "segundos": round(seg, 2), "tokens_entrada": te,
                                      "tokens_salida": ts})
        return texto

    def _ollama(self, sistema, mensajes, json_schema, max_tokens):
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
        return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)

    def disponible(self) -> dict:
        if self.proveedor == "grabado":
            return {"proveedor": "grabado", "ok": True, "grabadas": len(Grabadora.cargar())}
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=5)
            modelos = [m["name"] for m in r.json().get("models", [])]
            return {"proveedor": "ollama", "ok": self.modelo in modelos, "modelos": modelos}
        except Exception as e:  # noqa: BLE001
            return {"proveedor": "ollama", "ok": False, "error": str(e)}


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
