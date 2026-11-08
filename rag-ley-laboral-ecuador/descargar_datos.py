"""
Descarga los PDF oficiales de la ley laboral del Ecuador a data/raw/ (reanudable: si el archivo ya está, no se baja).

    python descargar_datos.py

Guarda la URL, la fecha de descarga, el tamaño y la huella SHA-256 en data/fuentes_descarga.json
(docs/fuentes.md explica cada fuente y su fecha de última reforma).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent
RAW = RAIZ / "data" / "raw"
REGISTRO = RAIZ / "data" / "fuentes_descarga.json"

FUENTES = {
    # Codificación 2005-017 (R.O. Supl. 167, 16-XII-2005), texto actualizado publicado por el SERCOP (portal estatal)
    "codigo_trabajo.pdf": "https://portal.compraspublicas.gob.ec/sercop/wp-content/uploads/2026/08/CODIGO-DEL-TRABAJO.pdf",
    # Constitución 2008 actualizada (enero 2021), publicada por el Ministerio de Defensa
    "constitucion.pdf": "https://www.defensa.gob.ec/wp-content/uploads/downloads/2021/02/Constitucion-de-la-Republica-del-Ecuador_act_ene-2021.pdf",
}

CABECERAS = {"User-Agent": "Mozilla/5.0 (portafolio de datos; descarga de normativa publica)"}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def bajar_reanudable(url: str, parcial: Path, intentos: int = 30) -> None:
    """Los portales del Estado son lentos y cortan seguido: se continúa desde el último byte (Range).

    Si hay curl en el sistema (Windows 10+ lo trae) se usa `curl -C -`, que en el portal del SERCOP resultó más
    estable que httpx; si no, el mismo procedimiento con httpx."""
    if shutil.which("curl"):
        for intento in range(1, intentos + 1):
            r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "-A", CABECERAS["User-Agent"],
                                "--speed-time", "120", "--speed-limit", "200", "-o", str(parcial), url])
            if r.returncode == 0:
                return
            print(f"  intento {intento}: curl salió con {r.returncode}, llevo {parcial.stat().st_size if parcial.exists() else 0} bytes")
        raise RuntimeError(f"No se pudo completar la descarga de {url}")
    total = None
    for intento in range(1, intentos + 1):
        ya = parcial.stat().st_size if parcial.exists() else 0
        if total is not None and ya >= total:
            return
        cab = dict(CABECERAS, **({"Range": f"bytes={ya}-"} if ya else {}))
        try:
            with httpx.stream("GET", url, headers=cab, timeout=httpx.Timeout(60, read=60),
                              follow_redirects=True) as r:
                r.raise_for_status()
                if ya and r.status_code != 206:  # el servidor no acepta Range: empezar de cero
                    ya = 0
                    parcial.unlink(missing_ok=True)
                if r.status_code == 206:
                    total = int(r.headers["Content-Range"].split("/")[-1])
                else:
                    total = int(r.headers.get("Content-Length", 0)) or None
                with open(parcial, "ab") as fh:
                    for bloque in r.iter_bytes():
                        fh.write(bloque)
            if total is None or parcial.stat().st_size >= total:
                return
        except (httpx.TimeoutException, httpx.TransportError) as e:
            print(f"  intento {intento}: {type(e).__name__}, llevo {parcial.stat().st_size if parcial.exists() else 0} bytes")
    raise RuntimeError(f"No se pudo completar la descarga de {url}")


def descargar() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    registro = json.loads(REGISTRO.read_text(encoding="utf-8")) if REGISTRO.exists() else {}
    for nombre, url in FUENTES.items():
        destino = RAW / nombre
        if destino.exists() and destino.stat().st_size > 10_000 and nombre in registro:
            print(f"ya descargado: {nombre}")
            continue
        print(f"descargando {nombre} <- {url}")
        parcial = destino.with_suffix(".part")
        bajar_reanudable(url, parcial)
        parcial.replace(destino)
        registro[nombre] = {
            "url": url,
            "fecha_descarga": dt.date.today().isoformat(),
            "bytes": destino.stat().st_size,
            "sha256": sha256(destino),
        }
        REGISTRO.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return registro


if __name__ == "__main__":
    for k, v in descargar().items():
        print(k, v["bytes"], "bytes", v["fecha_descarga"])
