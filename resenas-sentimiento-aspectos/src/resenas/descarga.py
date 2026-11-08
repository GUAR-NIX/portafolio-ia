"""
Descarga un subconjunto de Amazon Reviews 2023 (categoría Cell Phones & Accessories) sin guardar los 13 GB.

1. Lee en streaming los metadatos (4 GB) y guarda solo los productos con >= MIN_CALIFICACIONES calificaciones.
2. En cada uno de los 8 TIPOS_PRODUCTO (fundas, protectores de pantalla, cargadores, teléfonos…) elige los
   PRODUCTOS_POR_TIPO productos más calificados, con marca (campo `store`) conocida.
3. Lee en streaming las reseñas (9,3 GB) y, para esos productos, guarda una muestra de reservorio de hasta
   RESENAS_POR_PRODUCTO reseñas cada uno (muestra uniforme en el tiempo, no las primeras del archivo).

Reanudable: el paso 3 guarda un punto de control cada ~500 MB (offset del archivo + estado de los reservorios) y,
si la conexión se corta, sigue con una petición HTTP Range desde la última línea completa.
"""
from __future__ import annotations

import html
import json
import pickle
import random
import re
import time
from collections import Counter

import httpx
import pandas as pd

from . import config

_RE_ASIN = re.compile(rb'"parent_asin": "([A-Z0-9]+)"')
_RE_NCAL = re.compile(rb'"rating_number": (\d+)')


def _lineas(url: str, desde: int = 0, reintentos: int = 20):
    """Recorre un .jsonl remoto línea a línea; devuelve (offset_al_final_de_la_línea, bytes). Se recupera de cortes."""
    offset = desde
    fallos = 0
    while True:
        try:
            with httpx.stream("GET", url, headers={"Range": f"bytes={offset}-"}, follow_redirects=True,
                              timeout=httpx.Timeout(60, read=120)) as r:
                if r.status_code == 416:      # ya estábamos al final
                    return
                r.raise_for_status()
                resto = b""
                for trozo in r.iter_bytes(1 << 20):
                    datos = resto + trozo
                    partes = datos.split(b"\n")
                    resto = partes.pop()
                    for ln in partes:
                        offset += len(ln) + 1
                        if ln.strip():
                            yield offset, ln
                    fallos = 0
                if resto.strip():
                    offset += len(resto)
                    yield offset, resto
                return
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            fallos += 1
            if fallos > reintentos:
                raise
            print(f"  corte de conexión en el byte {offset:,} ({type(e).__name__}); reintento {fallos}…", flush=True)
            time.sleep(min(60, 5 * fallos))


def tamano_remoto(url: str) -> int:
    r = httpx.head(url, follow_redirects=True, timeout=30)
    return int(r.headers.get("content-length", 0))


def limpiar_texto(t: str | None) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


# --------------------------------------------------------------------------- 1. metadatos
def descargar_productos() -> pd.DataFrame:
    if config.PRODUCTOS_RAW.exists():
        print(f"[1/3] productos: ya existe {config.PRODUCTOS_RAW.name}")
        return pd.read_parquet(config.PRODUCTOS_RAW)
    total = tamano_remoto(config.URL_META)
    print(f"[1/3] metadatos en streaming ({total / 1e9:.1f} GB)…", flush=True)
    filas, vistos, t0 = [], 0, time.time()
    for offset, ln in _lineas(config.URL_META):
        vistos += 1
        if vistos % 200_000 == 0:
            print(f"  {vistos:,} productos · {offset / total:.0%} · {time.time() - t0:.0f} s", flush=True)
        m = _RE_NCAL.search(ln)
        if not m or int(m.group(1)) < config.MIN_CALIFICACIONES:
            continue
        d = json.loads(ln)
        cats = d.get("categories") or []
        filas.append({
            "parent_asin": d["parent_asin"],
            "titulo": d.get("title") or "",
            "marca": (d.get("store") or "").strip(),
            "tipo": cats[1] if len(cats) > 1 else "(sin tipo)",
            "subtipo": cats[2] if len(cats) > 2 else "",
            "calificacion_media": d.get("average_rating"),
            "n_calificaciones": d.get("rating_number"),
            "precio": d.get("price"),
        })
    df = pd.DataFrame(filas)
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    config.DIR_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PRODUCTOS_RAW, index=False)
    print(f"  {vistos:,} productos leídos -> {len(df):,} con >= {config.MIN_CALIFICACIONES} calificaciones")
    return df


# --------------------------------------------------------------------------- 2. selección
def tipo_corto(tipo: str, subtipo: str) -> str | None:
    """Nombre corto del tipo de producto, o None si no es uno de los TIPOS_PRODUCTO."""
    for clave, corto in config.TIPOS_PRODUCTO.items():
        partes = clave.split(" > ")
        if partes[0] == tipo and (len(partes) == 1 or partes[1] == subtipo):
            return corto
    return None


def elegir_productos(productos: pd.DataFrame) -> pd.DataFrame:
    p = productos[productos["marca"] != ""].copy()
    p["tipo_producto"] = [tipo_corto(t, s) for t, s in zip(p["tipo"], p["subtipo"])]
    elegidos = (p.dropna(subset=["tipo_producto"]).sort_values("n_calificaciones", ascending=False)
                .groupby("tipo_producto").head(config.PRODUCTOS_POR_TIPO))
    print(f"[2/3] {len(elegidos)} productos de {elegidos['tipo_producto'].nunique()} tipos, "
          f"{elegidos['marca'].nunique()} marcas", flush=True)
    return elegidos.reset_index(drop=True)


# --------------------------------------------------------------------------- 3. reseñas
def descargar_resenas(elegidos: pd.DataFrame) -> pd.DataFrame:
    if config.RESENAS_RAW.exists():
        print(f"[3/3] reseñas: ya existe {config.RESENAS_RAW.name}")
        return pd.read_parquet(config.RESENAS_RAW)
    asins = set(elegidos["parent_asin"])
    cap = config.RESENAS_POR_PRODUCTO
    if config.CHECKPOINT_DESCARGA.exists():
        est = pickle.loads(config.CHECKPOINT_DESCARGA.read_bytes())
        print(f"[3/3] reanudando desde el byte {est['offset']:,}", flush=True)
    else:
        est = {"offset": 0, "lineas": 0, "vistas": Counter(), "reservorio": {a: [] for a in asins},
               "rng": random.Random(config.SEMILLA).getstate()}
    rng = random.Random()
    rng.setstate(est["rng"])
    total = tamano_remoto(config.URL_RESENAS)
    print(f"[3/3] reseñas en streaming ({total / 1e9:.1f} GB) para {len(asins)} productos…", flush=True)
    ultimo_ckpt, t0, b0 = est["offset"], time.time(), est["offset"]

    def guardar_ckpt():
        est["rng"] = rng.getstate()
        tmp = config.CHECKPOINT_DESCARGA.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(est))
        tmp.replace(config.CHECKPOINT_DESCARGA)

    for offset, ln in _lineas(config.URL_RESENAS, est["offset"]):
        est["offset"] = offset
        est["lineas"] += 1
        m = _RE_ASIN.search(ln)
        if m and (a := m.group(1).decode()) in asins:
            n = est["vistas"][a] = est["vistas"][a] + 1
            res = est["reservorio"][a]
            if len(res) < cap:
                res.append(ln)
            else:
                j = rng.randrange(n)
                if j < cap:
                    res[j] = ln
        if offset - ultimo_ckpt > 500_000_000:
            guardar_ckpt()
            ultimo_ckpt = offset
            vel = (offset - b0) / max(1, time.time() - t0) / 1e6
            print(f"  {offset / total:.0%} · {est['lineas']:,} reseñas leídas · "
                  f"{sum(est['vistas'].values()):,} de los productos elegidos · {vel:.1f} MB/s", flush=True)
    guardar_ckpt()

    filas = []
    for a, lista in est["reservorio"].items():
        for ln in lista:
            d = json.loads(ln)
            filas.append({
                "parent_asin": a, "asin": d.get("asin"), "estrellas": int(d["rating"]),
                "titulo_resena": limpiar_texto(d.get("title")), "texto": limpiar_texto(d.get("text")),
                "fecha": pd.to_datetime(d["timestamp"], unit="ms"), "votos_utiles": d.get("helpful_vote", 0),
                "compra_verificada": bool(d.get("verified_purchase")), "con_imagenes": bool(d.get("images")),
            })
    df = pd.DataFrame(filas)
    vistas = pd.Series(est["vistas"], name="resenas_en_amazon_2023")
    df = df.merge(vistas, left_on="parent_asin", right_index=True, how="left")
    df = df.sort_values(["parent_asin", "fecha"]).reset_index(drop=True)
    df.insert(0, "id", [f"R{i:06d}" for i in range(len(df))])
    df.to_parquet(config.RESENAS_RAW, index=False)
    print(f"  {est['lineas']:,} reseñas leídas -> {len(df):,} guardadas ({df['parent_asin'].nunique()} productos)")
    return df


def main() -> None:
    productos = descargar_productos()
    elegidos = elegir_productos(productos)
    resenas = descargar_resenas(elegidos)
    # Los productos guardados son los elegidos, con el número de reseñas que tienen en el archivo de 2023
    conteo = resenas.groupby("parent_asin")["resenas_en_amazon_2023"].first()
    elegidos = elegidos.merge(conteo, left_on="parent_asin", right_index=True, how="left")
    elegidos.to_parquet(config.DIR_RAW / "productos_elegidos.parquet", index=False)
    print("Listo:", config.RESENAS_RAW.relative_to(config.RAIZ).as_posix())


if __name__ == "__main__":
    main()
