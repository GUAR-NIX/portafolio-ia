"""
descargar_datos.py
Descarga el dataset público de quejas financieras CFPB (Consumer Financial Protection Bureau)
desde el espejo Parquet en Hugging Face (milesbutler/consumer_complaints), consolida los
productos del CFPB en 6 mesas, deriva la etiqueta de urgencia y construye las particiones
TEMPORALES de trabajo en data/raw/quejas_cfpb.parquet:

    train             quejas recibidas antes del 01-04-2017 (taxonomía antigua del CFPB),
                      menos una reserva aleatoria de 3.000 quejas
    test_misma_epoca  esas 3.000 quejas reservadas (misma época que train: referencia sin deriva)
    val               01-04-2017 a 30-09-2017 (para elegir umbrales y calibrar)
    test              01-10-2017 en adelante (quejas recientes: evaluación principal)

Dentro de test se marca un subconjunto común balanceado (80 por mesa, en orden intercalado por
mesa) que se evalúa con TODOS los enfoques, incluido el LLM.

Uso:
    python descargar_datos.py            # no hace nada si la muestra ya existe
    python descargar_datos.py --rehacer  # reconstruye la muestra desde el Parquet original
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL_PARQUET = (
    "https://huggingface.co/datasets/milesbutler/consumer_complaints/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
RAIZ = Path(__file__).resolve().parent
CARPETA_RAW = RAIZ / "data" / "raw"
PARQUET_ORIGEN = CARPETA_RAW / "cfpb_test_full.parquet"
PARQUET_MUESTRA = CARPETA_RAW / "quejas_cfpb.parquet"
RESUMEN_FUENTE = CARPETA_RAW / "resumen_fuente_cfpb.json"
SEMILLA = 42

# Agrupación PROPIA (no oficial del CFPB) de los productos del CFPB en 6 mesas de atención.
# El CFPB renombró varios productos el 24-04-2017; ambos nombres van a la misma mesa.
MAPEO_PRODUCTO = {
    "Credit reporting, credit repair services, or other personal consumer reports": "reporte_credito",
    "Credit reporting": "reporte_credito",
    "Debt collection": "cobranza_deudas",
    "Mortgage": "hipotecas",
    "Credit card or prepaid card": "tarjetas_credito",
    "Credit card": "tarjetas_credito",
    "Prepaid card": "tarjetas_credito",
    "Checking or savings account": "cuentas_bancarias",
    "Bank account or service": "cuentas_bancarias",
    "Money transfer, virtual currency, or money service": "cuentas_bancarias",
    "Money transfers": "cuentas_bancarias",
    "Student loan": "prestamos_consumo",
    "Vehicle loan or lease": "prestamos_consumo",
    "Consumer Loan": "prestamos_consumo",
    "Payday loan, title loan, or personal loan": "prestamos_consumo",
    "Payday loan": "prestamos_consumo",
}
# Excluidos por no encajar en ninguna mesa: "Other financial service" y "Virtual currency".

CATEGORIAS_ORDEN = [
    "reporte_credito",
    "cobranza_deudas",
    "hipotecas",
    "tarjetas_credito",
    "cuentas_bancarias",
    "prestamos_consumo",
]

# ---------------------------------------------------------------------------
# Etiqueta de urgencia: se deriva de los campos Issue / Sub-issue que el propio consumidor
# eligió en el formulario del CFPB (no del texto), así que sirve como verdad de referencia
# independiente para evaluar detectores basados en el texto.
# ---------------------------------------------------------------------------
URGENCIA_POR_MOTIVO = {
    # Fraude y robo de identidad
    "Identity theft / Fraud / Embezzlement": "fraude",
    "Fraud or scam": "fraude",
    # Riesgo de ejecución hipotecaria (foreclosure)
    "Loan modification,collection,foreclosure": "ejecucion_hipotecaria",
    "Struggling to pay mortgage": "ejecucion_hipotecaria",
    # Reposesión de vehículo
    "Vehicle was repossessed or sold the vehicle": "reposesion",
    "Lender repossessed or sold the vehicle": "reposesion",
}
URGENCIA_POR_SUBMOTIVO = {
    "Debt resulted from identity theft": "fraude",
    "Debt was result of identity theft": "fraude",
    "Card opened as result of identity theft or fraud": "fraude",
    "Account opened as a result of fraud": "fraude",
    "Fraudulent loan": "fraude",
    "Threatened arrest/jail if do not pay": "amenaza_legal_embargo",
    "Threatened to arrest you or take you to jail if you do not pay": "amenaza_legal_embargo",
    "Seized/Attempted to seize property": "amenaza_legal_embargo",
    "Seized or attempted to seize your property": "amenaza_legal_embargo",
    "Attempted to/Collected exempt funds": "amenaza_legal_embargo",
    "Collected or attempted to collect exempt funds": "amenaza_legal_embargo",
    "Sued w/o proper notification of suit": "amenaza_legal_embargo",
    "Sued you without properly notifying you of lawsuit": "amenaza_legal_embargo",
    "Lender trying to repossess or disable the vehicle": "reposesion",
    "Loan balance remaining after the vehicle is repossessed and sold": "reposesion",
}

FECHA_CORTE_TRAIN = pd.Timestamp("2017-04-01")
FECHA_CORTE_TEST = pd.Timestamp("2017-10-01")
N_TEST_MISMA_EPOCA = 3000
N_COMUN_POR_CLASE = 80
N_VAL_EMB_POR_CLASE = 100
N_TRAIN_EMB_POR_CLASE = 300


def _parquet_completo(ruta: Path) -> bool:
    """Un Parquet termina con la marca PAR1: un archivo cortado no la tiene."""
    if not ruta.exists() or ruta.stat().st_size < 12:
        return False
    with open(ruta, "rb") as f:
        f.seek(-4, 2)
        return f.read(4) == b"PAR1"


def descargar_parquet(url: str, destino: Path) -> None:
    if _parquet_completo(destino):
        print(f"[OMITIDO] {destino.name} ya existe ({destino.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Descargando {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    parcial = destino.with_suffix(destino.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as r, open(parcial, "wb") as f:
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
    if not _parquet_completo(parcial):
        raise RuntimeError(f"La descarga de {destino.name} quedó incompleta; vuelve a ejecutar el script")
    parcial.replace(destino)
    print(f"  Descargado: {destino.stat().st_size / 1e6:.1f} MB")


def tipo_urgencia(motivo: str, submotivo: str) -> str:
    return URGENCIA_POR_SUBMOTIVO.get(submotivo) or URGENCIA_POR_MOTIVO.get(motivo) or ""


def _muestra_intercalada(df: pd.DataFrame, por_clase: int, rng: np.random.Generator) -> pd.Series:
    """Devuelve un orden 0..(6*por_clase-1) intercalado por mesa: cualquier prefijo múltiplo de 6 queda balanceado."""
    orden = pd.Series(-1, index=df.index, dtype=int)
    for j, cat in enumerate(CATEGORIAS_ORDEN):
        idx = df.index[df["categoria"] == cat].to_numpy()
        elegidos = rng.choice(idx, size=min(por_clase, len(idx)), replace=False)
        for k, i in enumerate(elegidos):
            orden.loc[i] = k * len(CATEGORIAS_ORDEN) + j
    return orden


def construir_corpus(df: pd.DataFrame, semilla: int = SEMILLA) -> tuple[pd.DataFrame, dict]:
    total_filas_origen = len(df)
    df = df.rename(columns={
        "Complaint ID": "id_queja",
        "Date received": "fecha_texto",
        "Product": "producto_original",
        "Sub-product": "subproducto",
        "Issue": "motivo_original",
        "Sub-issue": "submotivo",
        "Consumer Complaint": "texto_queja",
        "Company": "empresa",
        "State": "estado_eeuu",
        "Submitted via": "canal_envio",
        "Company Response to Consumer": "respuesta_empresa",
        "Timely response?": "respuesta_a_tiempo",
    }).copy()
    # El Parquet mezcla los formatos 01-31-2016 y 01/31/2016
    df["fecha_recepcion"] = pd.to_datetime(df["fecha_texto"].str.replace("-", "/"), format="%m/%d/%Y")

    df["categoria"] = df["producto_original"].map(MAPEO_PRODUCTO)
    dist_original = df["producto_original"].value_counts(dropna=False).to_dict()

    df = df[df["categoria"].isin(CATEGORIAS_ORDEN)].copy()
    df["texto_queja"] = df["texto_queja"].fillna("").astype(str).str.strip()
    df["n_palabras"] = df["texto_queja"].str.split().str.len()

    # Narrativas entre 15 y 260 palabras: evita textos vacíos y expedientes de miles de palabras
    valido = df[(df["n_palabras"] >= 15) & (df["n_palabras"] <= 260)].copy()
    valido = valido.drop_duplicates(subset=["texto_queja"], keep="first").reset_index(drop=True)
    for c in ("motivo_original", "submotivo", "subproducto"):
        valido[c] = valido[c].fillna("").astype(str)
    valido["tipo_urgencia"] = [
        tipo_urgencia(m, s) for m, s in zip(valido["motivo_original"], valido["submotivo"])
    ]
    valido["urgente"] = valido["tipo_urgencia"] != ""

    rng = np.random.default_rng(semilla)
    antiguas = valido["fecha_recepcion"] < FECHA_CORTE_TRAIN
    recientes = valido["fecha_recepcion"] >= FECHA_CORTE_TEST
    valido["particion"] = np.where(antiguas, "train", np.where(recientes, "test", "val"))
    reserva = rng.choice(valido.index[antiguas].to_numpy(), size=N_TEST_MISMA_EPOCA, replace=False)
    valido.loc[reserva, "particion"] = "test_misma_epoca"

    # Subconjuntos para los enfoques caros (embeddings y LLM), con orden intercalado por mesa
    valido["orden_comun"] = _muestra_intercalada(valido[valido["particion"] == "test"], N_COMUN_POR_CLASE, rng).reindex(valido.index, fill_value=-1)
    valido["orden_val_emb"] = _muestra_intercalada(valido[valido["particion"] == "val"], N_VAL_EMB_POR_CLASE, rng).reindex(valido.index, fill_value=-1)
    valido["orden_train_emb"] = _muestra_intercalada(valido[valido["particion"] == "train"], N_TRAIN_EMB_POR_CLASE, rng).reindex(valido.index, fill_value=-1)

    columnas_guardar = [
        "id_queja", "fecha_recepcion", "categoria", "producto_original", "subproducto",
        "motivo_original", "submotivo", "tipo_urgencia", "urgente", "empresa", "estado_eeuu",
        "canal_envio", "respuesta_empresa", "respuesta_a_tiempo", "n_palabras", "texto_queja",
        "particion", "orden_comun", "orden_val_emb", "orden_train_emb",
    ]
    final = valido[columnas_guardar].copy()

    resumen = {
        "fuente_url": URL_PARQUET,
        "filas_totales_parquet": int(total_filas_origen),
        "fecha_min": str(final["fecha_recepcion"].min().date()),
        "fecha_max": str(final["fecha_recepcion"].max().date()),
        "filas_validas_15_260_palabras": int(len(final)),
        "distribucion_original_por_producto": {str(k): int(v) for k, v in dist_original.items()},
        "filas_por_particion": final["particion"].value_counts().to_dict(),
        "distribucion_por_categoria": final["categoria"].value_counts().to_dict(),
        "pct_urgente": float(final["urgente"].mean() * 100.0),
    }
    return final, resumen


def main() -> None:
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)
    rehacer = "--rehacer" in sys.argv
    if PARQUET_MUESTRA.exists() and RESUMEN_FUENTE.exists() and not rehacer:
        print(f"[OK] {PARQUET_MUESTRA} ya existe ({PARQUET_MUESTRA.stat().st_size / 1e6:.2f} MB)")
        return
    descargar_parquet(URL_PARQUET, PARQUET_ORIGEN)
    print("Leyendo Parquet del CFPB y construyendo particiones temporales...")
    df = pd.read_parquet(PARQUET_ORIGEN)
    corpus, resumen = construir_corpus(df)
    corpus.to_parquet(PARQUET_MUESTRA, index=False)
    RESUMEN_FUENTE.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Guardado {PARQUET_MUESTRA} ({len(corpus):,} filas)")
    print(json.dumps(resumen["filas_por_particion"], indent=2))


if __name__ == "__main__":
    main()
