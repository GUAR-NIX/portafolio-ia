"""Pruebas de los endpoints de la API FastAPI (/salud, /modelo, /clasificar y /enrutar_lote)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)


def test_api_salud():
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_api_modelo():
    r = cliente.get("/modelo")
    assert r.status_code == 200
    d = r.json()
    assert "categorias" in d
    assert "comparacion_enfoques" in d


def test_api_clasificar_queja_clara_y_ambigua():
    # Queja clara de hipoteca
    r1 = cliente.post(
        "/clasificar",
        json={
            "id_queja": "Q1",
            "texto_queja": (
                "My mortgage loan servicer miscalculated my escrow property tax account "
                "and threatened foreclosure on my home loan after I applied for refinancing."
            ),
            "enfoque": "tfidf",
            "umbral_abstencion": 0.50,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["categoria_principal"] == "hipotecas"
    assert d1["enrutar_automaticamente"] is True

    # Con umbral muy alto (0.95) debe derivarse a triaje humano
    r2 = cliente.post(
        "/clasificar",
        json={
            "id_queja": "Q2",
            "texto_queja": "I have an issue with an account balance and a payment that was not credited.",
            "enfoque": "tfidf",
            "umbral_abstencion": 0.92,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["enrutar_automaticamente"] is False
    assert "Triaje humano" in d2["destino_operativo"]


def test_api_clasificar_rechaza_texto_corto():
    r = cliente.post("/clasificar", json={"id_queja": "Q_ERR", "texto_queja": "corto"})
    assert r.status_code == 422


def test_api_enrutar_lote():
    lote = {
        "quejas": [
            {
                "id_queja": "L1",
                "texto_queja": "Equifax and TransUnion have inaccurate credit report inquiries lowering my credit score.",
            },
            {
                "id_queja": "L2",
                "texto_queja": "A debt collection agency calls my cell phone every day about a medical debt I do not owe.",
            },
        ]
    }
    r = cliente.post("/enrutar_lote?enfoque=tfidf", json=lote)
    assert r.status_code == 200
    body = r.json()
    assert body["total_quejas"] == 2
    assert len(body["resultados"]) == 2
    assert body["resultados"][0]["categoria_principal"] == "reporte_credito"
    assert body["resultados"][1]["categoria_principal"] == "cobranza_deudas"


def test_api_enrutar_cascada_con_urgencia():
    r = cliente.post(
        "/enrutar",
        json={"texto_queja": "The bank started foreclosure on my home even though my loan modification "
                             "application with the mortgage servicer is still under review."},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["categoria_final"] == "hipotecas"
    assert d["urgente"] is True
    assert d["prioridad"].startswith("alta")
    assert {"pasos", "decidido_por", "umbral_cascada", "clasificador_barato"} <= set(d)
