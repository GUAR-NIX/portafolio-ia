"""El corpus versionado y el conjunto de evaluación cumplen lo que promete el README."""
import json

import pandas as pd

from rag_laboral import config
from rag_laboral.evaluar_recuperacion import cargar_preguntas, metricas, posicion
from rag_laboral.recuperar import cargar_articulos


def test_corpus_por_articulo_sin_basura():
    arts = cargar_articulos()
    ids = [a["id"] for a in arts]
    assert len(ids) == len(set(ids)) >= 600
    por_id = {a["id"]: a for a in arts}
    assert "quince días de descanso" in por_id["CT-69"]["texto"]
    assert por_id["CT-111"]["epigrafe"].startswith("Derecho a la décima tercera remuneración")
    for a in arts:
        assert "Página" not in a["texto"] or "Página web" in a["texto"]
        assert "JURISPRUDENCIA:" not in a["texto"] and "CONCORDANCIAS:" not in a["texto"]
        assert "LEXIS" not in a["texto"]
    assert {a["fuente"] for a in arts} == {"Código del Trabajo", "Constitución de la República"}


def test_conjunto_de_evaluacion_tamano_y_formato():
    q = cargar_preguntas()
    princ = q[q.conjunto == "principal"]
    assert (princ.tipo == "respondible").sum() >= 80
    assert princ.tipo.str.startswith("trampa").sum() >= 15
    assert q.id.is_unique
    assert (q[q.tipo == "respondible"].respuesta_referencia.str.len() > 10).all()
    assert (q[q.tipo != "respondible"].articulos_correctos == "").all()


def test_cada_respuesta_correcta_existe_y_su_evidencia_es_literal():
    """Nunca inventar artículos: el artículo correcto existe en el índice y la evidencia está en su texto."""
    por_id = {a["id"]: a for a in cargar_articulos()}
    for r in cargar_preguntas().itertuples():
        for c in r.correctos:
            assert c in por_id, (r.id, c)
        if r.correctos:
            a = por_id[r.correctos[0]]
            assert r.evidencia in f"{a['epigrafe']} {a['texto']}", r.id


def test_metricas_recall_y_mrr():
    assert posicion(["CT-1", "CT-69", "CT-70"], ["CT-69"]) == 2
    assert posicion(["CT-1"], ["CT-69"]) is None
    m = metricas([1, 2, None, 6])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.75
    assert abs(m["mrr@20"] - (1 + 0.5 + 0 + 1 / 6) / 4) < 1e-4


def test_readme_usa_las_cifras_de_reports():
    readme = (config.RAIZ / "README.md").read_text(encoding="utf-8")
    t = pd.DataFrame(json.loads((config.REPORTS / "recuperacion.json").read_text(encoding="utf-8"))["resultados"])
    fila = t[(t.conjunto == "principal") & (t.metodo == "rerank") & (t.sinonimos == "v2")].iloc[0]
    assert f"**{fila['recall@5']:.2f}**".replace(".", ",") in readme
    assert f"**{fila['mrr@20']:.2f}**".replace(".", ",") in readme
    gen = json.loads((config.REPORTS / "generacion.json").read_text(encoding="utf-8"))
    p = gen["por_conjunto"]["principal"]
    assert f"**{p['sin_umbral']['cita_exacta']:.0%}**".replace("%", " %") in readme
    assert f"{p['sin_umbral']['rechazo_correcto']:.0%} → **{p['con_umbral']['rechazo_correcto']:.0%}**".replace("%", " %") in readme
    assert f"τ = {gen['umbral']['valor']}".replace(".", ",") in readme
    assert f"{gen['segundos_llm']['mediana']:.0f} s" in readme


def test_cifras_de_reports_son_coherentes():
    rec = json.loads((config.REPORTS / "recuperacion.json").read_text(encoding="utf-8"))
    t = pd.DataFrame(rec["resultados"])
    assert set(t.metodo) == {"bm25", "vectorial", "hibrido", "rerank"}
    assert t["recall@5"].between(0, 1).all() and (t["recall@5"] <= t["recall@10"]).all()
    gen = json.loads((config.REPORTS / "generacion.json").read_text(encoding="utf-8"))
    p = gen["por_conjunto"]["principal"]["sin_umbral"]
    assert p["respondibles"] >= 80 and p["trampas"] >= 15
    assert gen["errores"] == 0


def test_ic_bootstrap_de_recall_y_mrr():
    pos = [1, 1, 2, 3, None, 7, 1, 4, 1, 2]
    m = metricas(pos)
    lo, hi = m["ic95_recall@5"]
    assert lo <= m["recall@5"] <= hi
    lo, hi = m["ic95_mrr@20"]
    assert lo <= m["mrr@20"] <= hi
    assert metricas(pos) == m                         # semilla fija: mismo resultado
    assert metricas([1] * 10)["ic95_recall@5"] == [1.0, 1.0]
    assert "ic95_recall@5" not in metricas(pos, bootstrap=False)


def test_reports_traen_ic_y_diferencias_pareadas():
    rec = json.loads((config.REPORTS / "recuperacion.json").read_text(encoding="utf-8"))
    for f in rec["resultados"]:
        for m in ("recall@5", "mrr@20"):
            lo, hi = f[f"ic95_{m}"]
            assert lo <= f[m] <= hi, f
    d = {(x["conjunto"], x["a"], x["b"], x["metrica"]): x for x in rec["diferencias_pareadas"]}
    x = d[("principal", "rerank", "vectorial", "recall@5")]
    assert x["diferencia"] == 0 and x["ic95"][0] < 0 < x["ic95"][1]      # empatan en Recall@5, como dice el README


def test_readme_cita_los_ic_de_reports():
    readme = (config.RAIZ / "README.md").read_text(encoding="utf-8")
    rec = json.loads((config.REPORTS / "recuperacion.json").read_text(encoding="utf-8"))
    f = next(x for x in rec["resultados"] if (x["conjunto"], x["metodo"], x["sinonimos"]) == ("principal", "rerank", "v2"))
    for m in ("recall@5", "mrr@20"):
        lo, hi = f[f"ic95_{m}"]
        assert f"[{lo:.2f}–{hi:.2f}]".replace(".", ",") in readme
