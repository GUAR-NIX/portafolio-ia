"""Las cifras clave del README salen de reports/, y la evaluación se reproduce desde lo grabado (sin Ollama)."""
import json
from pathlib import Path

import pytest

from kpis import config, evaluacion, generadores

RAIZ = Path(__file__).resolve().parents[1]
RES = json.loads((RAIZ / "reports" / "resultados.json").read_text(encoding="utf-8"))
README = (RAIZ / "README.md").read_text(encoding="utf-8")
GUIA = (RAIZ / "docs" / "GUIA_PASO_A_PASO.md").read_text(encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def ic(par) -> str:
    return f"({pct(par[0]).replace(' %', '')}–{pct(par[1])})"


@pytest.mark.parametrize("enfoque", generadores.ENFOQUES)
def test_readme_cifras_de_prueba(enfoque):
    p = RES["prueba"][enfoque]
    assert f"{pct(p['tasa_alucinacion'])} {ic(p['tasa_alucinacion_ic'])}" in README, enfoque
    assert f"{pct(p['cobertura'])} {ic(p['cobertura_ic'])}" in README, enfoque
    assert pct(p["informes_sin_error"]) in README, enfoque


def test_readme_version_y_validacion():
    assert f"`{RES['version_prompt_elegida']}`" in README
    for clave in ("estructurado_v1", "estructurado_v2"):
        assert pct(RES["validacion"][clave]["tasa_alucinacion"]) in README, clave


def test_readme_legibilidad():
    for enf, fila in RES["legibilidad"].items():
        if enf == "n_informes":
            continue
        assert f"{fila['total']:.2f}".replace(".", ",") in README, enf


def test_guia_tiene_8_preguntas_o_mas():
    assert GUIA.count("\n### P") >= 8


def test_sin_rutas_personales():
    usuario = "gu" + "arn"  # armado por partes para que este archivo no contenga la ruta
    archivos = (list(RAIZ.glob("*.md")) + list((RAIZ / "docs").glob("*.md")) + list((RAIZ / "src").rglob("*.py"))
                + list((RAIZ / "notebooks").glob("*.ipynb")) + list((RAIZ / "reports").glob("*.json")))
    for ruta in archivos:
        texto = ruta.read_text(encoding="utf-8")
        assert f"Users\\{usuario}" not in texto and f"Users/{usuario}" not in texto, ruta


def test_evaluacion_se_reproduce_desde_grabaciones():
    """Regenera (b) y (c) de un informe de prueba desde data/grabacion_llm.jsonl y compara con reports/."""
    import pandas as pd
    det = pd.read_csv(RAIZ / "reports" / "detalle_informes.csv")
    pais, anio = config.PRUEBA[0]
    for enf in ("libre", "estructurado", "verificado"):
        r = generadores.generar(enf, pais, anio, RES["version_prompt_elegida"], proveedor="grabado")
        fila = det[(det.conjunto == "prueba") & (det.enfoque == enf) & (det.pais == pais)].iloc[0]
        assert r["verificacion"]["n_cifras"] == fila["n_cifras"]
        assert r["verificacion"]["n_correcta"] == fila["n_correcta"]
    assert evaluacion.elegir_version_prompt() == RES["version_prompt_elegida"]
