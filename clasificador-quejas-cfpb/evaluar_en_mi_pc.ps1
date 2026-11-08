# Reproduce todo el proyecto en un comando (Windows).
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1           # usa embeddings y respuestas del LLM grabadas (~2 min)
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -ConLLM   # completa lo que falte consultando Ollama local (horas en CPU)
param([switch]$ConLLM)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -3.14 -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -m pip install -q -e .
$env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python descargar_datos.py
if ($ConLLM) {
    # Reanudables: lo ya grabado en data/ no se vuelve a pedir
    .\.venv\Scripts\python -m quejas.grabar embeddings
    .\.venv\Scripts\python -m quejas.grabar llm --hasta 300
    .\.venv\Scripts\python -m quejas.grabar llm_es
}
$env:QUEJAS_PROVEEDOR = "grabado"
.\.venv\Scripts\python -m quejas.entrenar
.\.venv\Scripts\python -m quejas.informe
.\.venv\Scripts\python -m pytest -q
Write-Host "Listo. API: .\.venv\Scripts\uvicorn api.main:app --reload   ·   App: .\.venv\Scripts\streamlit run app\app_streamlit.py"
