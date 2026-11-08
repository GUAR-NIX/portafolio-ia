# Reproduce todo el proyecto en un comando (Windows).
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1           # usa instantáneas y respuestas grabadas del LLM (~1 min)
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -ConLLM   # completa con Ollama lo que falte grabar (horas en CPU)
param([switch]$ConLLM)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -3.14 -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -m pip install -q -e .
$env:PYTHONIOENCODING = "utf-8"

if ($ConLLM) {
    Set-Location src
    ..\.venv\Scripts\python -m agente.correr_llm validacion
    ..\.venv\Scripts\python -m agente.correr_llm prueba_agente
    ..\.venv\Scripts\python -m agente.correr_llm prueba_base
    Set-Location $PSScriptRoot
}
.\.venv\Scripts\python -m agente.ejecutar_evaluacion
.\.venv\Scripts\python -m pytest -q
Write-Host "Listo. API: .\.venv\Scripts\uvicorn api.main:app --reload  |  App: .\.venv\Scripts\streamlit run app\app_streamlit.py"
