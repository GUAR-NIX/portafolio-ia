# Todo el proyecto en un comando (Windows).  Uso:
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1            # reproduce con lo grabado (~20 min)
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -ConLLM    # vuelve a llamar a Ollama (~3 h)
param([switch]$ConLLM)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) { py -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -m pip install -q -e .
$env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python descargar_datos.py                 # 13 GB en streaming -> ~7 MB (reanudable)
.\.venv\Scripts\python -m pytest -q                       # pruebas, sin Ollama
.\.venv\Scripts\python -m resenas.muestra                 # texto de las 700 reseñas etiquetadas (verificado con huellas)
if ($ConLLM) {
    $env:RESENAS_GRABAR = "1"
    .\.venv\Scripts\python -m resenas.enfoque_embeddings  # embeddings bge-m3 que falten en la caché
    .\.venv\Scripts\python -m resenas.correr_llm          # qwen2.5:7b sobre dev(40) + test + español
}
.\.venv\Scripts\python -m resenas.evaluar                 # métricas y figuras 01-03
.\.venv\Scripts\python -m resenas.aplicar                 # enfoque clásico sobre las 55 mil reseñas
.\.venv\Scripts\python -m resenas.hallazgos               # tablas y figuras 04-08
Write-Host "Listo. App:  .\.venv\Scripts\streamlit run app\streamlit_app.py"
