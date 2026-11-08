# Ejecuta TODO en tu PC (con Ollama abierto): entorno, datos, índice, pruebas y evaluación. Cada paso es reanudable.
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -Limite 10     (solo 10 preguntas al LLM)
param([int]$Limite = 0)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "`n== 1/7 Entorno virtual y librerias ==" -ForegroundColor Cyan
if (-not (Test-Path $py)) { py -m venv .venv }
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install --quiet -r requirements.txt
& $py -m pip install --quiet -e .

Write-Host "`n== 2/7 Pruebas (sin Ollama, con lo grabado) ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "`n== 3/7 PDF oficiales (SERCOP es lento: ~10 min) ==" -ForegroundColor Cyan
& $py descargar_datos.py

Write-Host "`n== 4/7 Troceo por articulo ==" -ForegroundColor Cyan
& $py -m rag_laboral.extraer

try { Invoke-RestMethod http://localhost:11434/api/tags | Out-Null }
catch { Write-Host "Abre Ollama primero (ollama serve) y vuelve a correr este script." -ForegroundColor Red; exit 1 }
$env:RAG_GRABAR = "1"

Write-Host "`n== 5/7 Embeddings bge-m3 de los articulos (~30 min en CPU) ==" -ForegroundColor Cyan
& $py -m rag_laboral.indexar

Write-Host "`n== 6/7 Recuperacion: 4 metodos x 3 variantes de sinonimos (re-ranker bge-reranker-v2-m3) ==" -ForegroundColor Cyan
& $py -m rag_laboral.evaluar_recuperacion

Write-Host "`n== 7/7 Generacion con qwen2.5:7b (~1 min por pregunta en CPU) y figuras ==" -ForegroundColor Cyan
if ($Limite -gt 0) { & $py -m rag_laboral.evaluar_generacion --correr --limite $Limite }
else { & $py -m rag_laboral.evaluar_generacion --correr }
& $py -m rag_laboral.graficos
Write-Host "`nListo: reports\recuperacion.json, reports\generacion.json y reports\figures\" -ForegroundColor Green
