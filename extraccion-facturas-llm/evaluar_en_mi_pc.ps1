# Ejecuta TODO en tu PC (con Ollama abierto): entorno, datos, pruebas y evaluación.
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1
#   powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -NPorPlantilla 4    (rápido, ~25 min)
param([int]$NPorPlantilla = 10)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "`n== 1/5 Entorno virtual y librerias ==" -ForegroundColor Cyan
if (-not (Test-Path $py)) { py -m venv .venv }
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet -r requirements.txt
& $py -m pip install --quiet -e .

Write-Host "`n== 2/5 200 facturas sinteticas (~10 s) ==" -ForegroundColor Cyan
if (-not (Test-Path "data\raw\facturas\FAC_200.pdf")) { & $py generar_datos.py }

Write-Host "`n== 3/5 Pruebas (sin Ollama) ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "`n== 4/5 Extraccion con el LLM (reanudable; ~2 min por factura en CPU) ==" -ForegroundColor Cyan
try { Invoke-RestMethod http://localhost:11434/api/tags | Out-Null }
catch { Write-Host "Abre Ollama primero (ollama serve) y vuelve a correr este script." -ForegroundColor Red; exit 1 }
$env:FACTURAS_GRABAR = "1"
& $py -m facturas.correr_llm --n-por-plantilla $NPorPlantilla

Write-Host "`n== 5/5 Metricas y figuras ==" -ForegroundColor Cyan
& $py -m facturas.evaluar
Write-Host "`nListo: reports\evaluacion.json y reports\figures\" -ForegroundColor Green
