param([switch]$SkipData)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan virtual environment gagal.' }
}
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv pip install --python $taskPython -r requirements.txt --cache-dir '.cache\uv' --link-mode copy
} else {
    & $taskPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Bootstrap pip gagal.' }
    & $taskPython -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal; periksa akses internet dan kompatibilitas wheel.' }
& $taskPython -m fashion_mlops doctor
if ($LASTEXITCODE -ne 0) { throw 'Pemeriksaan dependensi gagal.' }
if (-not $SkipData) {
    & $taskPython -m fashion_mlops prepare-data
    if ($LASTEXITCODE -ne 0) { throw 'Persiapan dataset gagal.' }
}
Write-Output 'Setup selesai. Ikuti langkah commit Git pada README sebelum training.'
