param([switch]$StartServer)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
if ($StartServer) {
    New-Item -ItemType Directory -Path outputs -Force | Out-Null
    $taskMlflow = Join-Path (Get-Location) '.venv\Scripts\mlflow.exe'
    $taskServer = Start-Process -FilePath $taskMlflow -ArgumentList @('server', '--backend-store-uri', 'sqlite:///mlflow.db', '--artifacts-destination', './mlartifacts', '--host', '127.0.0.1', '--port', '5000') -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru -RedirectStandardOutput 'outputs\mlflow-server.stdout.log' -RedirectStandardError 'outputs\mlflow-server.stderr.log'
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 60; $taskAttempt++) {
        try {
            $taskResponse = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/health' -UseBasicParsing -TimeoutSec 2
            if ($taskResponse.StatusCode -eq 200) { $taskReady = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
        if ($taskServer.HasExited) { throw 'MLflow server gagal start. Periksa outputs\mlflow-server.stderr.log.' }
    }
    if (-not $taskReady) { throw 'MLflow belum siap. Periksa log server.' }
    Write-Output "MLflow PID: $($taskServer.Id); UI: http://127.0.0.1:5000"
}
& $taskPython -m fashion_mlops all
if ($LASTEXITCODE -ne 0) { throw 'Pipeline berhenti; periksa pesan error. Tahap yang sudah selesai tetap tersimpan.' }
Write-Output 'Pipeline selesai. Laporan: reports\laporan.md. Ambil screenshot compare view dari MLflow UI.'
