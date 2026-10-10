$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\mlflow.exe' server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
if ($LASTEXITCODE -ne 0) { throw 'MLflow server berhenti dengan error.' }
