<#
.SYNOPSIS
    Starts all News-Detector-Learn services locally on Windows:
      1. Python ML service (FastAPI, port 8001)
      2. Node.js API server (Express, port 8080)
      3. React frontend (Vite dev server)

.NOTES
    Each service launches in its own PowerShell window so logs stay visible
    and the processes keep running independently of this script.
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "==> Starting News-Detector-Learn services..." -ForegroundColor Cyan

# 1. Python ML service (port 8001)
Write-Host "==> Launching ML service (FastAPI) on port 8001..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root\services\ml-api'; `$env:ML_PORT='8001'; python main.py"
)

# 2. Node.js API server (port 8080) - builds then starts, needs DATABASE_URL etc. from artifacts\api-server\.env
Write-Host "==> Launching API server on port 8080..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root'; `$env:NODE_ENV='development'; pnpm --filter @workspace/api-server run dev"
)

# 3. React frontend (Vite dev server)
Write-Host "==> Launching frontend (Vite) ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$root'; pnpm --filter @workspace/news-detector run dev"
)

Write-Host "==> All services launching in separate windows." -ForegroundColor Green
Write-Host "    ML service:   http://localhost:8001"
Write-Host "    API server:   http://localhost:8080"
Write-Host "    Frontend:     check the Vite window for the local URL (usually http://localhost:5173)"
