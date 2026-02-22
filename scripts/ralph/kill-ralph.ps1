#!/usr/bin/env pwsh
# Kill Ralph processes

$pythonProcs = Get-Process -Name python3 -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "ralph" }
$pwshProcs = Get-Process -Name pwsh -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "ralph" }

if ($pythonProcs) {
    Write-Host "[OK] Stopping Python Ralph processes..."
    $pythonProcs | Stop-Process -Force
}

if ($pwshProcs) {
    Write-Host "[OK] Stopping PowerShell Ralph processes..."
    $pwshProcs | Stop-Process -Force
}

Get-Job | Stop-Job -ErrorAction SilentlyContinue
Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue

Write-Host "[OK] Ralph stopped"
