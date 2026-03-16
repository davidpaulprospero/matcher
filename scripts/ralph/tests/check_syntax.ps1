$scriptPath = Join-Path $PSScriptRoot "..\ralph.ps1"
try {
    $null = [scriptblock]::Create((Get-Content $scriptPath -Raw))
    Write-Host "Syntax OK" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "Syntax Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
