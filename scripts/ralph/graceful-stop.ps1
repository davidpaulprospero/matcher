# Graceful Stop Utility for Ralph Loop
# Usage: .\graceful-stop.ps1           - Request graceful stop
#        .\graceful-stop.ps1 -Cancel   - Cancel graceful stop request
#        .\graceful-stop.ps1 -Status   - Check current status

param(
    [switch]$Cancel,
    [switch]$Status,
    [string]$Reason = "User requested"
)

$SignalFile = Join-Path $PSScriptRoot "graceful_stop.signal"

if ($Status) {
    if (Test-Path $SignalFile) {
        Write-Host ""
        Write-Host "  Graceful stop is PENDING" -ForegroundColor Yellow
        Write-Host ""
        try {
            $meta = Get-Content $SignalFile -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($meta) {
                Write-Host "  Reason: $($meta.reason)" -ForegroundColor DarkGray
                Write-Host "  Requested at: $($meta.requestedAt)" -ForegroundColor DarkGray
            }
        }
        catch {
            Write-Host "  (No metadata available)" -ForegroundColor DarkGray
        }
        Write-Host ""
        Write-Host "  Ralph will stop after the current sprint completes." -ForegroundColor White
        Write-Host "  Run with -Cancel to remove the stop request." -ForegroundColor DarkGray
    }
    else {
        Write-Host ""
        Write-Host "  No graceful stop pending" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Ralph will continue working normally." -ForegroundColor DarkGray
    }
}
elseif ($Cancel) {
    if (Test-Path $SignalFile) {
        Remove-Item $SignalFile -Force -ErrorAction SilentlyContinue
        Write-Host ""
        Write-Host "  Graceful stop cancelled" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Ralph will continue working normally." -ForegroundColor DarkGray
    }
    else {
        Write-Host ""
        Write-Host "  No graceful stop was pending" -ForegroundColor Yellow
    }
}
else {
    @{
        requestedAt = (Get-Date).ToString("o")
        reason = $Reason
    } | ConvertTo-Json | Set-Content $SignalFile -Encoding UTF8

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "     Graceful Stop Requested" -ForegroundColor Cyan
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Ralph will stop after the current sprint completes." -ForegroundColor White
    Write-Host ""
    Write-Host "  This means:" -ForegroundColor DarkGray
    Write-Host "    - Current story will finish" -ForegroundColor DarkGray
    Write-Host "    - Sprint will be archived if complete" -ForegroundColor DarkGray
    Write-Host "    - No new sprint will be started" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  To cancel: .\graceful-stop.ps1 -Cancel" -ForegroundColor DarkGray
    Write-Host "  To check:  .\graceful-stop.ps1 -Status" -ForegroundColor DarkGray
    Write-Host ""
}
