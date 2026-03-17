# Cleanup stuck Ralph processes
# Kill high-CPU claude/node processes (Ralph subprocesses)
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1

Write-Host "Killing orphaned node processes (CPU > 500s)..." -ForegroundColor Yellow

$nodeProcs = Get-Process -Name node -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 500 }
foreach ($proc in $nodeProcs) {
    try {
        $cpuRounded = [math]::Round($proc.CPU)
        Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        Write-Host "  Killed node PID $($proc.Id) (CPU: ${cpuRounded}s)" -ForegroundColor Green
    } catch {
        Write-Host "  Skip node PID $($proc.Id)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Killing remaining Ralph claude subprocesses (CPU > 60s)..." -ForegroundColor Yellow

$remainingClaude = Get-Process -Name claude -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 60 }
foreach ($proc in $remainingClaude) {
    try {
        $cpuRounded = [math]::Round($proc.CPU)
        Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        Write-Host "  Killed claude PID $($proc.Id) (CPU: ${cpuRounded}s)" -ForegroundColor Green
    } catch {
        Write-Host "  Skip claude PID $($proc.Id)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Remaining processes:" -ForegroundColor Cyan
Get-Process -Name claude -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, CPU -AutoSize
Get-Process -Name node -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 10 } | Format-Table Id, ProcessName, CPU -AutoSize
Write-Host "Done." -ForegroundColor Green
