# Cleanup stuck Ralph processes
# Kill high-CPU claude processes (Ralph subprocesses)
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1
Write-Host "Killing stuck claude processes..." -ForegroundColor Yellow

$claudeTargets = @(16164, 14180, 13832, 13916, 49084)
foreach ($pid in $claudeTargets) {
    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
        Write-Host "  Killed claude PID $pid" -ForegroundColor Green
    } catch {
        Write-Host "  Skip PID ${pid}: already gone or access denied" -ForegroundColor DarkGray
    }
}

# Kill orphaned long-running node processes (>500 CPU seconds)
Write-Host ""
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

# Kill remaining low-CPU claude processes (except likely current session)
Write-Host ""
Write-Host "Killing remaining Ralph claude subprocesses..." -ForegroundColor Yellow

$remainingClaude = Get-Process -Name claude -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 5 }
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
