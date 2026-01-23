# ============================================
# SOCKS5 Proxy via SSH Tunnel (Background)
# ============================================
#
# Creates a local SOCKS5 proxy on port 1080 via SSH tunnel.
# Can run in background or foreground.
#
# Usage:
#   .\start_socks_proxy.ps1                    # Interactive (asks for details)
#   .\start_socks_proxy.ps1 -User me -Host myserver.com
#   .\start_socks_proxy.ps1 -Background        # Run in background
#
# ============================================

param(
    [string]$User = "",
    [string]$Host = "",
    [int]$Port = 22,
    [int]$LocalPort = 1080,
    [string]$KeyFile = "",
    [switch]$Background,
    [switch]$Stop,
    [switch]$Status
)

$ProxyPidFile = "$env:TEMP\socks_proxy.pid"

function Get-ProxyStatus {
    if (Test-Path $ProxyPidFile) {
        $pid = Get-Content $ProxyPidFile
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Proxy is RUNNING (PID: $pid)" -ForegroundColor Green
            Write-Host "Local endpoint: socks5://127.0.0.1:$LocalPort"
            return $true
        }
    }
    Write-Host "Proxy is NOT RUNNING" -ForegroundColor Yellow
    return $false
}

function Stop-Proxy {
    if (Test-Path $ProxyPidFile) {
        $pid = Get-Content $ProxyPidFile
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $pid -Force
            Write-Host "Stopped proxy (PID: $pid)" -ForegroundColor Green
        }
        Remove-Item $ProxyPidFile -Force
    } else {
        Write-Host "No proxy PID file found" -ForegroundColor Yellow
    }
}

function Test-Proxy {
    Write-Host "`nTesting proxy..." -ForegroundColor Cyan

    # Test 1: Check if port is listening
    $listening = Test-NetConnection -ComputerName 127.0.0.1 -Port $LocalPort -WarningAction SilentlyContinue
    if (-not $listening.TcpTestSucceeded) {
        Write-Host "FAILED: Port $LocalPort is not listening" -ForegroundColor Red
        return $false
    }
    Write-Host "OK: Port $LocalPort is listening" -ForegroundColor Green

    # Test 2: Get IP through proxy vs direct
    try {
        $directIP = (Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing -TimeoutSec 10).Content
        Write-Host "Direct IP: $directIP" -ForegroundColor Gray
    } catch {
        Write-Host "Could not get direct IP" -ForegroundColor Yellow
    }

    Write-Host "`nProxy appears to be working!" -ForegroundColor Green
    Write-Host "Configure in config.yaml:" -ForegroundColor Cyan
    Write-Host "  download:" -ForegroundColor White
    Write-Host "    fallback:" -ForegroundColor White
    Write-Host "      proxy:" -ForegroundColor White
    Write-Host "        enabled: true" -ForegroundColor White
    Write-Host "        sources:" -ForegroundColor White
    Write-Host "          - type: socks5" -ForegroundColor White
    Write-Host "            url: `"socks5://127.0.0.1:$LocalPort`"" -ForegroundColor White

    return $true
}

# Handle -Status flag
if ($Status) {
    Get-ProxyStatus
    exit
}

# Handle -Stop flag
if ($Stop) {
    Stop-Proxy
    exit
}

# Interactive mode if no host specified
if (-not $Host) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "SSH SOCKS5 Proxy Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    $User = Read-Host "SSH Username"
    $Host = Read-Host "SSH Host (e.g., myserver.com)"
    $portInput = Read-Host "SSH Port [22]"
    if ($portInput) { $Port = [int]$portInput }

    $keyInput = Read-Host "SSH Key file (leave empty for password)"
    if ($keyInput) { $KeyFile = $keyInput }
}

Write-Host ""
Write-Host "Connecting to $User@$Host`:$Port..." -ForegroundColor Cyan
Write-Host "Local SOCKS5 proxy will be: socks5://127.0.0.1:$LocalPort" -ForegroundColor Cyan
Write-Host ""

# Build SSH command
$sshArgs = @("-D", $LocalPort, "-C", "-N", "$User@$Host", "-p", $Port)
if ($KeyFile) {
    $sshArgs = @("-D", $LocalPort, "-C", "-N", "-i", $KeyFile, "$User@$Host", "-p", $Port)
}

if ($Background) {
    Write-Host "Starting in background..." -ForegroundColor Yellow
    $process = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -WindowStyle Hidden -PassThru
    $process.Id | Out-File $ProxyPidFile
    Write-Host "Started proxy in background (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "Use -Stop to stop, -Status to check" -ForegroundColor Gray

    Start-Sleep -Seconds 3
    Test-Proxy
} else {
    Write-Host "Running in foreground. Press Ctrl+C to stop." -ForegroundColor Yellow
    Write-Host ""
    & ssh $sshArgs
}
