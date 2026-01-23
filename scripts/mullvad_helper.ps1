# ============================================
# Mullvad VPN Helper for YouTube Rate Limiting
# ============================================
#
# Usage:
#   .\mullvad_helper.ps1 connect     # Connect to VPN
#   .\mullvad_helper.ps1 rotate      # Get new IP (reconnect)
#   .\mullvad_helper.ps1 status      # Show connection status
#   .\mullvad_helper.ps1 test        # Test proxy and YouTube access
#   .\mullvad_helper.ps1 disconnect  # Disconnect VPN
#
# ============================================

param(
    [Parameter(Position=0)]
    [ValidateSet("connect", "disconnect", "rotate", "status", "test", "setup")]
    [string]$Action = "status"
)

$SOCKS5_PROXY = "socks5://10.64.0.1:1080"
$MULLVAD_CLI = "C:\Program Files\Mullvad VPN\resources\mullvad.exe"

function Test-MullvadInstalled {
    if (Test-Path $MULLVAD_CLI) {
        return $true
    }
    # Fallback: check PATH
    $mullvad = Get-Command mullvad -ErrorAction SilentlyContinue
    if ($mullvad) {
        $script:MULLVAD_CLI = "mullvad"
        return $true
    }
    Write-Host "ERROR: Mullvad CLI not found" -ForegroundColor Red
    Write-Host "Install from: https://mullvad.net/en/download/windows" -ForegroundColor Yellow
    return $false
}

function Invoke-Mullvad {
    param([string[]]$Arguments)
    & $MULLVAD_CLI @Arguments
}

function Show-Status {
    Write-Host "`n=== Mullvad Status ===" -ForegroundColor Cyan
    Invoke-Mullvad status

    Write-Host "`n=== Account ===" -ForegroundColor Cyan
    Invoke-Mullvad account, get 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Not logged in. Run: mullvad account login <account-number>" -ForegroundColor Yellow
    }

    Write-Host "`n=== SOCKS5 Proxy ===" -ForegroundColor Cyan
    Write-Host "Endpoint: $SOCKS5_PROXY" -ForegroundColor White
}

function Connect-Mullvad {
    Write-Host "Connecting to Mullvad..." -ForegroundColor Cyan
    Invoke-Mullvad connect

    # Wait for connection
    $attempts = 0
    while ($attempts -lt 10) {
        Start-Sleep -Seconds 1
        $status = Invoke-Mullvad status
        if ($status -match "Connected") {
            Write-Host "`nConnected!" -ForegroundColor Green
            Invoke-Mullvad status
            Write-Host "`nSOCKS5 Proxy available at: $SOCKS5_PROXY" -ForegroundColor Green
            return $true
        }
        $attempts++
    }

    Write-Host "Connection timeout" -ForegroundColor Red
    return $false
}

function Disconnect-Mullvad {
    Write-Host "Disconnecting..." -ForegroundColor Cyan
    Invoke-Mullvad disconnect
    Write-Host "Disconnected" -ForegroundColor Green
}

function Rotate-Server {
    Write-Host "Rotating to new server (new IP)..." -ForegroundColor Cyan

    # Get current relay
    $currentRelay = Invoke-Mullvad relay, get
    Write-Host "Current: $currentRelay" -ForegroundColor Gray

    # Reconnect to get new server
    Invoke-Mullvad reconnect

    Start-Sleep -Seconds 3

    # Show new status
    $newRelay = Invoke-Mullvad relay, get
    Write-Host "New: $newRelay" -ForegroundColor Green

    Invoke-Mullvad status
}

function Test-Proxy {
    Write-Host "`n=== Testing Mullvad Proxy ===" -ForegroundColor Cyan

    # Check if connected
    $status = mullvad status
    if ($status -notmatch "Connected") {
        Write-Host "Not connected to Mullvad. Run: .\mullvad_helper.ps1 connect" -ForegroundColor Red
        return $false
    }

    # Run the Python test script
    $testScript = Join-Path $PSScriptRoot "test_proxy.py"
    if (Test-Path $testScript) {
        python $testScript $SOCKS5_PROXY
    } else {
        # Fallback: basic curl test
        Write-Host "`n1. Getting direct IP..." -ForegroundColor White
        try {
            $directIP = (Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing -TimeoutSec 10).Content
            Write-Host "   Direct IP: $directIP" -ForegroundColor Gray
        } catch {
            Write-Host "   Could not get direct IP" -ForegroundColor Yellow
        }

        Write-Host "`n2. Mullvad connection info:" -ForegroundColor White
        mullvad status

        Write-Host "`nProxy endpoint: $SOCKS5_PROXY" -ForegroundColor Green
        Write-Host "Config is set to use this proxy automatically." -ForegroundColor Green
    }
}

function Show-Setup {
    Write-Host @"

=== Mullvad Setup Guide ===

1. CREATE ACCOUNT (no email needed):
   https://mullvad.net/en/account/create

2. ADD TIME ($5/month):
   https://mullvad.net/en/account/

3. LOGIN:
   mullvad account login <your-account-number>

4. CONNECT:
   .\mullvad_helper.ps1 connect

5. TEST:
   .\mullvad_helper.ps1 test

6. RUN PIPELINE:
   python main.py --voiceover script.srt --project ./myproject

=== Commands ===
.\mullvad_helper.ps1 connect     - Connect to VPN
.\mullvad_helper.ps1 status      - Show status
.\mullvad_helper.ps1 test        - Test YouTube access
.\mullvad_helper.ps1 rotate      - Get new IP
.\mullvad_helper.ps1 disconnect  - Disconnect

"@ -ForegroundColor Cyan
}

# Main
if (-not (Test-MullvadInstalled)) {
    exit 1
}

switch ($Action) {
    "connect"    { Connect-Mullvad }
    "disconnect" { Disconnect-Mullvad }
    "rotate"     { Rotate-Server }
    "status"     { Show-Status }
    "test"       { Test-Proxy }
    "setup"      { Show-Setup }
}
