# ============================================
# Tor SOCKS5 Proxy Helper
# ============================================
#
# Usage:
#   .\start_tor.ps1           # Start Tor proxy
#   .\start_tor.ps1 -Stop     # Stop Tor proxy
#   .\start_tor.ps1 -Status   # Check status
#   .\start_tor.ps1 -Test     # Test proxy
#   .\start_tor.ps1 -NewCircuit  # Get new IP
#
# ============================================

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Test,
    [switch]$NewCircuit
)

$TOR_EXE = "C:\Users\daves\Desktop\Tor Browser\Browser\TorBrowser\Tor\tor.exe"
$TOR_DATA_DIR = "$env:TEMP\tor_data"
$TOR_PID_FILE = "$env:TEMP\tor_proxy.pid"
$SOCKS_PORT = 9050
$CONTROL_PORT = 9051

function Test-TorInstalled {
    if (Test-Path $TOR_EXE) {
        return $true
    }
    Write-Host "ERROR: Tor not found at $TOR_EXE" -ForegroundColor Red
    return $false
}

function Get-TorStatus {
    if (Test-Path $TOR_PID_FILE) {
        $pid = Get-Content $TOR_PID_FILE
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Tor is RUNNING (PID: $pid)" -ForegroundColor Green
            Write-Host "SOCKS5 proxy: socks5://127.0.0.1:$SOCKS_PORT"
            return $true
        }
    }

    # Also check if any tor process is running
    $torProc = Get-Process -Name "tor" -ErrorAction SilentlyContinue
    if ($torProc) {
        Write-Host "Tor is RUNNING (PID: $($torProc.Id))" -ForegroundColor Green
        Write-Host "SOCKS5 proxy: socks5://127.0.0.1:$SOCKS_PORT"
        return $true
    }

    Write-Host "Tor is NOT RUNNING" -ForegroundColor Yellow
    return $false
}

function Start-Tor {
    Write-Host "Starting Tor..." -ForegroundColor Cyan

    # Create data directory
    if (-not (Test-Path $TOR_DATA_DIR)) {
        New-Item -ItemType Directory -Path $TOR_DATA_DIR -Force | Out-Null
    }

    # Start Tor in background
    $torArgs = @(
        "--SocksPort", $SOCKS_PORT,
        "--ControlPort", $CONTROL_PORT,
        "--DataDirectory", $TOR_DATA_DIR,
        "--Log", "notice file $TOR_DATA_DIR\tor.log"
    )

    $process = Start-Process -FilePath $TOR_EXE -ArgumentList $torArgs -WindowStyle Hidden -PassThru
    $process.Id | Out-File $TOR_PID_FILE

    Write-Host "Waiting for Tor to bootstrap..." -ForegroundColor Yellow

    # Wait for Tor to be ready (check log for bootstrap complete)
    $attempts = 0
    while ($attempts -lt 60) {
        Start-Sleep -Seconds 1
        $attempts++

        if (Test-Path "$TOR_DATA_DIR\tor.log") {
            $log = Get-Content "$TOR_DATA_DIR\tor.log" -Tail 5
            if ($log -match "Bootstrapped 100%") {
                Write-Host "Tor is ready!" -ForegroundColor Green
                Write-Host "SOCKS5 proxy: socks5://127.0.0.1:$SOCKS_PORT" -ForegroundColor Cyan
                return $true
            }
            # Show progress
            $bootstrap = $log | Select-String "Bootstrapped (\d+)%" | Select-Object -Last 1
            if ($bootstrap) {
                Write-Host "`rBootstrapping... $($bootstrap.Matches.Groups[1].Value)%" -NoNewline
            }
        }
    }

    Write-Host "`nTor bootstrap timeout" -ForegroundColor Red
    return $false
}

function Stop-Tor {
    Write-Host "Stopping Tor..." -ForegroundColor Cyan

    # Kill by PID file
    if (Test-Path $TOR_PID_FILE) {
        $pid = Get-Content $TOR_PID_FILE
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Remove-Item $TOR_PID_FILE -Force
    }

    # Also kill any tor.exe processes
    Get-Process -Name "tor" -ErrorAction SilentlyContinue | Stop-Process -Force

    Write-Host "Tor stopped" -ForegroundColor Green
}

function Test-TorProxy {
    Write-Host "`nTesting Tor proxy..." -ForegroundColor Cyan

    # Test with Python
    $testScript = @"
import httpx
try:
    # Get IP through Tor
    resp = httpx.get('https://check.torproject.org/api/ip', proxy='socks5://127.0.0.1:9050', timeout=30)
    data = resp.json()
    print(f"Tor IP: {data.get('IP', 'unknown')}")
    print(f"Is Tor: {data.get('IsTor', False)}")
except Exception as e:
    print(f"Error: {e}")
"@

    python -c $testScript
}

function Get-NewCircuit {
    Write-Host "Requesting new Tor circuit (new IP)..." -ForegroundColor Cyan

    # Send NEWNYM signal to control port
    # This requires control port authentication to be disabled or use password

    try {
        $client = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $CONTROL_PORT)
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $reader = New-Object System.IO.StreamReader($stream)

        $writer.WriteLine("AUTHENTICATE")
        $writer.Flush()
        $response = $reader.ReadLine()

        $writer.WriteLine("SIGNAL NEWNYM")
        $writer.Flush()
        $response = $reader.ReadLine()

        $client.Close()

        if ($response -match "250 OK") {
            Write-Host "New circuit requested!" -ForegroundColor Green
            Write-Host "Wait 10 seconds for new IP..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
            Test-TorProxy
        } else {
            Write-Host "Failed: $response" -ForegroundColor Red
        }
    } catch {
        Write-Host "Could not connect to Tor control port: $_" -ForegroundColor Red
        Write-Host "Restarting Tor for new circuit..." -ForegroundColor Yellow
        Stop-Tor
        Start-Sleep -Seconds 2
        Start-Tor
    }
}

# Main
if (-not (Test-TorInstalled)) {
    exit 1
}

if ($Stop) {
    Stop-Tor
} elseif ($Status) {
    Get-TorStatus
} elseif ($Test) {
    Test-TorProxy
} elseif ($NewCircuit) {
    Get-NewCircuit
} else {
    # Start if not running
    if (-not (Get-TorStatus)) {
        Start-Tor
    }
}
