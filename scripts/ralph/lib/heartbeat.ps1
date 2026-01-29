# scripts/ralph/lib/heartbeat.ps1
# Persistent heartbeat logging for diagnosing silent hangs

$script:HeartbeatFile = Join-Path $script:RalphDir "heartbeat.json"
$script:SessionLogFile = Join-Path $script:RalphDir "session.log"

function Write-Heartbeat {
    <#
    .SYNOPSIS
        Write current status to heartbeat file for crash diagnosis
    .PARAMETER Phase
        Current phase: "starting", "waiting_for_claude", "processing_result", "between_iterations"
    .PARAMETER Details
        Hashtable with additional context
    #>
    param(
        [Parameter(Mandatory)][string]$Phase,
        [hashtable]$Details = @{}
    )

    $heartbeat = @{
        timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        timestampUtc = (Get-Date).ToUniversalTime().ToString("o")
        phase = $Phase
        iteration = $script:State.IterationCount
        storyId = $script:State.CurrentStoryId
        focusArea = $script:State.CurrentFocusArea
        sprintNumber = $script:State.CurrentSprintNumber
        mode = $script:State.CurrentMode
        pid = $PID
        details = $Details
    }

    try {
        $heartbeat | ConvertTo-Json -Depth 4 | Set-Content $script:HeartbeatFile -Encoding UTF8 -ErrorAction Stop
    }
    catch {
        # Silent fail - don't break the loop for logging issues
    }
}

function Write-SessionLog {
    <#
    .SYNOPSIS
        Append entry to persistent session log
    .PARAMETER Event
        Event type: "loop_start", "iteration_start", "claude_call", "claude_return", "iteration_end", "loop_end", "error"
    .PARAMETER Message
        Human-readable message
    .PARAMETER Data
        Optional hashtable with structured data
    #>
    param(
        [Parameter(Mandatory)][string]$Event,
        [Parameter(Mandatory)][string]$Message,
        [hashtable]$Data = @{}
    )

    $entry = @{
        t = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss.fff")
        e = $Event
        m = $Message
        i = $script:State.IterationCount
        s = $script:State.CurrentStoryId
    }

    if ($Data.Count -gt 0) {
        $entry.d = $Data
    }

    $line = $entry | ConvertTo-Json -Compress

    try {
        Add-Content -Path $script:SessionLogFile -Value $line -Encoding UTF8 -ErrorAction Stop
    }
    catch {
        # Retry once after brief delay
        Start-Sleep -Milliseconds 100
        try {
            Add-Content -Path $script:SessionLogFile -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
        }
        catch {}
    }
}

function Get-LastHeartbeat {
    <#
    .SYNOPSIS
        Read the last heartbeat to diagnose what happened
    .RETURNS
        Hashtable with last heartbeat data, or $null if not found
    #>

    if (-not (Test-Path $script:HeartbeatFile)) {
        return $null
    }

    try {
        $content = Get-Content $script:HeartbeatFile -Raw -ErrorAction Stop
        return $content | ConvertFrom-Json -AsHashtable
    }
    catch {
        return $null
    }
}

function Show-LastSessionDiagnosis {
    <#
    .SYNOPSIS
        On startup, show what happened in the last session if it didn't exit cleanly
    #>

    $lastHb = Get-LastHeartbeat
    if (-not $lastHb) {
        return
    }

    $lastTime = [DateTime]::Parse($lastHb.timestamp)
    $age = (Get-Date) - $lastTime

    # If heartbeat is recent (within 10 minutes), session might still be running
    if ($age.TotalMinutes -lt 10) {
        # Check if the PID is still running
        $oldPid = $lastHb.pid
        $stillRunning = $false
        if ($oldPid) {
            try {
                $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
                if ($proc -and $proc.ProcessName -eq "pwsh") {
                    $stillRunning = $true
                }
            }
            catch {}
        }

        if ($stillRunning) {
            Write-Host "  Note: Another Ralph session may be running (PID $oldPid)" -ForegroundColor Yellow
            return
        }
    }

    # Session ended - show diagnosis
    Write-Host ""
    Write-Host "  Last session diagnosis:" -ForegroundColor Cyan
    Write-Host "    Ended: $($lastHb.timestamp) ($([math]::Round($age.TotalHours, 1)) hours ago)" -ForegroundColor DarkGray
    Write-Host "    Phase: $($lastHb.phase)" -ForegroundColor $(if ($lastHb.phase -eq "waiting_for_claude") { "Yellow" } else { "DarkGray" })
    Write-Host "    Story: $($lastHb.storyId)" -ForegroundColor DarkGray
    Write-Host "    Iteration: $($lastHb.iteration)" -ForegroundColor DarkGray

    if ($lastHb.phase -eq "waiting_for_claude") {
        Write-Host ""
        Write-Host "    Likely cause: Claude API hung or connection dropped" -ForegroundColor Yellow
        if ($lastHb.details.promptFile) {
            Write-Host "    Prompt was: $($lastHb.details.promptFile)" -ForegroundColor DarkGray
        }
        if ($lastHb.details.waitingSeconds) {
            Write-Host "    Was waiting for: $($lastHb.details.waitingSeconds)s" -ForegroundColor DarkGray
        }
    }
    elseif ($lastHb.phase -eq "between_iterations") {
        Write-Host ""
        Write-Host "    Session stopped between iterations (likely manual close)" -ForegroundColor DarkGray
    }

    # Show last few session log entries
    if (Test-Path $script:SessionLogFile) {
        Write-Host ""
        Write-Host "    Last 5 log entries:" -ForegroundColor Cyan
        $lastLines = Get-Content $script:SessionLogFile -Tail 5 -ErrorAction SilentlyContinue
        foreach ($line in $lastLines) {
            try {
                $entry = $line | ConvertFrom-Json
                $color = switch ($entry.e) {
                    "error" { "Red" }
                    "claude_call" { "Yellow" }
                    "claude_return" { "Green" }
                    default { "DarkGray" }
                }
                Write-Host "      $($entry.t) [$($entry.e)] $($entry.m)" -ForegroundColor $color
            }
            catch {
                Write-Host "      $line" -ForegroundColor DarkGray
            }
        }
    }

    Write-Host ""
}

function Clear-SessionLog {
    <#
    .SYNOPSIS
        Rotate/clear session log on fresh start (keeps last 1000 lines)
    #>

    if (-not (Test-Path $script:SessionLogFile)) {
        return
    }

    try {
        $lines = Get-Content $script:SessionLogFile -ErrorAction Stop
        if ($lines.Count -gt 1000) {
            $lines | Select-Object -Last 1000 | Set-Content $script:SessionLogFile -Encoding UTF8
            Write-Host "  Session log rotated (kept last 1000 entries)" -ForegroundColor DarkGray
        }
    }
    catch {}
}

# Functions are available via dot-sourcing
