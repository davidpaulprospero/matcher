# scripts/ralph/lib/queue.ps1
# Interview queue: focus area processing, progress tracking, completion flow

function Get-QueueData {
    <#
    .SYNOPSIS
        Read and parse queue.json. Returns $null if file doesn't exist or is invalid.
    #>
    if (-not (Test-Path $script:QueueFile)) {
        Write-Host "  [debug:GetQueueData] File not found: $($script:QueueFile)" -ForegroundColor Red
        return $null
    }
    try {
        $data = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
        return $data
    }
    catch {
        Write-Host "  [debug:GetQueueData] Parse error: $_" -ForegroundColor Red
        return $null
    }
}

function Get-InterviewFocusAreas {
    <#
    .SYNOPSIS
        Read incomplete focus areas from queue.json
    .RETURNS
        Array of incomplete focus area objects, or empty array if none
    #>
    $queue = Get-QueueData
    if (-not $queue) {
        Write-Host "  [debug:GetFocusAreas] Queue data is null" -ForegroundColor Red
        return @()
    }

    # Interview format: focusAreas array with completed flag
    if ($queue.focusAreas) {
        $result = @($queue.focusAreas | Where-Object { -not $_.completed })
        Write-Host "  [debug:GetFocusAreas] Found $($queue.focusAreas.Count) areas, $($result.Count) incomplete" -ForegroundColor DarkGray
        return $result
    }

    # Legacy format: queue array with completedAreas
    if ($queue.queue -and $queue.completedAreas) {
        return @($queue.queue | Where-Object { $queue.completedAreas -notcontains $_ } | ForEach-Object { @{ id = $_; completed = $false } })
    }

    Write-Host "  [debug:GetFocusAreas] No focusAreas or queue format found" -ForegroundColor Red
    return @()
}

function Get-InterviewContext {
    <#
    .SYNOPSIS
        Get the interview context string from queue.json
    #>
    $queue = Get-QueueData
    if ($queue -and $queue.interviewContext) { return $queue.interviewContext }
    return ""
}

function Update-ContextFromPRD {
    <#
    .SYNOPSIS
        Update queue.json's interviewContext from the current PRD's projectContext.
        Called after PRD generation so the dashboard and all context readers
        show the new focus area's context instead of the stale interview context.
    #>
    $prdPath = Join-Path $script:RalphDir "prd.json"
    if (-not (Test-Path $prdPath)) { return }

    try {
        $prd = Get-Content $prdPath -Raw | ConvertFrom-Json
        # Only update if projectContext is a string (not the seed PRD's dict)
        if ($prd.projectContext -and $prd.projectContext -is [string]) {
            $queue = Get-QueueData
            if ($queue) {
                $queue.interviewContext = $prd.projectContext
                Write-JsonNoBom -Path $script:QueueFile -Content ($queue | ConvertTo-Json -Depth 10)
                Write-Host "  Context updated for focus area: $($prd.focusArea)" -ForegroundColor DarkGray
            }
        }
    } catch {
        Write-Host "  [warn] Failed to update context from PRD: $_" -ForegroundColor Yellow
    }
}

function Get-NextQueuedFocusArea {
    <#
    .SYNOPSIS
        Get the next incomplete focus area ID
    .RETURNS
        Focus area ID string, or $null if none remaining
    #>
    $areas = Get-InterviewFocusAreas
    if ($areas.Count -gt 0) {
        $first = $areas | Select-Object -First 1
        $result = if ($first.id) { $first.id } else { $first }
        return $result
    }
    return $null
}

function Update-QueueProgress {
    <#
    .SYNOPSIS
        Mark a focus area as completed in queue.json (handles both formats)
    .PARAMETER AreaId
        The focus area ID to mark as completed
    .PARAMETER Silent
        If set, don't print success message
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$AreaId,
        [switch]$Silent
    )

    $queue = Get-QueueData
    if (-not $queue) { return }

    try {
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

        # Update interview format (focusAreas array)
        if ($queue.focusAreas) {
            foreach ($area in $queue.focusAreas) {
                if ($area.id -eq $AreaId) {
                    $area.completed = $true
                    $area.completedAt = $timestamp
                }
            }
        }

        # Update legacy format (completedAreas array)
        if ($null -ne $queue.completedAreas -and $queue.completedAreas -notcontains $AreaId) {
            $queue.completedAreas += $AreaId
        }

        # Update currentIndex for legacy format
        if ($queue.queue) {
            $idx = [array]::IndexOf($queue.queue, $AreaId)
            if ($idx -ge 0) { $queue.currentIndex = $idx + 1 }
        }

        # Update session info
        if ($queue.session) {
            $queue.session.lastActivityAt = $timestamp
            $queue.session.iterationCount = $script:IterationCount
        }

        Write-JsonNoBom -Path $script:QueueFile -Content ($queue | ConvertTo-Json -Depth 10)

        if (-not $Silent) {
            Write-Host "  Marked '$AreaId' as completed" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  Warning: Could not update queue.json" -ForegroundColor Yellow
    }
}

# Aliases for backward compatibility
function Update-InterviewProgress { param([string]$AreaId) Update-QueueProgress -AreaId $AreaId }
function Update-LegacyQueueProgress { param([string]$CompletedArea) Update-QueueProgress -AreaId $CompletedArea -Silent }

function Show-CompletionChoice {
    <#
    .SYNOPSIS
        After interview queue complete, prompt for next action
    .PARAMETER CompletedAreas
        Array of completed focus area IDs
    .RETURNS
        "interview", "trueauto", or "stop"
    #>
    param(
        [array]$CompletedAreas
    )

    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "   'I bent my Wookie!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Interview Queue Complete!" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""

    # Show completed areas
    Write-Host "  Completed focus areas:" -ForegroundColor Cyan
    foreach ($area in $CompletedAreas) {
        $areaId = if ($area.id) { $area.id } else { $area }
        Write-Host "    [x] $areaId" -ForegroundColor Green
    }

    # Show session stats
    $duration = (Get-Date) - $script:SessionStartTime
    Write-Host ""
    Write-Host "  Session Stats:" -ForegroundColor DarkGray
    Write-Host "    Iterations: $script:IterationCount" -ForegroundColor DarkGray
    Write-Host "    Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor DarkGray

    Write-Host ""
    Write-Host "  What's next?" -ForegroundColor White
    Write-Host ""
    Write-Host "  [I] New interview - Start fresh with new focus areas" -ForegroundColor Cyan
    Write-Host "  [T] TrueAuto mode - Continue with general improvements" -ForegroundColor Magenta
    Write-Host "  [S] Stop - Exit Ralph loop" -ForegroundColor Yellow
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Ii]" { return "interview" }
        "^[Tt]" { return "trueauto" }
        default { return "stop" }
    }
}
