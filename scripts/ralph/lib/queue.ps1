# scripts/ralph/lib/queue.ps1
# Interview queue: focus area processing, progress tracking, completion flow

function Get-Queue {
    <#
    .SYNOPSIS
        Load queue data from JSON file
    #>
    param([string]$Path = $script:QueueFile)
    Read-JsonFile -Path $Path
}

function Save-Queue {
    <#
    .SYNOPSIS
        Save queue data atomically
    #>
    param(
        [Parameter(Mandatory)][object]$Queue,
        [string]$Path = $script:QueueFile
    )
    Save-StateFile -Path $Path -Data $Queue
}

function Get-QueueData {
    <#
    .SYNOPSIS
        Read and parse queue.json. Returns $null if file doesn't exist or is invalid.
        Wraps Get-Queue with debug output.
    #>
    $data = Get-Queue
    if (-not $data -and -not (Test-Path $script:QueueFile)) {
        Write-Host "  [debug:GetQueueData] File not found: $($script:QueueFile)" -ForegroundColor Red
    } elseif (-not $data) {
        Write-Host "  [debug:GetQueueData] Parse error" -ForegroundColor Red
    }
    return $data
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

    if (-not $queue.focusAreas) {
        Write-Host "  [debug:GetFocusAreas] No focusAreas found in queue" -ForegroundColor Red
        return @()
    }

    $result = @($queue.focusAreas | Where-Object { -not $_.completed })
    Write-Host "  [debug:GetFocusAreas] Found $($queue.focusAreas.Count) areas, $($result.Count) incomplete" -ForegroundColor DarkGray
    return $result
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
    $prd = Get-Sprint
    if (-not $prd) { return }

    try {
        # Only update if projectContext is a string (not the seed PRD's dict)
        if ($prd.projectContext -and $prd.projectContext -is [string]) {
            $queue = Get-QueueData
            if ($queue) {
                $queue.interviewContext = $prd.projectContext
                Save-Queue -Queue $queue
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
        Mark a focus area as completed in queue.json.
        If the area isn't tracked in the queue yet, adds it automatically.
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

        if (-not $queue.focusAreas) {
            $queue.focusAreas = @()
        }

        # Find existing entry or add new one
        $found = $false
        foreach ($area in $queue.focusAreas) {
            if ($area.id -eq $AreaId) {
                $area.completed = $true
                $area.completedAt = $timestamp
                $found = $true
            }
        }

        if (-not $found) {
            # Area was picked by scoring, not originally queued — add it
            $queue.focusAreas = @($queue.focusAreas) + @([PSCustomObject]@{
                id = $AreaId
                completed = $true
                startedAt = $timestamp
                completedAt = $timestamp
            })
        }

        # Update session info
        if ($queue.session) {
            $queue.session.lastActivityAt = $timestamp
            $queue.session.iterationCount = $script:State.IterationCount
        }

        Save-Queue -Queue $queue

        if (-not $Silent) {
            Write-Host "  Marked '$AreaId' as completed" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  Warning: Could not update queue.json" -ForegroundColor Yellow
    }
}

function Sync-QueueFromHistory {
    <#
    .SYNOPSIS
        Reconcile queue.json with sprint_history.json and current PRD.
        - Marks queued areas as completed if found in sprint history
        - Adds areas from sprint history that aren't in the queue
        - Adds the current PRD's focus area as in-progress if not tracked
    .PARAMETER Silent
        If set, don't print per-area messages
    #>
    param([switch]$Silent)

    $queue = Get-QueueData
    if (-not $queue) { return }

    if (-not $queue.focusAreas) {
        $queue | Add-Member -NotePropertyName focusAreas -NotePropertyValue @() -Force
    }

    $history = Read-JsonFile -Path $script:SprintHistoryFile
    $updated = $false
    $queuedIds = @($queue.focusAreas | ForEach-Object { $_.id })

    # Sync completed areas from sprint history
    if ($history -and $history.focusAreaBreakdown) {
        $completedInHistory = @($history.focusAreaBreakdown.PSObject.Properties | ForEach-Object { $_.Name })

        foreach ($area in $queue.focusAreas) {
            if (-not $area.completed -and $completedInHistory -contains $area.id) {
                $area.completed = $true
                $area.completedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
                $updated = $true
                if (-not $Silent) {
                    Write-Host "  Queue sync: marked '$($area.id)' as completed (found in sprint history)" -ForegroundColor Green
                }
            }
        }

        # Add areas from history that aren't tracked in the queue
        foreach ($areaId in $completedInHistory) {
            if ($queuedIds -notcontains $areaId) {
                $queue.focusAreas = @($queue.focusAreas) + @([PSCustomObject]@{
                    id = $areaId
                    completed = $true
                    startedAt = $null
                    completedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
                })
                $queuedIds = @($queuedIds) + @($areaId)
                $updated = $true
                if (-not $Silent) {
                    Write-Host "  Queue sync: added '$areaId' (completed in sprint history)" -ForegroundColor Green
                }
            }
        }
    }

    # Track the current PRD's focus area as in-progress
    $prd = Read-JsonFile -Path $script:PrdFile -Silent
    if ($prd -and $prd.focusArea) {
        $hasIncompleteStories = $prd.userStories -and @($prd.userStories | Where-Object { -not $_.passes }).Count -gt 0

        if ($queuedIds -notcontains $prd.focusArea) {
            # Area not in queue at all — add it
            $queue.focusAreas = @($queue.focusAreas) + @([PSCustomObject]@{
                id = $prd.focusArea
                completed = $false
                startedAt = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
                completedAt = $null
            })
            $updated = $true
            if (-not $Silent) {
                Write-Host "  Queue sync: added '$($prd.focusArea)' (current sprint)" -ForegroundColor Green
            }
        } elseif ($hasIncompleteStories) {
            # Area exists but marked completed — re-open for active sprint
            foreach ($area in $queue.focusAreas) {
                if ($area.id -eq $prd.focusArea -and $area.completed) {
                    $area.completed = $false
                    $area.completedAt = $null
                    $updated = $true
                    if (-not $Silent) {
                        Write-Host "  Queue sync: re-opened '$($prd.focusArea)' (active sprint)" -ForegroundColor Green
                    }
                }
            }
        }
    }

    if ($updated) {
        Save-Queue -Queue $queue
    }
}

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
    $duration = (Get-Date) - $script:State.SessionStartTime
    Write-Host ""
    Write-Host "  Session Stats:" -ForegroundColor DarkGray
    Write-Host "    Iterations: $($script:State.IterationCount)" -ForegroundColor DarkGray
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
