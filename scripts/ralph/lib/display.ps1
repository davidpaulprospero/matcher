# scripts/ralph/lib/display.ps1
# Display and UI: banners, reasoning output, user input prompts, choice logging

function Write-RalphBanner {
    param(
        [switch]$Queue,
        [switch]$TrueAuto,
        [switch]$RalphsChoice,
        [switch]$RalphsChoiceAuto
    )

    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "   'Me fail English? That's unpossible!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Ralph Loop v2.0" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""

    if ($Queue) {
        Write-Host "  Mode: Interview Queue" -ForegroundColor Green
    }
    elseif ($TrueAuto) {
        Write-Host "  Mode: TrueAuto (continuous improvement)" -ForegroundColor Magenta
    }
    elseif ($RalphsChoice) {
        Write-Host "  Mode: Ralph's Choice (Ralph decides, you confirm)" -ForegroundColor Magenta
    }
    elseif ($RalphsChoiceAuto) {
        Write-Host "  Mode: Ralph's Choice Auto (fully autonomous)" -ForegroundColor Magenta
    }
    else {
        Write-Host "  Mode: Standard" -ForegroundColor White
    }

    Write-Host "  Session: $($script:State.SessionId)" -ForegroundColor DarkGray
    Write-Host ""
}

function Write-IterationBanner {
    param(
        [int]$Iteration,
        [string]$FocusArea,
        [string]$StoryId = ""
    )

    Write-Host ""
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Iteration $Iteration" -ForegroundColor Cyan
    if ($FocusArea) {
        Write-Host "  Focus: $FocusArea" -ForegroundColor Yellow
    }
    if ($StoryId) {
        Write-Host "  Story: $StoryId" -ForegroundColor Green
    }
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
}

function Show-RalphsReasoning {
    <#
    .SYNOPSIS
        Display scoring breakdown and recommendation
    .PARAMETER Scores
        Array of score hashtables from Get-AllFocusAreaScores
    .PARAMETER Decision
        Optional stay/switch decision hashtable
    .PARAMETER ShowAllScores
        If true, show all scores instead of top 5
    #>
    param(
        [Parameter(Mandatory=$true)]
        $Scores,
        $Decision = $null,
        [switch]$ShowAllScores
    )

    $config = Get-RalphConfig
    $sprintHistory = Get-SprintHistory
    $windowDays = if ($config.ralphsChoice.activityWindowDays) { $config.ralphsChoice.activityWindowDays } else { 7 }
    $activityByArea = Get-GitActivityByArea -WindowDays $windowDays

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host "     Ralph's Choice - Analyzing project state..." -ForegroundColor Magenta
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host ""

    # Git Activity summary
    Write-Host "  Git Activity (last $windowDays days):" -ForegroundColor Yellow
    $topActivity = $activityByArea.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5
    $anyPrinted = $false
    foreach ($entry in $topActivity) {
        if ($entry.Value -gt 0) {
            Write-Host "    * $($entry.Key.PadRight(20)) $($entry.Value) commits" -ForegroundColor White
            $anyPrinted = $true
        }
    }
    if (-not $anyPrinted) {
        Write-Host "    (no recent activity)" -ForegroundColor DarkGray
    }
    Write-Host ""

    # Sprint History summary
    Write-Host "  Sprint History:" -ForegroundColor Yellow
    $totalSprints = [int]$sprintHistory.totalSprintsCompleted
    $totalStories = [int]$sprintHistory.totalStoriesCompleted
    Write-Host "    Total: $totalSprints sprints, $totalStories stories" -ForegroundColor White

    if ($sprintHistory.focusAreaBreakdown) {
        foreach ($prop in $sprintHistory.focusAreaBreakdown.PSObject.Properties) {
            $areaId = $prop.Name
            $stats = $prop.Value
            $sprints = [int]$stats.sprints
            $stories = [int]$stats.stories

            # Find last sprint time
            $lastTime = "never"
            $areaSprints = @($sprintHistory.sprints | Where-Object { $_.focusArea -eq $areaId } | Sort-Object completedAt -Descending)
            if ($areaSprints.Count -gt 0) {
                $lastSprint = $areaSprints | Select-Object -First 1
                if ($lastSprint.completedAt) {
                    $hoursAgo = [math]::Round(((Get-Date) - [datetime]$lastSprint.completedAt).TotalHours, 0)
                    if ($hoursAgo -lt 24) {
                        $lastTime = "${hoursAgo}h ago"
                    } else {
                        $daysAgo = [math]::Floor($hoursAgo / 24)
                        $lastTime = "${daysAgo}d ago"
                    }
                }
            }

            Write-Host "    * ${areaId}: $sprints sprints ($stories stories) - last: $lastTime" -ForegroundColor White
        }
    }
    Write-Host ""

    # Category Coverage
    Write-Host "  Category Coverage:" -ForegroundColor Yellow
    foreach ($catId in $config.categoryOrder) {
        $category = $config.focusAreaCategories.$catId
        if (-not $category) { continue }

        $catSprints = 0
        foreach ($areaId in $category.areas) {
            if ($sprintHistory.focusAreaBreakdown -and $sprintHistory.focusAreaBreakdown.$areaId) {
                $catSprints += [int]$sprintHistory.focusAreaBreakdown.$areaId.sprints
            }
        }

        $percentage = if ($totalSprints -gt 0) { [math]::Round(($catSprints / $totalSprints) * 100, 0) } else { 0 }
        $status = if ($percentage -gt 50) { "(over-represented)" } elseif ($percentage -eq 0) { "(untouched)" } else { "" }
        Write-Host "    * ${catId}: ${percentage}% of work $status" -ForegroundColor White
    }
    Write-Host ""

    # Scores
    Write-Host "  Scores:" -ForegroundColor Yellow
    $displayScores = if ($ShowAllScores) { $Scores } else { $Scores | Select-Object -First 5 }
    foreach ($score in $displayScores) {
        $dots = "." * [Math]::Max(1, 25 - $score.areaId.Length)
        Write-Host "    $($score.areaId) $dots $($score.total)" -ForegroundColor White
    }
    Write-Host ""

    # Recommendation
    if (-not $Scores -or $Scores.Count -eq 0) {
        Write-Host "  No scores available" -ForegroundColor Yellow
        return
    }
    $topScore = $Scores[0]
    Write-Host "  > Ralph recommends: " -ForegroundColor Cyan -NoNewline
    Write-Host $topScore.areaId -ForegroundColor Green
    Write-Host "    Reason: " -ForegroundColor Cyan -NoNewline

    $reasons = @()
    if ($topScore.commits -gt 0) {
        $reasons += "Git activity ($($topScore.commits) commits)"
    }
    if ($topScore.sprintsDone -eq 0) {
        $reasons += "never worked on"
    }
    if ($topScore.balanceScore -gt 0.5) {
        $reasons += "category needs balance"
    }

    if ($reasons.Count -eq 0) {
        $reasons += "highest overall score"
    }

    Write-Host ($reasons -join " + ") -ForegroundColor White
    Write-Host ""

    # If this is a stay/switch decision, show that context
    if ($Decision) {
        Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Stay vs Switch Analysis:" -ForegroundColor Yellow
        Write-Host ""

        if ($Decision.stayReasons.Count -gt 0) {
            Write-Host "  Reasons to STAY in $($Decision.currentArea):" -ForegroundColor White
            foreach ($reason in $Decision.stayReasons) {
                Write-Host "    + $reason" -ForegroundColor Green
            }
        } else {
            Write-Host "  Reasons to STAY:" -ForegroundColor White
            Write-Host "    (none)" -ForegroundColor DarkGray
        }
        Write-Host ""

        if ($Decision.switchReasons.Count -gt 0) {
            Write-Host "  Reasons to SWITCH:" -ForegroundColor White
            foreach ($reason in $Decision.switchReasons) {
                Write-Host "    + $reason" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Reasons to SWITCH:" -ForegroundColor White
            Write-Host "    (none)" -ForegroundColor DarkGray
        }
        Write-Host ""

        $decisionText = if ($Decision.decision -eq "switch") { "SWITCH to $($Decision.newArea)" } else { "STAY in $($Decision.currentArea)" }
        Write-Host "  > Ralph recommends: " -ForegroundColor Cyan -NoNewline
        Write-Host $decisionText -ForegroundColor $(if ($Decision.decision -eq "switch") { "Yellow" } else { "Green" })
        Write-Host ""
    }
}

function Get-RalphsChoiceUserInput {
    <#
    .SYNOPSIS
        Get user input for Ralph's Choice decision
    .PARAMETER Scores
        Array of score hashtables
    .PARAMETER IsStaySwitch
        If true, this is a between-sprint decision
    .PARAMETER CurrentArea
        Current focus area (for stay/switch)
    .RETURNS
        Hashtable with: action (accept/manual/stay/showAll), selectedArea
    #>
    param(
        [Parameter(Mandatory=$true)]
        $Scores,
        [switch]$IsStaySwitch,
        [string]$CurrentArea = ""
    )

    Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
    if ($IsStaySwitch) {
        Write-Host "  [Enter] Accept   [S] See all scores   [K] Keep $CurrentArea   [M] Manual" -ForegroundColor White
    } else {
        Write-Host "  [Enter] Accept   [S] See all scores   [M] Manual override" -ForegroundColor White
    }
    Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Ss]$" {
            return @{ action = "showAll"; selectedArea = $null }
        }
        "^[Kk]$" {
            if ($IsStaySwitch -and $CurrentArea) {
                return @{ action = "stay"; selectedArea = $CurrentArea }
            }
            return @{ action = "accept"; selectedArea = $Scores[0].areaId }
        }
        "^[Mm]$" {
            Write-Host ""
            Write-Host "  Enter focus area ID:" -ForegroundColor Cyan
            $manualArea = Read-Host "  "
            return @{ action = "manual"; selectedArea = $manualArea }
        }
        default {
            return @{ action = "accept"; selectedArea = $Scores[0].areaId }
        }
    }
}

function Write-RalphsChoiceLog {
    <#
    .SYNOPSIS
        Log Ralph's Choice decisions to file
    .PARAMETER Decision
        The decision made (area selected, stay/switch)
    .PARAMETER Reason
        Reason for the decision
    .PARAMETER Scores
        Score breakdown at time of decision
    #>
    param(
        [string]$Decision,
        [string]$Reason,
        $Scores
    )

    $config = Get-RalphConfig
    $logFile = if ($config.ralphsChoice.autoMode.logFile) { $config.ralphsChoice.autoMode.logFile } else { "ralphs_choices.log" }
    $logPath = Join-Path $script:RalphDir $logFile

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $topScores = ($Scores | Select-Object -First 3 | ForEach-Object { "$($_.areaId):$($_.total)" }) -join ", "

    $logLine = "[$timestamp] Decision: $Decision | Reason: $Reason | TopScores: $topScores"
    try {
        Add-Content -Path $logPath -Value $logLine -Encoding UTF8
    } catch {
        Start-Sleep -Milliseconds 200
        try {
            Add-Content -Path $logPath -Value $logLine -Encoding UTF8
        } catch {
            Write-Host "  Warning: Failed to write choice log: $_" -ForegroundColor Yellow
        }
    }
}
