# Mutation testing for QueueProgressOnCompletion
# Applies mutations IN MEMORY and runs source-inspection assertions against mutated strings
# Does not modify any files on disk.

$loopsFile = Join-Path $PSScriptRoot "..\lib\loops.ps1"
$original = Get-Content $loopsFile -Raw

function Get-FuncBody($source, $funcName) {
    $pattern = "function $funcName\s*\{([\s\S]*?)(?=\nfunction\s|\z)"
    [regex]::Match($source, $pattern).Value
}

$mutations = @(
    @{
        Name = "M1: Remove Update-QueueProgress from TrueAutoLoop"
        Find = "Update-QueueProgress -AreaId `$status.focusArea -Silent"
        Replace = "# REMOVED queue progress update"
        FuncName = "Start-TrueAutoLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress'
        }
    },
    @{
        Name = "M2: Remove Update-QueueProgress from AdaptiveOvernightLoop"
        Find = "Update-QueueProgress -AreaId `$currentArea -Silent"
        Replace = "# REMOVED queue progress update"
        FuncName = "Start-AdaptiveOvernightLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress'
        }
    },
    @{
        Name = "M3: Wrong variable in TrueAutoLoop (use `$currentArea instead of `$status.focusArea)"
        Find = "Update-QueueProgress -AreaId `$status.focusArea -Silent"
        Replace = "Update-QueueProgress -AreaId `$currentArea -Silent"
        FuncName = "Start-TrueAutoLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea'
        }
    },
    @{
        Name = "M4: Wrong variable in AdaptiveOvernightLoop (use `$status.focusArea instead of `$currentArea)"
        Find = "Update-QueueProgress -AreaId `$currentArea -Silent"
        Replace = "Update-QueueProgress -AreaId `$status.focusArea -Silent"
        FuncName = "Start-AdaptiveOvernightLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress\s+-AreaId\s+\$currentArea'
        }
    },
    @{
        Name = "M5: Remove -Silent flag from TrueAutoLoop"
        Find = "Update-QueueProgress -AreaId `$status.focusArea -Silent"
        Replace = "Update-QueueProgress -AreaId `$status.focusArea"
        FuncName = "Start-TrueAutoLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea\s+-Silent'
        }
    },
    @{
        Name = "M6: Remove -Silent flag from AdaptiveOvernightLoop"
        Find = "Update-QueueProgress -AreaId `$currentArea -Silent"
        Replace = "Update-QueueProgress -AreaId `$currentArea"
        FuncName = "Start-AdaptiveOvernightLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'Update-QueueProgress\s+-AreaId\s+\$currentArea\s+-Silent'
        }
    },
    @{
        Name = "M7: Remove null guard in TrueAutoLoop"
        Find = "if (`$status.focusArea) {`n                    Update-QueueProgress"
        Replace = "Update-QueueProgress"
        FuncName = "Start-TrueAutoLoop"
        Test = {
            param($funcBody)
            $funcBody -match 'if\s*\(\$status\.focusArea\)\s*\{[\s\S]*?Update-QueueProgress'
        }
    },
    @{
        Name = "M8: Move Update-QueueProgress AFTER needsNewSprint in AdaptiveOvernight"
        Find = @"
            # Mark current area as complete in queue
            Update-QueueProgress -AreaId `$currentArea -Silent

            # Check for graceful stop
"@
        Replace = @"
            # Check for graceful stop
"@
        FuncName = "Start-AdaptiveOvernightLoop"
        Test = {
            param($funcBody)
            # QueueProgress must appear before needsNewSprint = $true (after archive)
            $archivePos = $funcBody.IndexOf('Save-SprintArchive')
            $queuePos = $funcBody.IndexOf('Update-QueueProgress')
            $newSprintPos = $funcBody.IndexOf('$needsNewSprint = $true', $archivePos)
            ($queuePos -gt $archivePos) -and ($queuePos -lt $newSprintPos)
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $mutated = $original.Replace($m.Find, $m.Replace)

    if ($mutated -eq $original) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in source" }
        continue
    }

    $funcBody = Get-FuncBody $mutated $m.FuncName
    $testResult = & $m.Test $funcBody

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== QUEUE PROGRESS MUTATION TESTING RESULTS ===" -ForegroundColor Cyan
Write-Host ""
foreach ($r in $results) {
    $color = switch ($r.Status) {
        "KILLED" { "Green" }
        "SURVIVED" { "Red" }
        default { "Yellow" }
    }
    Write-Host "$($r.Name): $($r.Status)" -ForegroundColor $color
    if ($r.Detail) { Write-Host "  -> $($r.Detail)" -ForegroundColor DarkGray }
}

$killed = @($results | Where-Object { $_.Status -eq "KILLED" }).Count
$survived = @($results | Where-Object { $_.Status -eq "SURVIVED" }).Count
$skipped = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
$total = $killed + $survived
Write-Host ""
if ($total -gt 0) {
    $pct = [math]::Round($killed / $total * 100)
    $color = if ($survived -eq 0) { "Green" } else { "Red" }
    Write-Host "Score: $killed/$total mutations killed ($pct%) | $skipped skipped" -ForegroundColor $color
} else {
    Write-Host "No mutations applied" -ForegroundColor Yellow
}
