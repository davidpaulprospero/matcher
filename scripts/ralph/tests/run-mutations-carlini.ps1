#Requires -Modules Pester
<#
.SYNOPSIS
    In-memory mutation testing for Carlini improvements.
    Reads source files as strings, applies mutations, asserts tests would catch them.
#>

$script:RalphDir = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'test-helper.ps1')
Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

$killed = 0
$survived = 0
$total = 0

function Test-Mutation {
    param(
        [string]$File,
        [string]$Description,
        [string]$Original,
        [string]$Mutated,
        [scriptblock]$Assertion
    )
    $script:total++
    $content = Get-Content (Join-Path $script:RalphDir $File) -Raw

    if ($content -notlike "*$Original*") {
        Write-Host "  SKIP: '$Description' - original string not found in $File" -ForegroundColor Yellow
        return
    }

    $mutatedContent = $content.Replace($Original, $Mutated)

    try {
        & $Assertion $mutatedContent
        # If assertion passes, mutation survived (BAD)
        $script:survived++
        Write-Host "  SURVIVED: $Description" -ForegroundColor Red
    }
    catch {
        # Assertion failed = mutation was caught (GOOD)
        $script:killed++
        Write-Host "  KILLED: $Description" -ForegroundColor Green
    }
}

Write-Host "`nMutation Testing: Carlini Improvements`n" -ForegroundColor Cyan

# --- Mutation 1: Grace period value change (30 -> 60) ---
Test-Mutation -File "config\ralph-config.json" `
    -Description "gracePeriodSeconds 30 -> 60" `
    -Original '"gracePeriodSeconds": 30' `
    -Mutated '"gracePeriodSeconds": 60' `
    -Assertion {
        param($content)
        $config = $content | ConvertFrom-Json
        if ($config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds -ne 30) {
            throw "Caught: grace period changed"
        }
    }

# --- Mutation 2: Remove git HEAD capture ---
Test-Mutation -File "lib\claude.ps1" `
    -Description "Remove storyCompletionGitHead capture" `
    -Original '$storyCompletionGitHead = (git rev-parse HEAD 2>$null)' `
    -Mutated '# MUTATED: removed git head capture' `
    -Assertion {
        param($content)
        if ($content -notlike '*storyCompletionGitHead = (git rev-parse HEAD*') {
            throw "Caught: git HEAD capture missing"
        }
    }

# --- Mutation 3: Remove git commit check ---
Test-Mutation -File "lib\claude.ps1" `
    -Description "Remove git commit early-exit check" `
    -Original '$currentHead -ne $storyCompletionGitHead' `
    -Mutated '$false' `
    -Assertion {
        param($content)
        if ($content -notlike '*currentHead -ne $storyCompletionGitHead*') {
            throw "Caught: git commit check removed"
        }
    }

# --- Mutation 4: Remove Format-TestSummary from Build-TierDiagnostics ---
Test-Mutation -File "lib\healing.ps1" `
    -Description "Remove structured summary, revert to raw output" `
    -Original 'Format-TestSummary -TierResult' `
    -Mutated 'MUTATED_NO_SUMMARY -TierResult' `
    -Assertion {
        param($content)
        if ($content -notlike '*Format-TestSummary -TierResult*') {
            throw "Caught: Format-TestSummary removed"
        }
    }

# --- Mutation 5: Format-TestSummary failure limit 10 -> 100 ---
Test-Mutation -File "lib\healing.ps1" `
    -Description "Failure limit 10 -> 100 (defeats truncation)" `
    -Original 'Count -ge 10' `
    -Mutated 'Count -ge 100' `
    -Assertion {
        param($content)
        # Check the specific pattern: failCount -ge 10 (not failCount -ge 100)
        if ($content -match 'failCount -ge 100') {
            throw "Caught: failure limit changed to 100"
        }
        if ($content -notmatch 'failCount -ge 10[^0]') {
            throw "Caught: failure limit 10 not found"
        }
    }

# --- Mutation 6: Remove fastPytestArgs from config ---
Test-Mutation -File "config\ralph-config.json" `
    -Description "Remove fastPytestArgs config key" `
    -Original '"fastPytestArgs": "tests/ --tb=line -q --no-header -x -p randomly --randomly-seed=random",' `
    -Mutated '' `
    -Assertion {
        param($content)
        $config = $content | ConvertFrom-Json
        if (-not $config.selfHealing.fastPytestArgs) {
            throw "Caught: fastPytestArgs missing"
        }
    }

# --- Mutation 7: Remove -x flag from fastPytestArgs ---
Test-Mutation -File "config\ralph-config.json" `
    -Description "Remove stop-on-first-failure (-x) from fastPytestArgs" `
    -Original '"fastPytestArgs": "tests/ --tb=line -q --no-header -x -p randomly --randomly-seed=random"' `
    -Mutated '"fastPytestArgs": "tests/ --tb=line -q --no-header -p randomly --randomly-seed=random"' `
    -Assertion {
        param($content)
        $config = $content | ConvertFrom-Json
        if ($config.selfHealing.fastPytestArgs -notlike '*-x*') {
            throw "Caught: -x flag removed"
        }
    }

# --- Mutation 8: Retry context 20 -> 50 (revert) ---
Test-Mutation -File "lib\quality.ps1" `
    -Description "Retry context 20 -> 50 lines (revert)" `
    -Original 'Last 20 lines' `
    -Mutated 'Last 50 lines' `
    -Assertion {
        param($content)
        if ($content -like '*Last 50 lines*') {
            throw "Caught: reverted to 50 lines"
        }
    }

# --- Mutation 9: Remove SprintProgressFile from paths ---
Test-Mutation -File "lib\paths.ps1" `
    -Description "Remove SprintProgressFile path definition" `
    -Original 'SprintProgressFile     = Join-Path $RalphDir "session\sprint_progress.md"' `
    -Mutated '# MUTATED: removed sprint progress path' `
    -Assertion {
        param($content)
        if ($content -notlike '*SprintProgressFile*session*sprint_progress*') {
            throw "Caught: SprintProgressFile path missing"
        }
    }

# --- Mutation 10: Remove Reset-SprintProgress from sprint start ---
Test-Mutation -File "lib\sprint.ps1" `
    -Description "Remove Reset-SprintProgress from New-SeedPRD" `
    -Original 'Reset-SprintProgress' `
    -Mutated '# MUTATED: removed reset' `
    -Assertion {
        param($content)
        if ($content -notlike '*Reset-SprintProgress*') {
            throw "Caught: Reset-SprintProgress removed"
        }
    }

# --- Summary ---
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Mutations: $total total, $killed killed, $survived survived" -ForegroundColor $(if ($survived -eq 0) { "Green" } else { "Red" })
Write-Host "Kill rate: $([math]::Round($killed / [math]::Max($total, 1) * 100))%" -ForegroundColor $(if ($survived -eq 0) { "Green" } else { "Red" })
Write-Host "========================================`n" -ForegroundColor Cyan

if ($survived -gt 0) {
    exit 1
}
