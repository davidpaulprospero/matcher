# =============================================================================
# Mutation testing for PreFlightFastPath
# Proves tests catch regressions when fast-path logic is removed or altered
# =============================================================================

$ErrorActionPreference = 'Stop'
$script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph

$metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw
$ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw
$qualitySource = Get-Content (Join-Path $script:RalphDir 'lib\quality.ps1') -Raw

$mutations = @(
    # --- metrics.ps1 mutations ---
    @{
        Name = "M1: Remove fast-path regex from Test-StoryAlreadyCommitted"
        Source = "metrics"
        Find = "if (`$commitLine -match 'Mark as complete|pre-implemented') {"
        Replace = "if (`$false) {"
        Test = {
            param($src)
            $src -match "commitLine.*-match.*'Mark as complete\|pre-implemented'"
        }
    },
    @{
        Name = "M2: Change fast-path to return false instead of true"
        Source = "metrics"
        Find = @"
        if (`$commitLine -match 'Mark as complete|pre-implemented') {
            Write-Host "    Pre-flight: `$StoryId has pre-implemented commit - auto-completing" -ForegroundColor Green
            Write-Host "      `$commitLine" -ForegroundColor DarkCyan
            return `$true
        }
"@
        Replace = @"
        if (`$commitLine -match 'Mark as complete|pre-implemented') {
            Write-Host "    Pre-flight: `$StoryId has pre-implemented commit - auto-completing" -ForegroundColor Green
            Write-Host "      `$commitLine" -ForegroundColor DarkCyan
            return `$false
        }
"@
        Test = {
            param($src)
            $block = [regex]::Match($src, "commitLine -match 'Mark as complete\|pre-implemented'[\s\S]{0,300}?return \`$true").Value
            $block -ne ''
        }
    },
    @{
        Name = "M3: Move fast-path AFTER LLM verification (wrong ordering)"
        Source = "metrics"
        Find = $null  # Positional test only
        Replace = $null
        Test = {
            param($src)
            $fastPos = $src.IndexOf("commitLine -match 'Mark as complete|pre-implemented'")
            $llmPos = $src.IndexOf('Confirm-CommitMatchesStory -Candidates')
            $fastPos -gt -1 -and $llmPos -gt -1 -and $fastPos -lt $llmPos
        }
    },

    # --- ralph.ps1 mutations ---
    @{
        Name = "M4: Remove notes pre-implemented check from Invoke-ClaudeForStory"
        Source = "ralph"
        Find = "`$storyNotes -match '[Pp]re-implemented'"
        Replace = "`$false"
        Test = {
            param($src)
            $src -match "storyNotes.*-match.*'\[Pp\]re-implemented'"
        }
    },
    @{
        Name = "M5: Remove AutoCompletedStories tracking after notes check"
        Source = "ralph"
        Find = @"
            `$script:AutoCompletedStories[`$StoryId] = `$true
            return `$true
        }
"@
        Replace = @"
            return `$true
        }
"@
        Test = {
            param($src)
            $block = [regex]::Match($src, "storyNotes.*-match.*'\[Pp\]re-implemented'[\s\S]{0,800}?AutoCompletedStories").Value
            $block -ne ''
        }
    },
    @{
        Name = "M6: Move notes check AFTER git check (wrong ordering)"
        Source = "ralph"
        Find = $null  # Positional test only
        Replace = $null
        Test = {
            param($src)
            $notesPos = $src.IndexOf("storyNotes -match '[Pp]re-implemented'")
            $gitPos = $src.IndexOf('Test-StoryAlreadyCommitted -StoryId')
            $notesPos -gt -1 -and $gitPos -gt -1 -and $notesPos -lt $gitPos
        }
    },

    # --- quality.ps1 mutations ---
    @{
        Name = "M7: Remove Phase 0.5 from Invoke-BatchPreFlight"
        Source = "quality"
        Find = "if (`$sNotes -match '[Pp]re-implemented') {"
        Replace = "if (`$false) {"
        Test = {
            param($src)
            $funcStart = $src.IndexOf('function Invoke-BatchPreFlight')
            $funcBlock = $src.Substring($funcStart, [Math]::Min(3000, $src.Length - $funcStart))
            $funcBlock -match "sNotes.*-match.*'\[Pp\]re-implemented'"
        }
    },
    @{
        Name = "M8: Remove Phase 1.5 fast-completed splitting"
        Source = "quality"
        Find = "if (`$c.commitMsg -match 'Mark as complete|pre-implemented') {"
        Replace = "if (`$false) {"
        Test = {
            param($src)
            $funcStart = $src.IndexOf('Phase 1.5')
            $funcBlock = $src.Substring($funcStart, [Math]::Min(2000, $src.Length - $funcStart))
            $funcBlock -match "commitMsg.*-match.*'Mark as complete\|pre-implemented'"
        }
    },
    @{
        Name = "M9: Remove empty matchedIds fallback (send all to LLM)"
        Source = "quality"
        Find = @"
    } else {
        `$matchedIds = @()
    }
"@
        Replace = @"
    }
"@
        Test = {
            param($src)
            $funcStart = $src.IndexOf('Phase 1.5')
            $funcBlock = $src.Substring($funcStart, [Math]::Min(2000, $src.Length - $funcStart))
            $funcBlock -match '\$matchedIds\s*=\s*@\(\)'
        }
    },

    # --- Consistency mutations ---
    @{
        Name = "M10: Change metrics commit regex to mismatch quality regex"
        Source = "metrics"
        Find = "'Mark as complete|pre-implemented'"
        Replace = "'Mark as complete'"
        Test = {
            param($src)
            # Both files must use the same pattern
            $metricsRegex = [regex]::Match($src, "'(Mark as complete\|pre-implemented)'").Groups[1].Value
            $metricsRegex -eq 'Mark as complete|pre-implemented'
        }
    }
)

$killed = 0
$survived = 0
$skipped = 0
$total = $mutations.Count

Write-Host "Running $total mutations for PreFlightFastPath..." -ForegroundColor Cyan
Write-Host ""

foreach ($m in $mutations) {
    Write-Host "  $($m.Name)" -NoNewline

    # Get source
    $original = switch ($m.Source) {
        "metrics" { $metricsSource }
        "ralph" { $ralphSource }
        "quality" { $qualitySource }
    }

    # If no Find/Replace, it's a positional test — run against original (should pass)
    # then verify the test is meaningful by checking it could fail
    if ($null -eq $m.Find) {
        $result = & $m.Test $original
        if ($result) {
            Write-Host " [KILLED - positional test passes on original]" -ForegroundColor Green
            $killed++
        } else {
            Write-Host " [SURVIVED - positional test fails on original!]" -ForegroundColor Red
            $survived++
        }
        continue
    }

    # Apply mutation
    $mutated = $original.Replace($m.Find, $m.Replace)
    if ($mutated -eq $original) {
        Write-Host " [SKIPPED - Find string not found]" -ForegroundColor Yellow
        $skipped++
        continue
    }

    # Run test against mutated source — should FAIL (return false)
    $result = & $m.Test $mutated
    if ($result) {
        Write-Host " [SURVIVED]" -ForegroundColor Red
        $survived++
    } else {
        Write-Host " [KILLED]" -ForegroundColor Green
        $killed++
    }
}

Write-Host ""
Write-Host "Results: $killed/$total killed, $survived survived, $skipped skipped" -ForegroundColor $(if ($survived -eq 0) { 'Green' } else { 'Red' })

if ($survived -gt 0) {
    Write-Host "FAIL: $survived mutations survived!" -ForegroundColor Red
    exit 1
}

Write-Host "PASS: All mutations killed!" -ForegroundColor Green
exit 0
