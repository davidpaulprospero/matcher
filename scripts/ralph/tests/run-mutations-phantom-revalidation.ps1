# Mutation testing for phantom revalidation fix in Invoke-ClaudeSubprocess
# The fix re-reads prd.json before overriding exit code to prevent false phantoms
# when Claude reverts passes:true→false during the grace period.
# Applies mutations IN MEMORY and runs assertions against mutated strings.
# Does not modify any files on disk.

$claudeFile = Join-Path $PSScriptRoot "..\lib\claude.ps1"
$original = Get-Content $claudeFile -Raw

# Extract Invoke-ClaudeSubprocess function body
$subprocessPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
function Get-SubprocessBody($source) {
    [regex]::Match($source, $subprocessPattern, 'Multiline').Groups[1].Value
}

# Extract just the override block (from guard to closing brace before finally)
function Get-OverrideBlock($funcBody) {
    [regex]::Match($funcBody, 'storyCompletionDetected\s+-and\s+\$exitCode\s+-ne\s+0[\s\S]*?(?=\s*\}\s*\n\s*finally)').Value
}

$mutations = @(
    @{
        Name = "M1: Remove PRD revalidation - always override (old behavior)"
        Find = @'
            $stillPasses = $false
            if ($StoryId -and $script:PrdFile -and (Test-Path $script:PrdFile)) {
                try {
                    $freshPrd = Get-Content $script:PrdFile -Raw -ErrorAction Stop | ConvertFrom-Json
                    $freshStory = $freshPrd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
                    $stillPasses = $freshStory -and $freshStory.passes -eq $true
                } catch {
                    # If we can't read, trust the original detection
                    $stillPasses = $true
                }
            } else {
                $stillPasses = $true
            }

            if ($stillPasses) {
                Write-Host "  [INFO] Overriding exit code $exitCode -> 0 (story completed, killed after grace period)" -ForegroundColor Cyan
                $exitCode = 0
            } else {
                Write-Host "  [INFO] Story passes was reverted during grace period - NOT overriding exit code $exitCode" -ForegroundColor Yellow
                $storyCompletionDetected = $false
            }
'@
        Replace = @'
            Write-Host "  [INFO] Overriding exit code $exitCode -> 0 (story completed, killed after grace period)" -ForegroundColor Cyan
            $exitCode = 0
'@
        Test = {
            param($overrideBlock)
            # Must have stillPasses check and PRD re-read
            $overrideBlock -match '\$stillPasses' -and $overrideBlock -match 'Get-Content\s+\$script:PrdFile'
        }
    },
    @{
        Name = "M2: Initialize stillPasses to true (always override)"
        Find = '$stillPasses = $false'
        Replace = '$stillPasses = $true'
        Test = {
            param($overrideBlock)
            # The init must be $false (safe default: don't override unless proven)
            $initLine = ($overrideBlock -split "`n" | Where-Object { $_ -match '^\s*\$stillPasses\s*=' } | Select-Object -First 1)
            $initLine -match '\$stillPasses\s*=\s*\$false'
        }
    },
    @{
        Name = "M3: Check passes -eq false instead of true (invert logic)"
        Find = '$stillPasses = $freshStory -and $freshStory.passes -eq $true'
        Replace = '$stillPasses = $freshStory -and $freshStory.passes -eq $false'
        Test = {
            param($overrideBlock)
            $overrideBlock -match 'freshStory\.passes\s+-eq\s+\$true' -and
            -not ($overrideBlock -match 'freshStory\.passes\s+-eq\s+\$false')
        }
    },
    @{
        Name = "M4: Remove storyCompletionDetected reset on revert"
        Find = '$storyCompletionDetected = $false'
        Replace = '# storyCompletionDetected left as-is'
        Test = {
            param($overrideBlock)
            # The else branch must reset storyCompletionDetected
            $elseBranch = [regex]::Match($overrideBlock, 'NOT overriding[\s\S]*?(?=\})').Value
            $elseBranch -match '\$storyCompletionDetected\s*=\s*\$false'
        }
    },
    @{
        Name = "M5: Remove Test-Path guard (could read non-existent file)"
        Find = '$StoryId -and $script:PrdFile -and (Test-Path $script:PrdFile)'
        Replace = '$StoryId -and $script:PrdFile'
        Test = {
            param($overrideBlock)
            $overrideBlock -match 'Test-Path\s+\$script:PrdFile'
        }
    },
    @{
        Name = "M6: Remove try/catch (crash on file lock)"
        Find = @'
                try {
                    $freshPrd = Get-Content $script:PrdFile -Raw -ErrorAction Stop | ConvertFrom-Json
                    $freshStory = $freshPrd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
                    $stillPasses = $freshStory -and $freshStory.passes -eq $true
                } catch {
                    # If we can't read, trust the original detection
                    $stillPasses = $true
                }
'@
        Replace = @'
                $freshPrd = Get-Content $script:PrdFile -Raw -ErrorAction Stop | ConvertFrom-Json
                $freshStory = $freshPrd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
                $stillPasses = $freshStory -and $freshStory.passes -eq $true
'@
        Test = {
            param($overrideBlock)
            $overrideBlock -match 'try\s*\{[\s\S]*?Get-Content\s+\$script:PrdFile[\s\S]*?catch'
        }
    },
    @{
        Name = "M7: Change catch fallback to false (reject on read failure)"
        Find = @'
                } catch {
                    # If we can't read, trust the original detection
                    $stillPasses = $true
                }
'@
        Replace = @'
                } catch {
                    # If we can't read, reject
                    $stillPasses = $false
                }
'@
        Test = {
            param($overrideBlock)
            # Catch block should trust original detection (set true)
            $catchBlock = [regex]::Match($overrideBlock, 'catch\s*\{[\s\S]*?\}').Value
            $catchBlock -match '\$stillPasses\s*=\s*\$true'
        }
    },
    @{
        Name = "M8: Swap if/else branches (override when reverted, skip when passing)"
        Find = @'
            if ($stillPasses) {
                Write-Host "  [INFO] Overriding exit code $exitCode -> 0 (story completed, killed after grace period)" -ForegroundColor Cyan
                $exitCode = 0
            } else {
                Write-Host "  [INFO] Story passes was reverted during grace period - NOT overriding exit code $exitCode" -ForegroundColor Yellow
                $storyCompletionDetected = $false
            }
'@
        Replace = @'
            if ($stillPasses) {
                Write-Host "  [INFO] Story passes was reverted during grace period - NOT overriding exit code $exitCode" -ForegroundColor Yellow
                $storyCompletionDetected = $false
            } else {
                Write-Host "  [INFO] Overriding exit code $exitCode -> 0 (story completed, killed after grace period)" -ForegroundColor Cyan
                $exitCode = 0
            }
'@
        Test = {
            param($overrideBlock)
            # exitCode = 0 must be inside the stillPasses=true branch
            $ifBlock = [regex]::Match($overrideBlock, 'if\s*\(\$stillPasses\)\s*\{([\s\S]*?)\}\s*else').Groups[1].Value
            $ifBlock -match '\$exitCode\s*=\s*0'
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

    $funcBody = Get-SubprocessBody $mutated
    $overrideBlock = Get-OverrideBlock $funcBody
    $testResult = & $m.Test $overrideBlock

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== MUTATION TESTING: Phantom Revalidation Fix ===" -ForegroundColor Cyan
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
    Write-Host "Score: $killed/$total mutations killed ($($pct)%) | $skipped skipped" -ForegroundColor $color
} else {
    Write-Host "No mutations applied" -ForegroundColor Yellow
}
