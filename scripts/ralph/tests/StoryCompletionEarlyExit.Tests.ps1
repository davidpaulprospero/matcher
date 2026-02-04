#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for story completion early exit feature in Invoke-ClaudeSubprocess
.DESCRIPTION
    Tests cover the fix for Claude continuing to run minutes after a story is
    marked passes:true in prd.json. The monitoring loop now reads prd.json on
    update, detects story completion, and kills Claude after a configurable
    grace period (default 15s).

    Root cause: The monitoring loop treated prd.json updates as generic "activity"
    and reset the inactivity timer, giving Claude a fresh window (up to 45 min for
    quality focus area with pytest multiplier) even though the story was already done.

    Fix: Added StoryId parameter to Invoke-ClaudeSubprocess and early exit logic
    that checks prd.json content when it changes, killing Claude after a short
    grace period when the story shows passes:true.
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

# =============================================================================
# SOURCE INSPECTION TESTS - Verify structural correctness
# =============================================================================

Describe 'Invoke-ClaudeSubprocess StoryId parameter' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'has StoryId parameter in Invoke-ClaudeSubprocess' {
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
        $funcBody | Should -Match '\[string\]\$StoryId\s*=\s*""'
    }

    It 'passes StoryId from Invoke-ClaudeWithInfiniteRetry to Invoke-ClaudeSubprocess' {
        $script:claudeSource | Should -Match 'Invoke-ClaudeSubprocess\s[^}]*-StoryId\s+\$StoryId'
    }

    It 'Invoke-ClaudeWithInfiniteRetry has StoryId parameter' {
        $funcPattern = 'function Invoke-ClaudeWithInfiniteRetry\s*\{([\s\S]*?)^\}'
        $funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
        $funcBody | Should -Match '\[string\]\$StoryId\s*=\s*""'
    }
}

Describe 'Early exit initialization' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'reads storyCompletionEarlyExit config' {
        $script:funcBody | Should -Match 'Config\.stallDetection\.storyCompletionEarlyExit'
    }

    It 'initializes storyCompletionDetected to false' {
        $script:funcBody | Should -Match '\$storyCompletionDetected\s*=\s*\$false'
    }

    It 'initializes storyCompletionTime to null' {
        $script:funcBody | Should -Match '\$storyCompletionTime\s*=\s*\$null'
    }

    It 'reads gracePeriodSeconds from config with fallback to 15' {
        $script:funcBody | Should -Match 'gracePeriodSeconds.*\}\s*else\s*\{\s*15\s*\}'
    }

    It 'reads enabled flag from config with fallback to true' {
        $script:funcBody | Should -Match 'earlyExitEnabled.*\}\s*else\s*\{\s*\$true\s*\}'
    }
}

Describe 'Early exit PRD check logic' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'checks earlyExitEnabled before reading PRD' {
        $script:funcBody | Should -Match '\$earlyExitEnabled\s+-and\s+\$prdUpdated\s+-and\s+\$StoryId'
    }

    It 'only checks when not already detected' {
        $script:funcBody | Should -Match '-not\s+\$storyCompletionDetected'
    }

    It 'reads PrdFile with -Raw and ConvertFrom-Json' {
        $script:funcBody | Should -Match 'Get-Content\s+\$script:PrdFile\s+-Raw.*ConvertFrom-Json'
    }

    It 'filters userStories by StoryId' {
        $script:funcBody | Should -Match 'userStories.*Where-Object.*\$_\.id\s+-eq\s+\$StoryId'
    }

    It 'checks passes equals true' {
        $script:funcBody | Should -Match '\$story\.passes\s+-eq\s+\$true'
    }

    It 'sets storyCompletionDetected to true on match' {
        $script:funcBody | Should -Match '\$storyCompletionDetected\s*=\s*\$true'
    }

    It 'records storyCompletionTime with Get-Date' {
        $script:funcBody | Should -Match '\$storyCompletionTime\s*=\s*Get-Date'
    }

    It 'wraps PRD read in try/catch for file lock safety' {
        # The try block must contain the Get-Content call
        $tryPattern = 'try\s*\{[^}]*Get-Content\s+\$script:PrdFile'
        $script:funcBody | Should -Match $tryPattern
    }
}

Describe 'Early exit grace period and kill logic' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'computes time since completion using datetime subtraction' {
        $script:funcBody | Should -Match '\(Get-Date\)\s*-\s*\$storyCompletionTime\)\.TotalSeconds'
    }

    It 'compares elapsed time against grace period' {
        $script:funcBody | Should -Match '\$sinceCompletion\s+-ge\s+\$storyCompletionGraceSec'
    }

    It 'kills process tree with taskkill /T /F' {
        $script:funcBody | Should -Match 'taskkill\s+/T\s+/F\s+/PID\s+\$process\.Id'
    }

    It 'has fallback Kill() call if process still running' {
        $script:funcBody | Should -Match '\$process\.Kill\(\)'
    }

    It 'breaks out of monitoring loop after kill' {
        # The break must be inside the grace period block
        $graceBlock = [regex]::Match($script:funcBody, 'Grace period expired[\s\S]*?break').Value
        $graceBlock | Should -Not -BeNullOrEmpty
    }

    It 'shows green message when story marked DONE' {
        $script:funcBody | Should -Match 'Story.*marked DONE.*grace period.*ForegroundColor Green'
    }

    It 'shows yellow message when grace period expires' {
        $script:funcBody | Should -Match 'Grace period expired.*terminating Claude.*ForegroundColor Yellow'
    }
}

Describe 'Activity reason includes story DONE context' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'shows "story DONE, grace period" in activity reason when completion detected' {
        $script:funcBody | Should -Match 'prd\.json \(story DONE, grace period\)'
    }

    It 'checks storyCompletionDetected in reason string selection' {
        $script:funcBody | Should -Match '\$prdUpdated\s+-and\s+\$storyCompletionDetected'
    }
}

# =============================================================================
# CONFIG TESTS
# =============================================================================

Describe 'ralph-config.json storyCompletionEarlyExit' -Tag 'Unit', 'EarlyExit' {
    BeforeAll {
        $configPath = Join-Path $script:RalphDir 'config\ralph-config.json'
        $script:config = Get-Content $configPath -Raw | ConvertFrom-Json
    }

    It 'has storyCompletionEarlyExit section under stallDetection' {
        $script:config.stallDetection.storyCompletionEarlyExit | Should -Not -BeNullOrEmpty
    }

    It 'has enabled set to true' {
        $script:config.stallDetection.storyCompletionEarlyExit.enabled | Should -BeTrue
    }

    It 'has gracePeriodSeconds set to 15' {
        $script:config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds | Should -Be 15
    }

    It 'gracePeriodSeconds is a positive number' {
        $script:config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds | Should -BeGreaterThan 0
    }
}

# =============================================================================
# MUTATION TESTS - Catch value/logic inversions
# =============================================================================

Describe 'Mutation testing - early exit values' -Tag 'Unit', 'EarlyExit', 'Mutation' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'default grace period is exactly 15 (not 0, 5, 30, or 60)' {
        $script:funcBody | Should -Match 'else\s*\{\s*15\s*\}'
        $script:funcBody | Should -Not -Match 'else\s*\{\s*0\s*\}'
        $script:funcBody | Should -Not -Match 'else\s*\{\s*5\s*\}'
        $script:funcBody | Should -Not -Match 'else\s*\{\s*30\s*\}'
        $script:funcBody | Should -Not -Match 'else\s*\{\s*60\s*\}'
    }

    It 'default enabled is $true not $false' {
        # Match the earlyExitEnabled fallback specifically
        $enabledLine = ($script:funcBody -split "`n" | Where-Object { $_ -match 'earlyExitEnabled' -and $_ -match 'else' })
        $enabledLine | Should -Match '\$true'
        $enabledLine | Should -Not -Match '\$false'
    }

    It 'storyCompletionDetected initializes to $false not $true' {
        # Match the initialization line (before the while loop), not the assignment inside the if-block
        # The init line is: $storyCompletionDetected = $false
        # The assignment line is: $storyCompletionDetected = $true (inside if block with story.passes)
        $script:funcBody | Should -Match '\$storyCompletionDetected\s*=\s*\$false'
    }

    It 'storyCompletionTime initializes to $null not Get-Date' {
        # Match the initialization line, not the assignment inside the if-block
        $script:funcBody | Should -Match '\$storyCompletionTime\s*=\s*\$null'
    }

    It 'checks passes -eq $true not -eq $false' {
        $passesLine = ($script:funcBody -split "`n" | Where-Object { $_ -match 'story\.passes' })
        $passesLine | Should -Match '-eq\s+\$true'
        $passesLine | Should -Not -Match '-eq\s+\$false'
    }

    It 'uses -ge for grace period comparison not -gt or -le' {
        $compareLine = ($script:funcBody -split "`n" | Where-Object { $_ -match 'sinceCompletion' -and $_ -match 'storyCompletionGraceSec' })
        $compareLine | Should -Match '-ge'
        $compareLine | Should -Not -Match '-gt\s'
        $compareLine | Should -Not -Match '-le\s'
        $compareLine | Should -Not -Match '-lt\s'
    }

    It 'uses taskkill /T /F (tree + force) not just /F or /T alone' {
        # Find the grace period kill block: from "Grace period expired" to "break"
        $graceBlock = [regex]::Match($script:funcBody, 'Grace period expired[\s\S]{0,500}?break').Value
        $graceBlock | Should -Not -BeNullOrEmpty
        $graceBlock | Should -Match 'taskkill /T /F /PID'
    }
}

Describe 'Mutation testing - early exit guard conditions' -Tag 'Unit', 'EarlyExit', 'Mutation' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'
        $script:funcBody = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline').Groups[1].Value
    }

    It 'PRD check requires all three: earlyExitEnabled AND prdUpdated AND StoryId' {
        $guardLine = ($script:funcBody -split "`n" | Where-Object { $_ -match 'earlyExitEnabled.*prdUpdated.*StoryId' })
        $guardLine | Should -Not -BeNullOrEmpty
        $guardLine | Should -Match '\$earlyExitEnabled\s+-and\s+\$prdUpdated\s+-and\s+\$StoryId'
    }

    It 'grace period kill requires both storyCompletionDetected AND storyCompletionTime' {
        $script:funcBody | Should -Match '\$storyCompletionDetected\s+-and\s+\$storyCompletionTime'
    }

    It 'does NOT check earlyExitEnabled in the grace period kill block (already gated at detection)' {
        # The kill block should only check storyCompletionDetected + storyCompletionTime
        # not re-check earlyExitEnabled (it was already checked at detection time)
        $killGuard = ($script:funcBody -split "`n" | Where-Object { $_ -match 'storyCompletionDetected\s+-and\s+\$storyCompletionTime' })
        $killGuard | Should -Not -Match 'earlyExitEnabled'
    }
}

Describe 'Mutation testing - config key names' -Tag 'Unit', 'EarlyExit', 'Mutation' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $configPath = Join-Path $script:RalphDir 'config\ralph-config.json'
        $script:configRaw = Get-Content $configPath -Raw
    }

    It 'claude.ps1 references same config key as ralph-config.json: storyCompletionEarlyExit' {
        $script:claudeSource | Should -Match 'storyCompletionEarlyExit'
        $script:configRaw | Should -Match 'storyCompletionEarlyExit'
    }

    It 'claude.ps1 references same property as config: gracePeriodSeconds' {
        $script:claudeSource | Should -Match 'gracePeriodSeconds'
        $script:configRaw | Should -Match 'gracePeriodSeconds'
    }

    It 'claude.ps1 references same property as config: enabled' {
        $script:claudeSource | Should -Match 'earlyExitConfig.*enabled'
        $script:configRaw | Should -Match '"storyCompletionEarlyExit"[\s\S]*?"enabled"'
    }
}

# =============================================================================
# BEHAVIORAL SIMULATION TESTS
# =============================================================================

Describe 'PRD story completion detection simulation' -Tag 'Unit', 'EarlyExit', 'Simulation' {
    It 'detects passes:true for a specific story in PRD JSON' {
        $prdJson = @{
            userStories = @(
                @{ id = "US-53-001"; passes = $true; title = "Already done" }
                @{ id = "US-53-002"; passes = $false; title = "Not done" }
                @{ id = "US-53-003"; passes = $true; title = "Also done" }
            )
        } | ConvertTo-Json -Depth 3

        $prd = $prdJson | ConvertFrom-Json
        $story = $prd.userStories | Where-Object { $_.id -eq "US-53-003" }
        $story | Should -Not -BeNullOrEmpty
        $story.passes | Should -BeTrue
    }

    It 'returns null for non-existent story ID' {
        $prdJson = @{
            userStories = @(
                @{ id = "US-53-001"; passes = $true }
            )
        } | ConvertTo-Json -Depth 3

        $prd = $prdJson | ConvertFrom-Json
        $story = $prd.userStories | Where-Object { $_.id -eq "US-99-999" }
        $story | Should -BeNullOrEmpty
    }

    It 'does not detect incomplete story as done' {
        $prdJson = @{
            userStories = @(
                @{ id = "US-53-010"; passes = $false; title = "In progress" }
            )
        } | ConvertTo-Json -Depth 3

        $prd = $prdJson | ConvertFrom-Json
        $story = $prd.userStories | Where-Object { $_.id -eq "US-53-010" }
        $story | Should -Not -BeNullOrEmpty
        $story.passes | Should -BeFalse
    }

    It 'grace period calculation works correctly' {
        $completionTime = (Get-Date).AddSeconds(-20)  # 20 seconds ago
        $gracePeriod = 15
        $sinceCompletion = ((Get-Date) - $completionTime).TotalSeconds
        $sinceCompletion | Should -BeGreaterOrEqual $gracePeriod
    }

    It 'grace period not expired for recent completion' {
        $completionTime = (Get-Date).AddSeconds(-2)  # 2 seconds ago
        $gracePeriod = 15
        $sinceCompletion = ((Get-Date) - $completionTime).TotalSeconds
        $sinceCompletion | Should -BeLessThan $gracePeriod
    }
}

Describe 'Config fallback behavior simulation' -Tag 'Unit', 'EarlyExit', 'Simulation' {
    It 'falls back to enabled=true when config section is null' {
        $earlyExitConfig = $null
        $earlyExitEnabled = if ($earlyExitConfig -and $null -ne $earlyExitConfig.enabled) { $earlyExitConfig.enabled } else { $true }
        $earlyExitEnabled | Should -BeTrue
    }

    It 'falls back to gracePeriod=15 when config section is null' {
        $earlyExitConfig = $null
        $storyCompletionGraceSec = if ($earlyExitConfig -and $earlyExitConfig.gracePeriodSeconds) { $earlyExitConfig.gracePeriodSeconds } else { 15 }
        $storyCompletionGraceSec | Should -Be 15
    }

    It 'reads enabled=false from config when explicitly set' {
        $earlyExitConfig = @{ enabled = $false; gracePeriodSeconds = 15 }
        $earlyExitEnabled = if ($earlyExitConfig -and $null -ne $earlyExitConfig.enabled) { $earlyExitConfig.enabled } else { $true }
        $earlyExitEnabled | Should -BeFalse
    }

    It 'reads custom grace period from config' {
        $earlyExitConfig = @{ enabled = $true; gracePeriodSeconds = 30 }
        $storyCompletionGraceSec = if ($earlyExitConfig -and $earlyExitConfig.gracePeriodSeconds) { $earlyExitConfig.gracePeriodSeconds } else { 15 }
        $storyCompletionGraceSec | Should -Be 30
    }
}

Describe 'Early exit does not fire without StoryId' -Tag 'Unit', 'EarlyExit', 'Simulation' {
    It 'empty StoryId prevents PRD check (simulates healing calls)' {
        $earlyExitEnabled = $true
        $prdUpdated = $true
        $StoryId = ""
        $storyCompletionDetected = $false

        $shouldCheck = $earlyExitEnabled -and $prdUpdated -and $StoryId -and -not $storyCompletionDetected
        $shouldCheck | Should -BeFalse
    }

    It 'non-empty StoryId allows PRD check' {
        $earlyExitEnabled = $true
        $prdUpdated = $true
        $StoryId = "US-53-010"
        $storyCompletionDetected = $false

        $shouldCheck = $earlyExitEnabled -and $prdUpdated -and $StoryId -and -not $storyCompletionDetected
        $shouldCheck | Should -BeTrue
    }
}
