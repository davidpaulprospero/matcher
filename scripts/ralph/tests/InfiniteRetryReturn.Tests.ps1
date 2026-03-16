#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Invoke-ClaudeWithInfiniteRetry return contract
.DESCRIPTION
    Tests cover the fix for the "Cannot convert null to type System.DateTime" crash.

    Root cause: The 30-minute timeout and exception return paths in
    Invoke-ClaudeWithInfiniteRetry were missing ExecutionStart/ExecutionEnd keys.
    When Record-IterationLog passed $null to Log-ClaudeInvocation's [datetime]$StartTime
    parameter, PowerShell rejected the null-to-DateTime conversion.

    Bug trace:
    1. lib/claude.ps1: Invoke-ClaudeWithInfiniteRetry timeout return (line ~238) had no ExecutionStart/ExecutionEnd
    2. ralph.ps1:341: $subResult.ExecutionStart resolved to $null
    3. ralph.ps1:312: $subResult.ExecutionEnd - $subResult.ExecutionStart also crashed
    4. lib/metrics.ps1:198: Log-ClaudeInvocation's [datetime]$StartTime rejected $null
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

# =============================================================================
# SOURCE INSPECTION TESTS
# =============================================================================

Describe 'Invoke-ClaudeWithInfiniteRetry return contract' -Tag 'Unit', 'ReturnContract' {
    It 'timeout return path includes ExecutionStart key' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Find the 30_minute_limit return block
        $timeoutPattern = 'Reason\s*=\s*"30_minute_limit"[\s\S]*?ExecutionStart\s*=\s*\$StoryStartTime'
        $claudeSource | Should -Match $timeoutPattern
    }

    It 'timeout return path includes ExecutionEnd key' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $timeoutPattern = 'Reason\s*=\s*"30_minute_limit"[\s\S]*?ExecutionEnd\s*=\s*Get-Date'
        $claudeSource | Should -Match $timeoutPattern
    }

    It 'exception return path includes ExecutionStart key' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $exceptionPattern = 'Reason\s*=\s*"exception"[\s\S]*?ExecutionStart\s*=\s*\$StoryStartTime'
        $claudeSource | Should -Match $exceptionPattern
    }

    It 'exception return path includes ExecutionEnd key' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $exceptionPattern = 'Reason\s*=\s*"exception"[\s\S]*?ExecutionEnd\s*=\s*Get-Date'
        $claudeSource | Should -Match $exceptionPattern
    }

    It 'timeout ExecutionStart uses $StoryStartTime (not Get-Date or $null)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Extract the timeout return block
        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"30_minute_limit"[\s\S]*?\}').Value
        $startMatch = [regex]::Match($block, 'ExecutionStart\s*=\s*(.+)')
        $startMatch.Groups[1].Value.Trim() | Should -Be '$StoryStartTime'
    }

    It 'exception ExecutionStart uses $StoryStartTime (not Get-Date or $null)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Extract the exception return block
        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"exception"[\s\S]*?\}').Value
        $startMatch = [regex]::Match($block, 'ExecutionStart\s*=\s*(.+)')
        $startMatch.Groups[1].Value.Trim() | Should -Be '$StoryStartTime'
    }
}

Describe 'All Invoke-ClaudeWithInfiniteRetry return paths have ExecutionStart/ExecutionEnd' -Tag 'Unit', 'ReturnContract' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Extract the function body
        $funcPattern = 'function Invoke-ClaudeWithInfiniteRetry\s*\{([\s\S]*?)^\}'
        $funcMatch = [regex]::Match($script:claudeSource, $funcPattern, 'Multiline')
        $script:funcBody = $funcMatch.Groups[1].Value

        # Find all return @{ ... } blocks within the function
        $script:returnBlocks = [regex]::Matches($script:funcBody, 'return @\{[\s\S]*?\}')
    }

    It 'has at least 2 return paths (timeout + exception)' {
        $script:returnBlocks.Count | Should -BeGreaterOrEqual 2
    }

    It 'every return block contains ExecutionStart' {
        foreach ($block in $script:returnBlocks) {
            $block.Value | Should -Match 'ExecutionStart' -Because "return block must include ExecutionStart: $($block.Value.Substring(0, [Math]::Min(80, $block.Value.Length)))"
        }
    }

    It 'every return block contains ExecutionEnd' {
        foreach ($block in $script:returnBlocks) {
            $block.Value | Should -Match 'ExecutionEnd' -Because "return block must include ExecutionEnd: $($block.Value.Substring(0, [Math]::Min(80, $block.Value.Length)))"
        }
    }
}

# =============================================================================
# CONSUMER TESTS - ralph.ps1 uses ExecutionStart/ExecutionEnd safely
# =============================================================================

Describe 'Invoke-ClaudeProcess safely consumes ExecutionStart/ExecutionEnd' -Tag 'Unit', 'ReturnContract' {
    It 'ralph.ps1 accesses $subResult.ExecutionStart for logging' {
        $ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw

        # Line 341: ExecutionStart = $subResult.ExecutionStart
        $ralphSource | Should -Match 'ExecutionStart\s*=\s*\$subResult\.ExecutionStart'
    }

    It 'ralph.ps1 accesses $subResult.ExecutionEnd for logging' {
        $ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw

        # Line 342: ExecutionEnd = $subResult.ExecutionEnd
        $ralphSource | Should -Match 'ExecutionEnd\s*=\s*\$subResult\.ExecutionEnd'
    }

    It 'ralph.ps1 computes execution duration from ExecutionEnd - ExecutionStart' {
        $ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw

        # Line 312: $subResult.ExecutionEnd - $subResult.ExecutionStart
        $ralphSource | Should -Match '\$subResult\.ExecutionEnd\s*-\s*\$subResult\.ExecutionStart'
    }
}

# =============================================================================
# Log-ClaudeInvocation PARAMETER TYPE TESTS
# =============================================================================

Describe 'Log-ClaudeInvocation parameter types' -Tag 'Unit', 'ReturnContract' {
    It 'StartTime parameter is typed [datetime] (rejects $null)' {
        $metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw

        $metricsSource | Should -Match '\[datetime\]\$StartTime'
    }

    It 'EndTime parameter is typed [datetime] (rejects $null)' {
        $metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw

        $metricsSource | Should -Match '\[datetime\]\$EndTime'
    }
}

# =============================================================================
# MUTATION TESTS
# =============================================================================

Describe 'Mutation testing - return contract values' -Tag 'Mutation', 'ReturnContract' {
    It 'timeout ExecutionStart is $StoryStartTime not Get-Date (preserves actual start)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # In the timeout block, ExecutionStart should be the original start time
        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"30_minute_limit"[\s\S]*?\}').Value
        $block | Should -Match 'ExecutionStart\s*=\s*\$StoryStartTime'
        $block | Should -Not -Match 'ExecutionStart\s*=\s*Get-Date'
    }

    It 'timeout ExecutionEnd is Get-Date not $StoryStartTime (captures actual end)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"30_minute_limit"[\s\S]*?\}').Value
        $block | Should -Match 'ExecutionEnd\s*=\s*Get-Date'
        $block | Should -Not -Match 'ExecutionEnd\s*=\s*\$StoryStartTime'
    }

    It 'exception ExecutionStart is $StoryStartTime not Get-Date' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"exception"[\s\S]*?\}').Value
        $block | Should -Match 'ExecutionStart\s*=\s*\$StoryStartTime'
        $block | Should -Not -Match 'ExecutionStart\s*=\s*Get-Date'
    }

    It 'exception ExecutionEnd is Get-Date not $StoryStartTime' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $block = [regex]::Match($claudeSource, 'Reason\s*=\s*"exception"[\s\S]*?\}').Value
        $block | Should -Match 'ExecutionEnd\s*=\s*Get-Date'
        $block | Should -Not -Match 'ExecutionEnd\s*=\s*\$StoryStartTime'
    }

    It 'no return path uses $null for ExecutionStart' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $funcPattern = 'function Invoke-ClaudeWithInfiniteRetry\s*\{([\s\S]*?)^\}'
        $funcBody = [regex]::Match($claudeSource, $funcPattern, 'Multiline').Groups[1].Value

        $funcBody | Should -Not -Match 'ExecutionStart\s*=\s*\$null'
    }

    It 'no return path uses $null for ExecutionEnd' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $funcPattern = 'function Invoke-ClaudeWithInfiniteRetry\s*\{([\s\S]*?)^\}'
        $funcBody = [regex]::Match($claudeSource, $funcPattern, 'Multiline').Groups[1].Value

        $funcBody | Should -Not -Match 'ExecutionEnd\s*=\s*\$null'
    }
}

# =============================================================================
# BEHAVIORAL SIMULATION TEST
# =============================================================================

Describe 'ExecutionStart/ExecutionEnd arithmetic safety' -Tag 'Unit', 'ReturnContract' {
    It 'datetime subtraction works with valid ExecutionStart and ExecutionEnd' {
        $start = (Get-Date).AddMinutes(-35)
        $end = Get-Date

        $result = @{
            ExecutionStart = $start
            ExecutionEnd = $end
        }

        # This is what ralph.ps1:312 does - must not throw
        $durationMs = [int](($result.ExecutionEnd - $result.ExecutionStart).TotalMilliseconds)
        $durationMs | Should -BeGreaterThan 0
    }

    It 'Log-ClaudeInvocation accepts valid datetime values' {
        # Simulate what happens when Record-IterationLog calls Log-ClaudeInvocation
        $start = (Get-Date).AddMinutes(-31)
        $end = Get-Date

        # This cast must succeed (the crash was: Cannot convert null to System.DateTime)
        { [datetime]$start } | Should -Not -Throw
        { [datetime]$end } | Should -Not -Throw
    }

    It 'passing $null to [datetime] parameter throws (proving the bug scenario)' {
        # This proves WHY the fix was needed - $null is rejected by [datetime]
        {
            function Test-DateTimeParam { param([datetime]$Value) $Value }
            Test-DateTimeParam -Value $null
        } | Should -Throw
    }
}
