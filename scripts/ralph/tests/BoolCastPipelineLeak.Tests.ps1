#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for [bool] cast pipeline leak prevention.
.DESCRIPTION
    Verifies that all Invoke-ClaudeForStory/FocusArea/Process call sites
    use [bool](... | Select-Object -Last 1) to prevent Object[] leaks
    into [bool] parameters like Update-SprintProgress -Success.

    Root cause: Functions called inside Resolve-ClaudeResult (Update-LearningDb,
    Save-StoryProgress, Update-TestBaseline) leaked return values into the
    PowerShell output pipeline. This caused Invoke-ClaudeProcess to return
    Object[] instead of a clean $true/$false, crashing Update-SprintProgress
    with "Cannot convert System.Object[] to System.Boolean".
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
}

Describe 'Bool cast on Invoke-ClaudeForStory callers in loops.ps1' -Tag 'Unit', 'PipelineLeak' {
    BeforeAll {
        $script:loopsSource = Get-Content (Join-Path $script:RalphDir 'lib\loops.ps1') -Raw
    }

    It 'every $success = Invoke-ClaudeForStory uses [bool] cast' {
        $assignments = [regex]::Matches($script:loopsSource, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeForStory.*$')
        $assignments.Count | Should -BeGreaterThan 0 -Because "there should be Invoke-ClaudeForStory calls"

        foreach ($m in $assignments) {
            $m.Value | Should -Match '\[bool\]' -Because "call site must cast to [bool]: $($m.Value.Trim())"
        }
    }

    It 'every $success = Invoke-ClaudeForStory pipes through Select-Object -Last 1' {
        $assignments = [regex]::Matches($script:loopsSource, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeForStory.*$')

        foreach ($m in $assignments) {
            $m.Value | Should -Match 'Select-Object\s+-Last\s+1' -Because "call site must use Select-Object -Last 1: $($m.Value.Trim())"
        }
    }

    It 'every $prdGenerated = Invoke-ClaudeForFocusArea uses [bool] cast' {
        $assignments = [regex]::Matches($script:loopsSource, '(?m)^\s*\$prdGenerated\s*=\s*.*Invoke-ClaudeForFocusArea.*$')
        $assignments.Count | Should -BeGreaterThan 0 -Because "there should be Invoke-ClaudeForFocusArea calls"

        foreach ($m in $assignments) {
            $m.Value | Should -Match '\[bool\]' -Because "call site must cast to [bool]: $($m.Value.Trim())"
        }
    }

    It 'every $prdGenerated = Invoke-ClaudeForFocusArea pipes through Select-Object -Last 1' {
        $assignments = [regex]::Matches($script:loopsSource, '(?m)^\s*\$prdGenerated\s*=\s*.*Invoke-ClaudeForFocusArea.*$')

        foreach ($m in $assignments) {
            $m.Value | Should -Match 'Select-Object\s+-Last\s+1' -Because "call site must use Select-Object -Last 1: $($m.Value.Trim())"
        }
    }

    It 'has exactly 6 Invoke-ClaudeForStory call sites in loops.ps1' {
        $matches = [regex]::Matches($script:loopsSource, 'Invoke-ClaudeForStory\s+-StoryId')
        $matches.Count | Should -Be 6
    }

    It 'has exactly 6 Invoke-ClaudeForFocusArea call sites' {
        $matches = [regex]::Matches($script:loopsSource, 'Invoke-ClaudeForFocusArea\s+-FocusAreaId')
        $matches.Count | Should -Be 6
    }
}

Describe 'Bool cast on Invoke-ClaudeProcess in ralph.ps1' -Tag 'Unit', 'PipelineLeak' {
    BeforeAll {
        $script:ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw
    }

    It 'Invoke-ClaudeForStory uses [bool] cast on Invoke-ClaudeProcess result' {
        $funcBody = [regex]::Match($script:ralphSource, 'function Invoke-ClaudeForStory[\s\S]*?(?=\nfunction\s|\z)').Value
        $funcBody | Should -Not -BeNullOrEmpty

        $processCall = [regex]::Match($funcBody, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeProcess.*$').Value
        $processCall | Should -Not -BeNullOrEmpty -Because "Invoke-ClaudeForStory should call Invoke-ClaudeProcess"
        $processCall | Should -Match '\[bool\]' -Because "result must be cast to [bool]"
        $processCall | Should -Match 'Select-Object\s+-Last\s+1' -Because "pipeline must extract last value"
    }
}

Describe 'Bool cast on Invoke-ClaudeProcess in claude.ps1' -Tag 'Unit', 'PipelineLeak' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'Invoke-ClaudeForFocusArea uses [bool] cast on Invoke-ClaudeProcess result' {
        $funcBody = [regex]::Match($script:claudeSource, 'function Invoke-ClaudeForFocusArea[\s\S]*?(?=\nfunction\s|\z)').Value
        $funcBody | Should -Not -BeNullOrEmpty

        $processCall = [regex]::Match($funcBody, '(?m)^\s*\$result\s*=\s*.*Invoke-ClaudeProcess.*$').Value
        $processCall | Should -Not -BeNullOrEmpty -Because "Invoke-ClaudeForFocusArea should call Invoke-ClaudeProcess"
        $processCall | Should -Match '\[bool\]' -Because "result must be cast to [bool]"
        $processCall | Should -Match 'Select-Object\s+-Last\s+1' -Because "pipeline must extract last value"
    }
}

Describe 'Output suppression in Resolve-ClaudeResult' -Tag 'Unit', 'PipelineLeak' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $script:resolveBody = [regex]::Match($script:claudeSource, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
    }

    It 'Update-TestBaseline is suppressed with $null =' {
        $script:resolveBody | Should -Match '\$null\s*=\s*Update-TestBaseline'
    }

    It 'Save-StoryProgress is suppressed with $null =' {
        $script:resolveBody | Should -Match '\$null\s*=\s*Save-StoryProgress'
    }

    It 'all Update-LearningDb calls are suppressed with $null =' {
        $calls = [regex]::Matches($script:resolveBody, '(?m)^.*Update-LearningDb.*$')
        $calls.Count | Should -BeGreaterThan 0 -Because "there should be Update-LearningDb calls"

        foreach ($call in $calls) {
            $line = $call.Value.Trim()
            if ($line -match '^\s*#') { continue }
            $line | Should -Match '\$null\s*=\s*Update-LearningDb' -Because "Update-LearningDb must be suppressed: $line"
        }
    }

    It 'all Log-StoryVerification calls are suppressed' {
        $calls = [regex]::Matches($script:resolveBody, '(?m)^.*Log-StoryVerification.*$')

        foreach ($call in $calls) {
            $line = $call.Value.Trim()
            if ($line -match '^\s*#') { continue }
            $line | Should -Match '(\$null|\$\w+)\s*=\s*Log-StoryVerification' -Because "Log-StoryVerification must be suppressed: $line"
        }
    }

    It 'Get-SprintTokenBudget is piped to Out-Null' {
        $script:resolveBody | Should -Match 'Get-SprintTokenBudget\s*\|\s*Out-Null'
    }
}
