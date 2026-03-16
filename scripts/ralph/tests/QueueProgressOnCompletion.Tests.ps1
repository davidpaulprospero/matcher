#Requires -Modules Pester

<#
.SYNOPSIS
    Tests that ALL loop implementations call Update-QueueProgress on sprint completion
.DESCRIPTION
    Root cause: The TrueAuto and AdaptiveOvernight loops were missing
    Update-QueueProgress calls after sprint completion. This caused completed
    focus areas to remain marked as incomplete in queue.json, leading to
    infinite re-sprints on the same area instead of advancing through the queue.

    Fix: Added Update-QueueProgress -AreaId ... -Silent calls to all loops
    that handle sprint completion, matching the existing pattern in
    StandardLoop and RalphsChoiceLoop.
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

# =============================================================================
# SOURCE INSPECTION: Every loop with sprint completion MUST call Update-QueueProgress
# =============================================================================

Describe 'All loop functions call Update-QueueProgress on sprint completion' -Tag 'Unit', 'QueueProgress' {
    BeforeAll {
        $script:loopsSource = Get-Content (Join-Path $script:RalphDir 'lib\loops.ps1') -Raw
    }

    It 'Start-InterviewQueueLoop calls Update-QueueProgress' {
        # The interview queue loop (line ~182) already has the call
        $funcPattern = 'function Start-InterviewQueueLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }

    It 'Start-TrueAutoLoop calls Update-QueueProgress' {
        $funcPattern = 'function Start-TrueAutoLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }

    It 'Start-StandardLoop calls Update-QueueProgress' {
        $funcPattern = 'function Start-StandardLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }

    It 'Start-RalphsChoiceLoop calls Update-QueueProgress' {
        $funcPattern = 'function Start-RalphsChoiceLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }

    It 'Start-RalphsChoiceAutoLoop calls Update-QueueProgress' {
        $funcPattern = 'function Start-RalphsChoiceAutoLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }

    It 'Start-AdaptiveOvernightLoop calls Update-QueueProgress' {
        $funcPattern = 'function Start-AdaptiveOvernightLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress'
    }
}

Describe 'Update-QueueProgress is called BEFORE new sprint generation in TrueAuto' -Tag 'Unit', 'QueueProgress' {
    BeforeAll {
        $script:loopsSource = Get-Content (Join-Path $script:RalphDir 'lib\loops.ps1') -Raw
        $funcPattern = 'function Start-TrueAutoLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $script:trueAutoBody = [regex]::Match($script:loopsSource, $funcPattern).Value
    }

    It 'Update-QueueProgress appears BEFORE "TrueAuto will generate new stories"' {
        $queuePos = $script:trueAutoBody.IndexOf('Update-QueueProgress')
        $generatePos = $script:trueAutoBody.IndexOf('TrueAuto will generate new stories')
        $queuePos | Should -BeGreaterThan -1
        $generatePos | Should -BeGreaterThan -1
        $queuePos | Should -BeLessThan $generatePos
    }

    It 'Update-QueueProgress appears AFTER Save-SprintArchive' {
        $archivePos = $script:trueAutoBody.IndexOf('Save-SprintArchive')
        $queuePos = $script:trueAutoBody.IndexOf('Update-QueueProgress')
        $archivePos | Should -BeGreaterThan -1
        $queuePos | Should -BeGreaterThan $archivePos
    }

    It 'uses -Silent flag' {
        $script:trueAutoBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea\s+-Silent'
    }

    It 'checks focusArea is not null before calling' {
        $script:trueAutoBody | Should -Match 'if\s*\(\$status\.focusArea\)\s*\{[\s\S]*?Update-QueueProgress'
    }
}

Describe 'Update-QueueProgress is called BEFORE new sprint in AdaptiveOvernight' -Tag 'Unit', 'QueueProgress' {
    BeforeAll {
        $script:loopsSource = Get-Content (Join-Path $script:RalphDir 'lib\loops.ps1') -Raw
        $funcPattern = 'function Start-AdaptiveOvernightLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $script:overnightBody = [regex]::Match($script:loopsSource, $funcPattern).Value
    }

    It 'Update-QueueProgress appears BEFORE the sprint-complete needsNewSprint = true' {
        $queuePos = $script:overnightBody.IndexOf('Update-QueueProgress')
        # Find the $needsNewSprint = $true that follows Save-SprintArchive (sprint-complete path)
        # not the earlier one (no-sprint-exists path)
        $archivePos = $script:overnightBody.IndexOf('Save-SprintArchive')
        $newSprintPos = $script:overnightBody.IndexOf('$needsNewSprint = $true', $archivePos)
        $queuePos | Should -BeGreaterThan -1
        $newSprintPos | Should -BeGreaterThan -1
        $queuePos | Should -BeLessThan $newSprintPos
    }

    It 'Update-QueueProgress appears AFTER Save-SprintArchive' {
        $archivePos = $script:overnightBody.IndexOf('Save-SprintArchive')
        $queuePos = $script:overnightBody.IndexOf('Update-QueueProgress')
        $archivePos | Should -BeGreaterThan -1
        $queuePos | Should -BeGreaterThan $archivePos
    }

    It 'uses -Silent flag' {
        $script:overnightBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$currentArea\s+-Silent'
    }
}

# =============================================================================
# MUTATION TESTS
# =============================================================================

Describe 'Mutation: removing Update-QueueProgress from any loop is detected' -Tag 'Unit', 'QueueProgress', 'Mutation' {
    BeforeAll {
        $script:loopsSource = Get-Content (Join-Path $script:RalphDir 'lib\loops.ps1') -Raw
    }

    It 'TrueAutoLoop uses correct area variable ($status.focusArea)' {
        $funcPattern = 'function Start-TrueAutoLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea'
    }

    It 'StandardLoop uses correct area variable ($status.focusArea)' {
        $funcPattern = 'function Start-StandardLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea'
    }

    It 'RalphsChoiceLoop uses correct area variable ($status.focusArea)' {
        $funcPattern = 'function Start-RalphsChoiceLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$status\.focusArea'
    }

    It 'AdaptiveOvernightLoop uses correct area variable ($currentArea)' {
        $funcPattern = 'function Start-AdaptiveOvernightLoop\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $funcBody = [regex]::Match($script:loopsSource, $funcPattern).Value
        $funcBody | Should -Match 'Update-QueueProgress\s+-AreaId\s+\$currentArea'
    }
}
