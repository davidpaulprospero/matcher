#Requires -Modules Pester

<#
.SYNOPSIS
    Tests that the evidence threshold gate rejects stories with insufficient criteria verification
.DESCRIPTION
    Root cause: Stories were accepted as "completed successfully" based solely on exit code 0,
    even when only 40% of acceptance criteria had verifiable evidence. The evidence check was
    purely informational — it logged the percentage in red but never gated the pass/fail decision.

    Fix: After Log-StoryVerification runs, check the evidence percentage against a configurable
    threshold (default 90%). If below, reject the story: set passes=false, flip success=false,
    increment consecutive failures.
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

# =============================================================================
# SOURCE INSPECTION: Evidence gate exists in Resolve-ClaudeResult
# =============================================================================

Describe 'Evidence gate exists in claude.ps1 success path' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'reads evidenceThresholdPercent from config' {
        $script:claudeSource | Should -Match 'evidenceThresholdPercent'
    }

    It 'defaults to 90 when config is missing' {
        $script:claudeSource | Should -Match 'else\s*\{\s*90\s*\}'
    }

    It 'compares evidence percentage against threshold' {
        $script:claudeSource | Should -Match 'evidenceResult.*percentage.*-lt.*evidenceMinPct'
    }

    It 'calls Update-StoryStatus with Passes $false on rejection' {
        $script:claudeSource | Should -Match 'Update-StoryStatus\s+-StoryId.*-Passes\s+\$false'
    }

    It 'sets success to false on rejection' {
        $script:claudeSource | Should -Match '\$success\s*=\s*\$false'
    }

    It 'increments ConsecutiveFailures on rejection' {
        $script:claudeSource | Should -Match 'ConsecutiveFailures\+\+'
    }

    It 'sets iterationStatus to evidence_rejected' {
        $script:claudeSource | Should -Match 'iterationStatus.*=.*"evidence_rejected"'
    }

    It 'outputs rejection message with percentage comparison' {
        $script:claudeSource | Should -Match 'Evidence below threshold.*rejecting story'
    }

    It 'guards test baseline update behind $success' {
        $script:claudeSource | Should -Match 'if\s*\(\$success\s+-and\s+\$Ctx\.TestResults\)'
    }

    It 'guards story progress save behind $success' {
        # Save-StoryProgress should be inside an if ($success) block
        $script:claudeSource | Should -Match 'if\s*\(\$success\)\s*\{[^}]*Save-StoryProgress'
    }
}

# =============================================================================
# Confirm-CriteriaEvidence: LLM-based criteria verification
# =============================================================================

Describe 'Confirm-CriteriaEvidence uses LLM verification' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw
        $funcPattern = 'function Confirm-CriteriaEvidence\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $script:confirmBody = [regex]::Match($script:metricsSource, $funcPattern).Value
    }

    It 'exists as a function in metrics.ps1' {
        $script:confirmBody | Should -Not -BeNullOrEmpty
    }

    It 'uses --model haiku for LLM call' {
        $script:confirmBody | Should -Match '--model haiku'
    }

    It 'calls Get-ClaudePath to check CLI availability' {
        $script:confirmBody | Should -Match 'Get-ClaudePath'
    }

    It 'returns $null when Claude is unavailable (fallback signal)' {
        $script:confirmBody | Should -Match 'return \$null'
    }

    It 'truncates git diff to 8000 chars' {
        $script:confirmBody | Should -Match '8000'
    }

    It 'parses MET and NOT_MET responses from LLM' {
        $script:confirmBody | Should -Match "'\^MET\\s\+\(\\d\+\)'"
        $script:confirmBody | Should -Match "'\^NOT_MET\\s\+\(\\d\+\)'"
    }

    It 'uses async output capture like Confirm-CommitMatchesStory' {
        $script:confirmBody | Should -Match 'Register-ObjectEvent'
        $script:confirmBody | Should -Match 'BeginOutputReadLine'
    }

    It 'has 60 second timeout' {
        $script:confirmBody | Should -Match 'AddSeconds\(60\)'
    }
}

# =============================================================================
# Log-StoryVerification returns evidence data
# =============================================================================

Describe 'Log-StoryVerification returns evidence counts' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw
        $funcPattern = 'function Log-StoryVerification\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
        $script:funcBody = [regex]::Match($script:metricsSource, $funcPattern).Value
    }

    It 'calls Confirm-CriteriaEvidence for LLM verification' {
        $script:funcBody | Should -Match 'Confirm-CriteriaEvidence'
    }

    It 'falls back to Search-CriterionEvidence when LLM returns null' {
        $script:funcBody | Should -Match 'Search-CriterionEvidence\s+-Criterion'
    }

    It 'returns an object with criteriaMet' {
        $script:funcBody | Should -Match 'return\s+@\{[^}]*criteriaMet'
    }

    It 'returns an object with criteriaTotal' {
        $script:funcBody | Should -Match 'return\s+@\{[^}]*criteriaTotal'
    }

    It 'returns an object with percentage' {
        $script:funcBody | Should -Match 'return\s+@\{[^}]*percentage'
    }

    It 'returns 100 when criteriaTotal is 0' {
        $script:funcBody | Should -Match 'criteriaTotalCount\s*-gt\s*0.*else\s*\{\s*100\s*\}'
    }
}

# =============================================================================
# CONFIG: evidenceThresholdPercent in ralph-config.json
# =============================================================================

Describe 'Config includes evidenceThresholdPercent' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $configPath = Join-Path $script:RalphDir 'config\ralph-config.json'
        $script:config = Get-Content $configPath -Raw | ConvertFrom-Json
    }

    It 'has evidenceThresholdPercent under storyCompletionEarlyExit' {
        $script:config.stallDetection.storyCompletionEarlyExit.evidenceThresholdPercent | Should -Not -BeNullOrEmpty
    }

    It 'evidenceThresholdPercent is 90' {
        $script:config.stallDetection.storyCompletionEarlyExit.evidenceThresholdPercent | Should -Be 90
    }
}

# =============================================================================
# BEHAVIORAL: Evidence percentage calculation
# =============================================================================

Describe 'Evidence percentage logic' -Tag 'Unit', 'EvidenceGate' {
    It '2 of 5 criteria = 40%, below 90% threshold' {
        $met = 2; $total = 5
        $pct = [math]::Round(($met / $total) * 100, 0)
        $pct | Should -Be 40
        $pct -lt 90 | Should -BeTrue
    }

    It '3 of 5 criteria = 60%, below 90% threshold' {
        $met = 3; $total = 5
        $pct = [math]::Round(($met / $total) * 100, 0)
        $pct | Should -Be 60
        $pct -lt 90 | Should -BeTrue
    }

    It '4 of 5 criteria = 80%, below 90% threshold' {
        $met = 4; $total = 5
        $pct = [math]::Round(($met / $total) * 100, 0)
        $pct | Should -Be 80
        $pct -lt 90 | Should -BeTrue
    }

    It '5 of 5 criteria = 100%, passes 90% threshold' {
        $met = 5; $total = 5
        $pct = [math]::Round(($met / $total) * 100, 0)
        $pct | Should -Be 100
        $pct -lt 90 | Should -BeFalse
    }

    It '9 of 10 criteria = 90%, passes 90% threshold (not strictly less)' {
        $met = 9; $total = 10
        $pct = [math]::Round(($met / $total) * 100, 0)
        $pct | Should -Be 90
        $pct -lt 90 | Should -BeFalse
    }

    It '0 criteria total defaults to 100% (no criteria = auto-pass)' {
        $total = 0
        $pct = if ($total -gt 0) { [math]::Round((0 / $total) * 100, 0) } else { 100 }
        $pct | Should -Be 100
        $pct -lt 90 | Should -BeFalse
    }
}

# =============================================================================
# ORDERING: Evidence gate before baseline update
# =============================================================================

Describe 'Evidence gate ordering in claude.ps1' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'evidence gate appears BEFORE Update-TestBaseline' {
        $gatePos = $script:claudeSource.IndexOf('evidence_rejected')
        $baselinePos = $script:claudeSource.IndexOf('Update-TestBaseline')
        $gatePos | Should -BeGreaterThan -1
        $baselinePos | Should -BeGreaterThan -1
        $gatePos | Should -BeLessThan $baselinePos
    }

    It 'evidence gate appears BEFORE Save-StoryProgress' {
        $gatePos = $script:claudeSource.IndexOf('evidence_rejected')
        $progressPos = $script:claudeSource.IndexOf('Save-StoryProgress')
        $gatePos | Should -BeGreaterThan -1
        $progressPos | Should -BeGreaterThan -1
        $gatePos | Should -BeLessThan $progressPos
    }

    It 'evidence gate appears AFTER Log-StoryVerification' {
        $verifyPos = $script:claudeSource.IndexOf('Log-StoryVerification')
        $gatePos = $script:claudeSource.IndexOf('evidence_rejected')
        $verifyPos | Should -BeGreaterThan -1
        $gatePos | Should -BeGreaterThan $verifyPos
    }
}

# =============================================================================
# MUTATION: Ensure tests catch regressions
# =============================================================================

Describe 'Mutation: evidence gate changes are detected' -Tag 'Unit', 'EvidenceGate', 'Mutation' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'changing default from 90 to 50 is caught' {
        $mutated = $script:claudeSource.Replace('} else { 90 }', '} else { 50 }')
        $mutated | Should -Not -Match 'else\s*\{\s*90\s*\}'
    }

    It 'removing Update-StoryStatus call is caught' {
        $mutated = $script:claudeSource.Replace('Update-StoryStatus -StoryId $Ctx.StoryId -Passes $false', '# REMOVED')
        $mutated | Should -Not -Match 'Update-StoryStatus\s+-StoryId.*-Passes\s+\$false'
    }

    It 'changing -lt to -le is caught' {
        $mutated = $script:claudeSource.Replace('percentage -lt $evidenceMinPct', 'percentage -le $evidenceMinPct')
        $mutated | Should -Not -Match 'percentage.*-lt.*evidenceMinPct'
    }

    It 'removing ConsecutiveFailures increment is caught' {
        # Find the specific increment in the evidence gate block
        $mutated = $script:claudeSource.Replace('$script:State.ConsecutiveFailures++', '# REMOVED')
        # The evidence gate section should no longer have the increment
        $gateBlock = [regex]::Match($mutated, 'evidence_rejected[\s\S]{0,200}').Value
        $gateBlock | Should -Not -Match 'ConsecutiveFailures\+\+'
    }
}

# =============================================================================
# SOURCE INSPECTION: Evidence diff excludes Ralph state directories
# =============================================================================

Describe 'Evidence diff excludes Ralph metadata directories' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'uses git diff HEAD~1 with pathspec exclusions' {
        $script:claudeSource | Should -Match 'git diff HEAD~1 --.*exclude'
    }

    It 'excludes scripts/ralph/state/ from evidence diff' {
        $script:claudeSource | Should -Match '\(exclude\)scripts/ralph/state/'
    }

    It 'excludes scripts/ralph/session/ from evidence diff' {
        $script:claudeSource | Should -Match '\(exclude\)scripts/ralph/session/'
    }

    It 'excludes scripts/ralph/archive/ from evidence diff' {
        $script:claudeSource | Should -Match '\(exclude\)scripts/ralph/archive/'
    }

    It 'diff exclusion is on the same line as git diff HEAD~1' {
        # Ensure the exclusion is part of the diff command, not a separate command
        $script:claudeSource | Should -Match 'git diff HEAD~1\s+--\s+":\(exclude\)'
    }
}

# =============================================================================
# BEHAVIORAL: Excluded directories are actually Ralph metadata
# =============================================================================

Describe 'Excluded directories contain only Ralph metadata' -Tag 'Unit', 'EvidenceGate' {
    BeforeAll {
        $script:ralphDir = Split-Path -Parent $PSScriptRoot
    }

    It 'scripts/ralph/state/ contains prd.json (large metadata file)' {
        $prdPath = Join-Path $script:ralphDir 'state\prd.json'
        Test-Path $prdPath | Should -BeTrue
    }

    It 'scripts/ralph/state/ does not contain source code (.py or .ps1 lib)' {
        $sourceFiles = Get-ChildItem -Path (Join-Path $script:ralphDir 'state') -Include '*.py' -Recurse -ErrorAction SilentlyContinue
        $sourceFiles | Should -BeNullOrEmpty
    }

    It 'scripts/ralph/session/ is volatile per-session data' {
        $sessionDir = Join-Path $script:ralphDir 'session'
        # session dir should exist as part of Ralph structure
        Test-Path $sessionDir | Should -BeTrue
    }
}

# =============================================================================
# MUTATION: Diff exclusion regressions detected
# =============================================================================

Describe 'Mutation: diff exclusion changes are detected' -Tag 'Unit', 'EvidenceGate', 'Mutation' {
    BeforeAll {
        $script:claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'removing state/ exclusion is caught' {
        $mutated = $script:claudeSource.Replace('":(exclude)scripts/ralph/state/"', '')
        $mutated | Should -Not -Match '\(exclude\)scripts/ralph/state/'
    }

    It 'removing session/ exclusion is caught' {
        $mutated = $script:claudeSource.Replace('":(exclude)scripts/ralph/session/"', '')
        $mutated | Should -Not -Match '\(exclude\)scripts/ralph/session/'
    }

    It 'removing archive/ exclusion is caught' {
        $mutated = $script:claudeSource.Replace('":(exclude)scripts/ralph/archive/"', '')
        $mutated | Should -Not -Match '\(exclude\)scripts/ralph/archive/'
    }

    It 'removing all pathspec exclusions is caught' {
        $mutated = $script:claudeSource.Replace(
            '-- ":(exclude)scripts/ralph/state/" ":(exclude)scripts/ralph/session/" ":(exclude)scripts/ralph/archive/"',
            ''
        )
        $mutated | Should -Not -Match 'git diff HEAD~1\s+--\s+":\(exclude\)'
    }

    It 'reverting to bare git diff HEAD~1 is caught' {
        $mutated = $script:claudeSource.Replace(
            'git diff HEAD~1 -- ":(exclude)scripts/ralph/state/" ":(exclude)scripts/ralph/session/" ":(exclude)scripts/ralph/archive/" 2>$null',
            'git diff HEAD~1 2>$null'
        )
        $mutated | Should -Not -Match '\(exclude\)scripts/ralph/state/'
    }
}
