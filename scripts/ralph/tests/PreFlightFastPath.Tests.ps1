# =============================================================================
# PreFlightFastPath.Tests.ps1
# Tests for pre-implemented story auto-completion fast paths
# Covers: metrics.ps1 Test-StoryAlreadyCommitted, ralph.ps1 Invoke-ClaudeForStory,
#         quality.ps1 Invoke-BatchPreFlight
# =============================================================================

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:metricsSource = Get-Content (Join-Path $script:RalphDir 'lib\metrics.ps1') -Raw
    $script:ralphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw
    $script:qualitySource = Get-Content (Join-Path $script:RalphDir 'lib\quality.ps1') -Raw
}

# =============================================================================
# Test-StoryAlreadyCommitted fast path (metrics.ps1)
# =============================================================================

Describe 'Test-StoryAlreadyCommitted pre-implemented fast path' -Tag 'Unit', 'PreFlight' {

    It 'has fast path for "Mark as complete" commits' {
        $script:metricsSource | Should -Match "commitLine.*-match.*'Mark as complete\|pre-implemented'"
    }

    It 'fast path returns $true (auto-complete)' {
        # Extract the fast-path block
        $block = [regex]::Match($script:metricsSource, "if \(\`$commitLine -match 'Mark as complete\|pre-implemented'\)[\s\S]{0,300}?return \`$true").Value
        $block | Should -Not -BeNullOrEmpty -Because "fast path should return true"
        $block | Should -Match 'return \$true'
    }

    It 'fast path appears BEFORE LLM verification' {
        $fastPathPos = $script:metricsSource.IndexOf("commitLine -match 'Mark as complete|pre-implemented'")
        $llmVerifyPos = $script:metricsSource.IndexOf('Confirm-CommitMatchesStory -Candidates')
        $fastPathPos | Should -BeGreaterThan -1 -Because "fast path must exist"
        $llmVerifyPos | Should -BeGreaterThan -1 -Because "LLM verify must exist"
        $fastPathPos | Should -BeLessThan $llmVerifyPos -Because "fast path must come before LLM call"
    }

    It 'fast path matches both "Mark as complete" and "pre-implemented"' {
        # Test regex against sample commit messages
        $regex = 'Mark as complete|pre-implemented'
        'chore: [US-74-006] Mark as complete — pre-implemented in US-72-007' | Should -Match $regex
        'fix: [US-74-003] pre-implemented via US-70-006' | Should -Match $regex
        'feat: [US-74-007] Add new scoring function' | Should -Not -Match $regex
    }

    It 'does NOT fast-path regular feature commits' {
        $regex = 'Mark as complete|pre-implemented'
        'feat: [US-74-007] Implement voiceover listicle detection' | Should -Not -Match $regex
        'fix: [US-74-008] Fix config loading edge case' | Should -Not -Match $regex
        'test: [US-74-009] Add cache integration tests' | Should -Not -Match $regex
    }
}

# =============================================================================
# Invoke-ClaudeForStory pre-implemented notes fast path (ralph.ps1)
# =============================================================================

Describe 'Invoke-ClaudeForStory pre-implemented notes fast path' -Tag 'Unit', 'PreFlight' {

    It 'checks story notes for pre-implemented pattern' {
        $script:ralphSource | Should -Match "storyNotes.*-match.*'\[Pp\]re-implemented'"
    }

    It 'calls Complete-StoryAutomatically with pre-implemented-in-notes reason' {
        $block = [regex]::Match($script:ralphSource, "storyNotes.*-match.*'\[Pp\]re-implemented'[\s\S]{0,800}?return \`$true").Value
        $block | Should -Not -BeNullOrEmpty -Because "must have fast path block"
        $block | Should -Match 'Complete-StoryAutomatically'
        $block | Should -Match 'pre-implemented-in-notes'
    }

    It 'adds to AutoCompletedStories tracking' {
        $block = [regex]::Match($script:ralphSource, "storyNotes.*-match.*'\[Pp\]re-implemented'[\s\S]{0,500}?AutoCompletedStories").Value
        $block | Should -Not -BeNullOrEmpty -Because "must track in AutoCompletedStories"
    }

    It 'returns $true after auto-completing' {
        $block = [regex]::Match($script:ralphSource, "storyNotes.*-match.*'\[Pp\]re-implemented'[\s\S]{0,500}?return \`$true").Value
        $block | Should -Not -BeNullOrEmpty -Because "must return true"
    }

    It 'pre-implemented check appears BEFORE Test-StoryAlreadyCommitted call' {
        $notesCheckPos = $script:ralphSource.IndexOf("storyNotes -match '[Pp]re-implemented'")
        $gitCheckPos = $script:ralphSource.IndexOf('Test-StoryAlreadyCommitted -StoryId')
        $notesCheckPos | Should -BeGreaterThan -1 -Because "notes check must exist"
        $gitCheckPos | Should -BeGreaterThan -1 -Because "git check must exist"
        $notesCheckPos | Should -BeLessThan $gitCheckPos -Because "notes check should skip git/LLM entirely"
    }

    It 'handles null notes gracefully (empty string fallback)' {
        $script:ralphSource | Should -Match '\$storyNotes\s*=\s*if\s*\(\$storyObj\.notes\)'
    }
}

# =============================================================================
# Invoke-BatchPreFlight Phase 0.5 (quality.ps1)
# =============================================================================

Describe 'Invoke-BatchPreFlight Phase 0.5 pre-implemented notes' -Tag 'Unit', 'PreFlight' {

    It 'has Phase 0.5 comment marker' {
        $script:qualitySource | Should -Match 'Phase 0\.5.*[Ff]ast path'
    }

    It 'checks story notes for pre-implemented pattern' {
        # Within the Invoke-BatchPreFlight function
        $funcStart = $script:qualitySource.IndexOf('function Invoke-BatchPreFlight')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(3000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match "sNotes.*-match.*'\[Pp\]re-implemented'"
    }

    It 'calls Complete-StoryAutomatically for pre-implemented stories' {
        $funcStart = $script:qualitySource.IndexOf('function Invoke-BatchPreFlight')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(3000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match 'Complete-StoryAutomatically.*pre-implemented-in-notes'
    }

    It 'refreshes incomplete stories list after pre-implemented completions' {
        $funcStart = $script:qualitySource.IndexOf('function Invoke-BatchPreFlight')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(3000, $script:qualitySource.Length - $funcStart))
        # Should re-read PRD and re-filter after completing pre-implemented stories
        $phase05Match = [regex]::Match($funcBlock, 'preImplCompleted.*\-gt 0[\s\S]{0,500}?Get-Sprint')
        $phase05Match.Success | Should -BeTrue -Because "must refresh PRD after auto-completing"
    }

    It 'returns early if all stories complete after Phase 0.5' {
        $funcStart = $script:qualitySource.IndexOf('function Invoke-BatchPreFlight')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(3000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match 'all stories complete after pre-implemented check'
    }
}

# =============================================================================
# Invoke-BatchPreFlight Phase 1.5 (quality.ps1)
# =============================================================================

Describe 'Invoke-BatchPreFlight Phase 1.5 pre-implemented commits' -Tag 'Unit', 'PreFlight' {

    It 'has Phase 1.5 comment marker' {
        $script:qualitySource | Should -Match 'Phase 1\.5.*[Ff]ast path'
    }

    It 'splits candidates into fast-path and LLM-verification buckets' {
        $script:qualitySource | Should -Match '\$fastCompleted'
        $script:qualitySource | Should -Match '\$needsLlmVerification'
    }

    It 'fast-path regex matches "Mark as complete|pre-implemented" on commitMsg' {
        $funcStart = $script:qualitySource.IndexOf('Phase 1.5')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(2000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match "commitMsg.*-match.*'Mark as complete\|pre-implemented'"
    }

    It 'only calls Confirm-CommitMatchesStory for needsLlmVerification bucket' {
        $funcStart = $script:qualitySource.IndexOf('Phase 1.5')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(2000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match 'needsLlmVerification\.Count.*-gt 0'
        $funcBlock | Should -Match 'Confirm-CommitMatchesStory.*-Candidates.*needsLlmVerification'
    }

    It 'sets $matchedIds to empty array when no LLM verification needed' {
        $funcStart = $script:qualitySource.IndexOf('Phase 1.5')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(2000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match '\$matchedIds\s*=\s*@\(\)'
    }

    It 'calls Complete-StoryAutomatically for fast-completed candidates' {
        $funcStart = $script:qualitySource.IndexOf('Phase 1.5')
        $funcBlock = $script:qualitySource.Substring($funcStart, [Math]::Min(2000, $script:qualitySource.Length - $funcStart))
        $funcBlock | Should -Match 'Complete-StoryAutomatically.*pre-implemented-commit'
    }
}

# =============================================================================
# Cross-cutting: all three fast paths use consistent patterns
# =============================================================================

Describe 'Pre-flight fast path consistency' -Tag 'Unit', 'PreFlight' {

    It 'all three files use the same commit regex pattern' {
        # metrics.ps1 and quality.ps1 should both match "Mark as complete|pre-implemented"
        $metricsRegex = [regex]::Match($script:metricsSource, "'(Mark as complete\|pre-implemented)'").Groups[1].Value
        $qualityRegex = [regex]::Match($script:qualitySource, "'(Mark as complete\|pre-implemented)'").Groups[1].Value
        $metricsRegex | Should -BeExactly 'Mark as complete|pre-implemented'
        $qualityRegex | Should -BeExactly 'Mark as complete|pre-implemented'
    }

    It 'notes regex pattern is consistent between ralph.ps1 and quality.ps1' {
        $ralphRegex = [regex]::Match($script:ralphSource, "'(\[Pp\]re-implemented)'").Groups[1].Value
        $qualityRegex = [regex]::Match($script:qualitySource, "'(\[Pp\]re-implemented)'").Groups[1].Value
        $ralphRegex | Should -BeExactly '[Pp]re-implemented'
        $qualityRegex | Should -BeExactly '[Pp]re-implemented'
    }
}
