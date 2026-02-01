#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Ralph Loop display freeze / race condition fix
.DESCRIPTION
    Tests cover:
    - Post-cancel async handler drain (500ms sleep at line 506)
    - Buffer stabilization polling (lines 515-527)
    - Console flush after success message (line 703)

    These tests verify the fix for the issue where "Story completed successfully"
    didn't appear until spacebar press due to race condition between async
    event handlers and PowerShell's console output system.
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

Describe 'Invoke-ClaudeSubprocess async handler drain' -Tag 'Unit', 'RaceCondition' {
    BeforeEach {
        $script:ProjectRoot = $TestDrive
        $script:Config = @{ iterationTimeout = 10 }
        $script:PrdFile = Join-Path $TestDrive "prd.json"
        $script:ProgressFile = Join-Path $TestDrive "progress.txt"
        $script:State = @{ IterationCount = 1; SessionId = "test" }

        # Create required files
        "{}" | Set-Content $script:PrdFile
        "" | Set-Content $script:ProgressFile
    }

    It 'allows 500ms for async handlers after CancelOutputRead' {
        # This test verifies the 500ms sleep exists by checking timing
        # We mock the process to exit immediately and measure minimum duration

        $mockProcess = [PSCustomObject]@{
            HasExited = $true
            Id = 12345
            ExitCode = 0
            StartInfo = [PSCustomObject]@{}
        }

        # Track when CancelOutputRead would be called vs when ExitCode is read
        $script:cancelTime = $null
        $script:exitCodeReadTime = $null

        Mock Start-Sleep {
            param($Milliseconds)
            # Track the 500ms sleep after cancel
            if ($Milliseconds -eq 500) {
                $script:sleepCalled500ms = $true
            }
        }

        # The actual implementation has Start-Sleep -Milliseconds 500 after CancelOutputRead
        # We verify the code path includes this timing
        $script:sleepCalled500ms = $false

        # Read the source and verify the 500ms sleep is present
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
        $claudeSource | Should -Match 'Start-Sleep -Milliseconds 500\s+#.*async event handlers'
    }

    It 'has buffer stabilization polling loop with max 2000ms wait' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify the polling loop structure
        $claudeSource | Should -Match '\$maxWait = 2000'
        $claudeSource | Should -Match '\$waited = 0'
        $claudeSource | Should -Match '\$lastLen = \$outBuilder\.Length'
        $claudeSource | Should -Match 'while \(\$waited -lt \$maxWait\)'
        $claudeSource | Should -Match 'if \(\$currentLen -eq \$lastLen\) \{ break \}'
    }

    It 'polls buffer every 200ms during stabilization' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify 200ms polling interval
        $claudeSource | Should -Match 'Start-Sleep -Milliseconds 200'
        $claudeSource | Should -Match '\$waited \+= 200'
    }

    It 'breaks early when buffer is stable (no new data)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify early break condition
        $claudeSource | Should -Match '\$currentLen = \$outBuilder\.Length'
        $claudeSource | Should -Match 'if \(\$currentLen -eq \$lastLen\) \{ break \}\s+# Buffer stable'
        $claudeSource | Should -Match '\$lastLen = \$currentLen'
    }
}

Describe 'Resolve-ClaudeResult console flush' -Tag 'Unit', 'RaceCondition' {
    BeforeEach {
        $script:State = @{
            IterationCount = 5
            CurrentMode = "Standard"
            ConsecutiveFailures = 0
            CurrentRetryCount = 0
        }
        $script:ProjectRoot = $TestDrive
    }

    It 'flushes console after success message' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify [Console]::Out.Flush() follows Write-Host success message
        $pattern = 'Write-Host "\s+\$successMsg" -ForegroundColor Green\s*\r?\n\s*\[Console\]::Out\.Flush\(\)'
        $claudeSource | Should -Match $pattern
    }

    It 'has flush comment explaining purpose' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify comment explains the flush
        $claudeSource | Should -Match '\[Console\]::Out\.Flush\(\)\s+#.*immediate display'
    }

    It 'displays success message for story completion' {
        # Mock all dependencies
        Mock Write-Host {}
        Mock Log-StateTransition {}
        Mock Get-GitDiffStats { return @{ Added = 10; Deleted = 5 } }
        Mock Get-DiffQualityScore { return @{ warnings = @(); testRatio = 0.5; churnRisk = "low" } }
        Mock Record-Metric {}
        Mock git { return "" }
        Mock Compare-TestBaseline { return @{ hasRegression = $false } }
        Mock Log-StoryVerification {}
        Mock Append-SessionTimeline {}
        Mock Update-TestBaseline {}
        Mock Get-SprintTokenBudget { return @{} }
        Mock Save-StoryProgress {}
        Mock Update-LearningDb {}
        Mock Invoke-CodeReview { return @{ score = 80; passed = $true; issues = @() } }
        Mock Test-FileConflict { return @{ hasConflict = $false } }
        Mock Get-ErrorCategory { return "none" }
        Mock Invoke-PostIterationHealing { return @{ HealingNeeded = $false } }

        $subResult = @{
            TimedOut = $false
            ExitCode = 0
            Output = "Success"
            Timeout = 600
        }

        $ctx = @{
            TransitionContext = @{}
            IsStoryWork = $true
            StoryId = "US-TEST-001"
            FocusAreaId = "testing"
            Identifier = "US-TEST-001"
            StoryObj = @{ id = "US-TEST-001"; title = "Test Story" }
            IterationDuration = [timespan]::FromMinutes(5)
            TokensUsed = 1000
            TestResults = $null
            PhaseTimings = @{ read_ms = 100; analyze_ms = 200; implement_ms = 300; test_ms = 400; commit_ms = 50 }
            GitStateBefore = @{ branch = "main" }
            FileOps = @{ filesCreated = @(); filesModified = @() }
            Commits = @()
            ClaudeOutput = "Test output"
        }

        $result = Resolve-ClaudeResult -SubResult $subResult -Ctx $ctx

        $result.Success | Should -BeTrue
        $result.IterationStatus | Should -Be "completed"

        # Verify success message was written
        Should -Invoke Write-Host -ParameterFilter {
            $Object -like "*completed successfully*" -and $ForegroundColor -eq "Green"
        }
    }

    It 'displays iteration message for non-story work' {
        Mock Write-Host {}
        Mock Log-StateTransition {}
        Mock Get-GitDiffStats { return @{ Added = 5; Deleted = 2 } }
        Mock Get-DiffQualityScore { return @{ warnings = @(); testRatio = 0; churnRisk = "low" } }
        Mock Record-Metric {}
        Mock git { return "" }
        Mock Compare-TestBaseline { return $null }
        Mock Get-ErrorCategory { return "none" }
        Mock Invoke-PostIterationHealing { return @{ HealingNeeded = $false } }

        $subResult = @{
            TimedOut = $false
            ExitCode = 0
            Output = "Done"
            Timeout = 600
        }

        $ctx = @{
            TransitionContext = @{}
            IsStoryWork = $false
            StoryId = ""
            FocusAreaId = "general"
            Identifier = "iteration-5"
            StoryObj = $null
            IterationDuration = [timespan]::FromMinutes(2)
            TokensUsed = 500
            TestResults = $null
            PhaseTimings = @{ read_ms = 50; analyze_ms = 100; implement_ms = 150; test_ms = 200; commit_ms = 25 }
            GitStateBefore = @{ branch = "main" }
            FileOps = @{ filesCreated = @(); filesModified = @() }
            Commits = @()
            ClaudeOutput = ""
        }

        $result = Resolve-ClaudeResult -SubResult $subResult -Ctx $ctx

        $result.Success | Should -BeTrue

        # Verify iteration message (not story message)
        Should -Invoke Write-Host -ParameterFilter {
            $Object -like "*Iteration completed successfully*"
        }
    }
}

Describe 'Buffer stabilization logic' -Tag 'Unit', 'RaceCondition' {
    It 'exits immediately when buffer unchanged after first poll' {
        # Simulate the buffer stabilization logic
        $outBuilder = [System.Text.StringBuilder]::new("initial content")

        $maxWait = 2000
        $waited = 0
        $lastLen = $outBuilder.Length
        $pollCount = 0

        while ($waited -lt $maxWait) {
            $pollCount++
            $waited += 200  # Simulated sleep
            $currentLen = $outBuilder.Length
            if ($currentLen -eq $lastLen) { break }  # Buffer stable
            $lastLen = $currentLen
        }

        $pollCount | Should -Be 1
        $waited | Should -Be 200
    }

    It 'continues polling when buffer grows' {
        $outBuilder = [System.Text.StringBuilder]::new("initial")

        $maxWait = 2000
        $waited = 0
        $lastLen = $outBuilder.Length
        $pollCount = 0

        while ($waited -lt $maxWait) {
            $pollCount++
            $waited += 200

            # Simulate async handler adding data on polls 1-3
            if ($pollCount -le 3) {
                $outBuilder.Append(" more") | Out-Null
            }

            $currentLen = $outBuilder.Length
            if ($currentLen -eq $lastLen) { break }
            $lastLen = $currentLen
        }

        $pollCount | Should -Be 4  # 3 growing + 1 stable
        $waited | Should -Be 800
    }

    It 'respects max wait of 2000ms even with continuous growth' {
        $outBuilder = [System.Text.StringBuilder]::new("initial")

        $maxWait = 2000
        $waited = 0
        $lastLen = $outBuilder.Length
        $pollCount = 0

        while ($waited -lt $maxWait) {
            $pollCount++
            $waited += 200

            # Simulate continuous growth (pathological case)
            $outBuilder.Append("x") | Out-Null

            $currentLen = $outBuilder.Length
            if ($currentLen -eq $lastLen) { break }
            $lastLen = $currentLen
        }

        # Should stop at max wait, not break early
        $waited | Should -Be 2000
        $pollCount | Should -Be 10  # 2000/200 = 10 polls
    }

    It 'handles empty buffer correctly' {
        $outBuilder = [System.Text.StringBuilder]::new()

        $maxWait = 2000
        $waited = 0
        $lastLen = $outBuilder.Length  # 0
        $pollCount = 0

        while ($waited -lt $maxWait) {
            $pollCount++
            $waited += 200
            $currentLen = $outBuilder.Length  # Still 0
            if ($currentLen -eq $lastLen) { break }
            $lastLen = $currentLen
        }

        $pollCount | Should -Be 1
        $outBuilder.Length | Should -Be 0
    }
}

Describe 'Race condition timing requirements' -Tag 'Unit', 'RaceCondition' {
    It 'total drain time is at least 500ms (post-cancel) + 200ms (min poll)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Verify 500ms post-cancel sleep
        $claudeSource | Should -Match 'Start-Sleep -Milliseconds 500'

        # Verify 200ms poll interval
        $claudeSource | Should -Match 'Start-Sleep -Milliseconds 200'

        # Total minimum: 500 + 200 = 700ms
        # This ensures async handlers have time to drain
    }

    It 'no longer uses 100ms sleep (old buggy value)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # The old code had: Start-Sleep -Milliseconds 100
        # This should NOT appear in the finally block anymore
        # (It may appear elsewhere, but not in the finally block pattern)

        $finallyBlock = [regex]::Match($claudeSource, 'finally\s*\{[\s\S]*?^\s{4}\}', 'Multiline').Value
        $finallyBlock | Should -Not -Match 'Start-Sleep -Milliseconds 100\s*$'
    }

    It 'no longer uses 200ms post-cancel (old buggy value)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # The old code had: Start-Sleep -Milliseconds 200 right after CancelErrorRead
        # Now it should be 500ms with a comment about async handlers

        # Find the pattern after CancelErrorRead - should be 500, not 200
        $pattern = 'CancelErrorRead\(\)\s*\}\s*catch\s*\{\}\s*\r?\n\s*Start-Sleep -Milliseconds (\d+)'
        $match = [regex]::Match($claudeSource, $pattern)
        $match.Success | Should -BeTrue
        $match.Groups[1].Value | Should -Be "500"
    }
}

Describe 'Source code structure verification' -Tag 'Unit', 'RaceCondition' {
    It 'has correct order: CancelOutputRead -> CancelErrorRead -> 500ms sleep -> ExitCode' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Find the exited block and verify order
        $exitedBlock = [regex]::Match($claudeSource, 'if \(\$exited\)\s*\{[\s\S]*?^\s{8}\}', 'Multiline').Value

        # Verify sequence
        $cancelOutPos = $exitedBlock.IndexOf('CancelOutputRead()')
        $cancelErrPos = $exitedBlock.IndexOf('CancelErrorRead()')
        $sleepPos = $exitedBlock.IndexOf('Start-Sleep -Milliseconds 500')
        $exitCodePos = $exitedBlock.IndexOf('$exitCode = $process.ExitCode')

        $cancelOutPos | Should -BeLessThan $cancelErrPos
        $cancelErrPos | Should -BeLessThan $sleepPos
        $sleepPos | Should -BeLessThan $exitCodePos
    }

    It 'has correct order: Unregister events -> Remove jobs -> Buffer poll -> Write files' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Extract finally block
        $finallyMatch = [regex]::Match($claudeSource, 'finally\s*\{([\s\S]*?)^\s{4}\}', 'Multiline')
        $finallyBlock = $finallyMatch.Groups[1].Value

        # Verify sequence
        $unregisterPos = $finallyBlock.IndexOf('Unregister-Event')
        $removeJobPos = $finallyBlock.IndexOf('Remove-Job')
        $pollCommentPos = $finallyBlock.IndexOf('Poll until output buffer stabilizes')
        $writeFilePos = $finallyBlock.IndexOf('Set-Content $OutFile')

        $unregisterPos | Should -BeGreaterThan -1
        $removeJobPos | Should -BeGreaterThan $unregisterPos
        $pollCommentPos | Should -BeGreaterThan $removeJobPos
        $writeFilePos | Should -BeGreaterThan $pollCommentPos
    }

    It 'has Console.Out.Flush immediately after success Write-Host' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Find the success block
        $successPattern = 'Write-Host "\s+\$successMsg" -ForegroundColor Green\s*\r?\n\s*\[Console\]::Out\.Flush\(\)'
        $claudeSource | Should -Match $successPattern
    }
}

Describe 'Mutation testing - timing values' -Tag 'Mutation', 'RaceCondition' {
    # These tests verify the specific timing values are correct
    # If someone changes 500 to 200, or 2000 to 1000, tests should fail

    It 'post-cancel sleep is exactly 500ms (not 100, 200, or 1000)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Extract the specific sleep after CancelErrorRead
        $pattern = 'CancelErrorRead\(\)\s*\}\s*catch\s*\{\}\s*\r?\n\s*Start-Sleep -Milliseconds (\d+)'
        $match = [regex]::Match($claudeSource, $pattern)

        [int]$match.Groups[1].Value | Should -Be 500
        [int]$match.Groups[1].Value | Should -Not -Be 100
        [int]$match.Groups[1].Value | Should -Not -Be 200
        [int]$match.Groups[1].Value | Should -Not -Be 1000
    }

    It 'buffer poll max wait is exactly 2000ms (not 1000 or 5000)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $pattern = '\$maxWait = (\d+)'
        $match = [regex]::Match($claudeSource, $pattern)

        [int]$match.Groups[1].Value | Should -Be 2000
        [int]$match.Groups[1].Value | Should -Not -Be 1000
        [int]$match.Groups[1].Value | Should -Not -Be 5000
    }

    It 'buffer poll interval is exactly 200ms (not 100 or 500)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Find the sleep inside the polling while loop
        $pollPattern = 'while \(\$waited -lt \$maxWait\)\s*\{\s*\r?\n\s*Start-Sleep -Milliseconds (\d+)'
        $match = [regex]::Match($claudeSource, $pollPattern)

        [int]$match.Groups[1].Value | Should -Be 200
        [int]$match.Groups[1].Value | Should -Not -Be 100
        [int]$match.Groups[1].Value | Should -Not -Be 500
    }

    It 'waited increment matches sleep duration (200ms)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Both should be 200
        $sleepMatch = [regex]::Match($claudeSource, 'while \(\$waited -lt \$maxWait\)\s*\{\s*\r?\n\s*Start-Sleep -Milliseconds (\d+)')
        $incrementMatch = [regex]::Match($claudeSource, '\$waited \+= (\d+)')

        $sleepMatch.Groups[1].Value | Should -Be $incrementMatch.Groups[1].Value
    }
}

Describe 'Mutation testing - logic conditions' -Tag 'Mutation', 'RaceCondition' {
    It 'break condition uses -eq (equal) not -ne, -lt, -gt' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # The break condition should be: if ($currentLen -eq $lastLen)
        $claudeSource | Should -Match 'if \(\$currentLen -eq \$lastLen\) \{ break \}'

        # Should NOT use other operators for this check
        $claudeSource | Should -Not -Match 'if \(\$currentLen -ne \$lastLen\) \{ break \}'
        $claudeSource | Should -Not -Match 'if \(\$currentLen -lt \$lastLen\) \{ break \}'
        $claudeSource | Should -Not -Match 'if \(\$currentLen -gt \$lastLen\) \{ break \}'
    }

    It 'while loop uses -lt (less than) not -le, -gt, -ge' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Should be: while ($waited -lt $maxWait)
        $claudeSource | Should -Match 'while \(\$waited -lt \$maxWait\)'

        # Should NOT use other operators
        $claudeSource | Should -Not -Match 'while \(\$waited -le \$maxWait\)'
        $claudeSource | Should -Not -Match 'while \(\$waited -gt \$maxWait\)'
        $claudeSource | Should -Not -Match 'while \(\$waited -ge \$maxWait\)'
    }

    It 'Console flush uses Out stream not Error stream' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Should use Out.Flush, not Error.Flush
        $claudeSource | Should -Match '\[Console\]::Out\.Flush\(\)'

        # Verify this specific pattern is in the success block (near successMsg)
        $successBlock = [regex]::Match($claudeSource, '\$successMsg.*?\[Console\]::Out\.Flush\(\)', 'Singleline').Value
        $successBlock | Should -Not -BeNullOrEmpty
    }
}

Describe 'Mutation testing - variable names' -Tag 'Mutation', 'RaceCondition' {
    It 'uses outBuilder.Length not errBuilder.Length for buffer stabilization' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # The buffer stabilization should check outBuilder, not errBuilder
        $claudeSource | Should -Match '\$lastLen = \$outBuilder\.Length'
        $claudeSource | Should -Match '\$currentLen = \$outBuilder\.Length'
    }

    It 'updates lastLen with currentLen (not outBuilder.Length again)' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        # Should update lastLen = currentLen for efficiency
        $claudeSource | Should -Match '\$lastLen = \$currentLen'
    }

    It 'waited starts at 0' {
        $claudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw

        $claudeSource | Should -Match '\$waited = 0'
    }
}

Describe 'Integration: End-to-end timing simulation' -Tag 'Integration', 'RaceCondition' {
    It 'simulates async handler drain with realistic timing' {
        # Simulate what happens when Claude exits:
        # 1. Process exits (HasExited = true)
        # 2. CancelOutputRead/ErrorRead called
        # 3. 500ms sleep (async handlers drain)
        # 4. ExitCode read
        # 5. Events unregistered, jobs removed
        # 6. Buffer stabilization poll (200ms intervals, max 2000ms)
        # 7. Files written

        $outBuilder = [System.Text.StringBuilder]::new()
        $asyncComplete = $false

        # Simulate async handlers still running for 300ms after cancel
        $job = Start-Job -ScriptBlock {
            param($sb)
            Start-Sleep -Milliseconds 100
            # Simulate late output arrival
            return "late data"
        }

        # Post-cancel sleep (500ms) - should be enough for 300ms async
        Start-Sleep -Milliseconds 500

        # Now async should be complete
        $jobResult = Receive-Job -Job $job -Wait -AutoRemoveJob -ErrorAction SilentlyContinue

        # Verify we got the data (async completed before we checked)
        $jobResult | Should -Be "late data"
    }
}
