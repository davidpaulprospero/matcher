#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Quick Edit Mode disable functionality.
.DESCRIPTION
    Verifies that Disable-QuickEditMode exists, is callable, correctly
    manipulates console mode flags, and uses the right Win32 constants.

    Includes source-code inspection tests to catch mutations in critical
    constants and logic (ENABLE_QUICK_EDIT value, handle ID, bit operations).

    Note: Some runtime tests are skipped when running in a non-interactive
    subprocess (e.g., CI, piped commands) because there is no console handle.
#>

BeforeAll {
    # Source test helper and import functions
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    $script:RalphDir = Split-Path -Parent $PSScriptRoot
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions @('Disable-QuickEditMode')

    # Read source files for code inspection tests
    $script:DisplaySource = Get-Content (Join-Path $script:RalphDir 'lib\display.ps1') -Raw
    $script:RalphSource = Get-Content (Join-Path $script:RalphDir 'ralph.ps1') -Raw

    # Detect if we have a real console (interactive window vs subprocess pipe)
    $script:HasConsole = $false
    if ($env:OS -eq 'Windows_NT') {
        try {
            if (([System.Management.Automation.PSTypeName]'Win32.ConsoleMode').Type) {
                $h = [Win32.ConsoleMode]::GetStdHandle(-10)
                $m = [uint32]0
                $script:HasConsole = [Win32.ConsoleMode]::GetConsoleMode($h, [ref]$m)
            }
        } catch {}
    }
}

Describe 'Disable-QuickEditMode - Runtime' {

    It 'Function exists and is callable' {
        Get-Command Disable-QuickEditMode -ErrorAction SilentlyContinue | Should -Not -BeNullOrEmpty
    }

    It 'Returns a boolean' {
        $result = Disable-QuickEditMode
        $result | Should -BeOfType [bool]
    }

    It 'Does not throw on invocation' {
        { Disable-QuickEditMode } | Should -Not -Throw
    }

    It 'Returns $false on non-Windows' -Skip:($env:OS -eq 'Windows_NT') {
        Disable-QuickEditMode | Should -Be $false
    }

    It 'Loads Win32.ConsoleMode type on Windows' -Skip:($env:OS -ne 'Windows_NT') {
        Disable-QuickEditMode | Out-Null
        ([System.Management.Automation.PSTypeName]'Win32.ConsoleMode').Type | Should -Not -BeNullOrEmpty
    }

    It 'Is idempotent (calling twice does not throw)' {
        { Disable-QuickEditMode; Disable-QuickEditMode } | Should -Not -Throw
    }

    It 'Returns $false when no console is attached (subprocess)' -Skip:($script:HasConsole -or $env:OS -ne 'Windows_NT') {
        Disable-QuickEditMode | Should -Be $false
    }

    It 'Returns $true and clears Quick Edit bit when console is attached' -Skip:(-not $script:HasConsole) {
        $result = Disable-QuickEditMode
        $result | Should -Be $true

        $ENABLE_QUICK_EDIT = 0x0040
        $hStdin = [Win32.ConsoleMode]::GetStdHandle(-10)
        $mode = [uint32]0
        [Win32.ConsoleMode]::GetConsoleMode($hStdin, [ref]$mode) | Out-Null
        ($mode -band $ENABLE_QUICK_EDIT) | Should -Be 0
    }
}

Describe 'Disable-QuickEditMode - Source Code Inspection' {

    It 'Uses correct ENABLE_QUICK_EDIT constant 0x0040' {
        $script:DisplaySource | Should -Match 'ENABLE_QUICK_EDIT\s*=\s*0x0040'
    }

    It 'Uses correct STD_INPUT_HANDLE constant -10' {
        $script:DisplaySource | Should -Match 'STD_INPUT_HANDLE\s*=\s*-10'
    }

    It 'Uses -band with -bnot to clear the bit (not -bor)' {
        $script:DisplaySource | Should -Match '-band\s*\(\s*-bnot\s*\$ENABLE_QUICK_EDIT\s*\)'
    }

    It 'Checks OS is Windows_NT before proceeding' {
        $script:DisplaySource | Should -Match "env:OS\s*-ne\s*'Windows_NT'"
    }

    It 'Returns $true when Quick Edit was already disabled' {
        # The "already disabled" path should return $true not $false
        $script:DisplaySource | Should -Match 'Quick Edit was already disabled[\s\S]{0,20}return \$true'
    }

    It 'Returns $false when GetConsoleMode fails' {
        $script:DisplaySource | Should -Match 'if\s*\(\s*-not\s*\$gotMode\s*\)\s*\{\s*return\s*\$false\s*\}'
    }

    It 'Contains DllImport for GetStdHandle' {
        $script:DisplaySource | Should -Match 'DllImport[\s\S]*?kernel32[\s\S]*?GetStdHandle'
    }

    It 'Contains DllImport for GetConsoleMode' {
        $script:DisplaySource | Should -Match 'DllImport[\s\S]*?kernel32[\s\S]*?GetConsoleMode'
    }

    It 'Contains DllImport for SetConsoleMode' {
        $script:DisplaySource | Should -Match 'DllImport[\s\S]*?kernel32[\s\S]*?SetConsoleMode'
    }

    It 'Returns $false in catch block (non-fatal)' {
        $script:DisplaySource | Should -Match 'catch\s*\{[\s\S]*?return\s*\$false[\s\S]*?\}'
    }
}

Describe 'Disable-QuickEditMode - Integration in ralph.ps1' {

    It 'Calls Disable-QuickEditMode at startup in ralph.ps1' {
        $script:RalphSource | Should -Match 'Disable-QuickEditMode'
    }

    It 'Stores result in $quickEditDisabled variable' {
        $script:RalphSource | Should -Match '\$quickEditDisabled\s*=\s*Disable-QuickEditMode'
    }

    It 'Loads display.ps1 before calling Disable-QuickEditMode' {
        # display.ps1 dot-source should appear before the call
        $displayLoadPos = $script:RalphSource.IndexOf("display.ps1")
        $callPos = $script:RalphSource.IndexOf("Disable-QuickEditMode")
        $displayLoadPos | Should -BeLessThan $callPos
        $displayLoadPos | Should -BeGreaterOrEqual 0
    }

    It 'Calls before loops.ps1 is loaded (early enough in startup)' {
        $callPos = $script:RalphSource.IndexOf("Disable-QuickEditMode")
        $loopsLoadPos = $script:RalphSource.IndexOf("loops.ps1")
        $callPos | Should -BeLessThan $loopsLoadPos
    }
}

Describe 'QuickEditMode periodic safeguard in claude.ps1' {

    BeforeAll {
        $script:ClaudeSource = Get-Content (Join-Path $script:RalphDir 'lib\claude.ps1') -Raw
    }

    It 'Defines quickEditRecheckIntervalSec variable in Invoke-ClaudeSubprocess' {
        $script:ClaudeSource | Should -Match 'quickEditRecheckIntervalSec\s*=\s*60'
    }

    It 'Initializes lastQuickEditRecheck to 0' {
        $script:ClaudeSource | Should -Match '\$lastQuickEditRecheck\s*=\s*0'
    }

    It 'Re-checks Quick Edit Mode within the monitoring loop' {
        # The re-disable logic should be after the while loop starts
        $script:ClaudeSource | Should -Match 'while.*HasExited.*\{[\s\S]{0,1000}lastQuickEditRecheck'
    }

    It 'Uses try/catch around Disable-QuickEditMode in the safeguard' {
        # Should have try/catch to prevent crashes when console is unavailable
        # Check for try/catch pattern around Disable-QuickEditMode
        $script:ClaudeSource | Should -Match 'try\s*\{[^}]*Disable-QuickEditMode[^}]*\}[\s]*catch'
    }

    It 'Compares elapsed time against quickEditRecheckIntervalSec' {
        $script:ClaudeSource | Should -Match '\$totalElapsed\s*-\s*\$lastQuickEditRecheck\s*-ge\s*\$quickEditRecheckIntervalSec'
    }

    It 'Updates lastQuickEditRecheck after re-checking' {
        $script:ClaudeSource | Should -Match '\$lastQuickEditRecheck\s*=\s*\$totalElapsed'
    }

    It 'Does not output messages during periodic re-check (silent safeguard)' {
        # Should NOT have Write-Host inside the periodic re-check block
        $match = $script:ClaudeSource -match '(if\s*\(\s*\$totalElapsed\s*-\s*\$lastQuickEditRecheck[\s\S]{0,300}\$qeDisabled\s*=\s*Disable-QuickEditMode[\s\S]{0,100})'
        if ($match) {
            $block = $Matches[1]
            $block | Should -Not -Match 'Write-Host'
        }
    }
}
