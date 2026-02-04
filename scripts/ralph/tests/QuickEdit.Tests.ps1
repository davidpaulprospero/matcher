#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Quick Edit Mode disable functionality.
.DESCRIPTION
    Verifies that Disable-QuickEditMode exists, is callable, and correctly
    manipulates console mode flags to prevent Mark mode from freezing the
    Ralph monitoring loop.

    Note: Some tests are skipped when running in a non-interactive subprocess
    (e.g., CI, piped commands) because there is no console handle to manipulate.
#>

BeforeAll {
    # Source test helper and import functions
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    $script:RalphDir = Split-Path -Parent $PSScriptRoot
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions @('Disable-QuickEditMode')

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

Describe 'Disable-QuickEditMode' {

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
        $result = Disable-QuickEditMode
        $result | Should -Be $false
    }

    It 'Loads Win32.ConsoleMode type on Windows' -Skip:($env:OS -ne 'Windows_NT') {
        Disable-QuickEditMode | Out-Null
        ([System.Management.Automation.PSTypeName]'Win32.ConsoleMode').Type | Should -Not -BeNullOrEmpty
    }

    It 'Is idempotent (calling twice does not throw)' {
        { Disable-QuickEditMode; Disable-QuickEditMode } | Should -Not -Throw
    }

    It 'Returns $false when no console is attached (subprocess)' -Skip:($script:HasConsole -or $env:OS -ne 'Windows_NT') {
        # In a subprocess with no console, GetConsoleMode fails and function returns $false
        $result = Disable-QuickEditMode
        $result | Should -Be $false
    }

    It 'Returns $true and clears Quick Edit bit when console is attached' -Skip:(-not $script:HasConsole) {
        $result = Disable-QuickEditMode
        $result | Should -Be $true

        # Verify the bit is actually cleared
        $ENABLE_QUICK_EDIT = 0x0040
        $hStdin = [Win32.ConsoleMode]::GetStdHandle(-10)
        $mode = [uint32]0
        [Win32.ConsoleMode]::GetConsoleMode($hStdin, [ref]$mode) | Out-Null
        ($mode -band $ENABLE_QUICK_EDIT) | Should -Be 0
    }
}
