#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Ralph script organization conventions
.DESCRIPTION
    Enforces the convention:
    - scripts/ralph/lib/*.ps1  → Shared functions (sourced by ralph.ps1)
    - scripts/ralph/*.ps1      → Standalone entry points (use lib functions, don't define shared ones)
#>

BeforeDiscovery {
    # Compute paths at discovery time
    $script:RalphDir = Join-Path $PSScriptRoot ".."
    $script:LibDir = Join-Path $script:RalphDir "lib"

    # Get all standalone scripts (exclude watch-lib.ps1 which has TODO to move)
    $script:StandaloneScripts = Get-ChildItem -Path $script:RalphDir -Filter "*.ps1" -File |
        Where-Object { $_.Name -ne "watch-lib.ps1" } |
        ForEach-Object { @{ Name = $_.Name; Path = $_.FullName } }

    # Get all lib scripts
    $script:LibScripts = Get-ChildItem -Path $script:LibDir -Filter "*.ps1" -File -ErrorAction SilentlyContinue |
        ForEach-Object { @{ Name = $_.Name; Path = $_.FullName } }

    # Extract functions defined in lib
    $libFunctions = @()
    foreach ($libScript in (Get-ChildItem -Path $script:LibDir -Filter "*.ps1" -File -ErrorAction SilentlyContinue)) {
        $content = Get-Content $libScript.FullName -Raw
        $funcMatches = [regex]::Matches($content, 'function\s+([A-Za-z][A-Za-z0-9_-]*)\s*[\{\(:]')
        foreach ($match in $funcMatches) {
            $libFunctions += $match.Groups[1].Value
        }
    }

    # Extract functions defined ONLY in standalone scripts (not also in lib)
    $script:StandaloneFunctions = @()
    foreach ($standaloneScript in (Get-ChildItem -Path $script:RalphDir -Filter "*.ps1" -File | Where-Object { $_.Name -ne "watch-lib.ps1" })) {
        $content = Get-Content $standaloneScript.FullName -Raw
        $funcMatches = [regex]::Matches($content, 'function\s+([A-Za-z][A-Za-z0-9_-]*)\s*[\{\(:]')
        foreach ($match in $funcMatches) {
            $funcName = $match.Groups[1].Value
            if ($funcName -notin $libFunctions) {
                $script:StandaloneFunctions += @{
                    FuncName = $funcName
                    DefinedIn = $standaloneScript.Name
                }
            }
        }
    }

    # Build cross-check test cases (lib x standalone functions)
    $script:CrossCallTests = @()
    foreach ($libScript in $script:LibScripts) {
        foreach ($func in $script:StandaloneFunctions) {
            $script:CrossCallTests += @{
                LibName = $libScript.Name
                LibPath = $libScript.Path
                FuncName = $func.FuncName
                DefinedIn = $func.DefinedIn
            }
        }
    }
}

Describe "Script Organization Convention" {

    Context "Standalone scripts have required header" {
        It "[<Name>] should have STANDALONE SCRIPT header" -ForEach $StandaloneScripts {
            $content = Get-Content $Path -Raw
            $content | Should -Match "STANDALONE SCRIPT"
        }
    }

    Context "Lib files do NOT have standalone header" {
        It "[lib/<Name>] should NOT have STANDALONE SCRIPT header" -ForEach $LibScripts {
            $content = Get-Content $Path -Raw
            $content | Should -Not -Match "STANDALONE SCRIPT"
        }
    }

    Context "Lib files do not call standalone-only functions" {
        It "[lib/<LibName>] should not call <FuncName> (from <DefinedIn>)" -ForEach $CrossCallTests {
            $content = Get-Content $LibPath -Raw
            # Pattern: function name as word boundary followed by (
            $content | Should -Not -Match "\b$FuncName\s*\("
        }
    }

    Context "watch-lib.ps1 migration complete" {
        It "watch-lib.ps1 should be moved to lib/watch.ps1" {
            $ralphDir = Join-Path $PSScriptRoot ".."
            $watchLib = Join-Path $ralphDir "watch-lib.ps1"
            $libWatch = Join-Path $ralphDir "lib\watch.ps1"
            # Old location should NOT exist
            Test-Path $watchLib | Should -Be $false
            # New location SHOULD exist
            Test-Path $libWatch | Should -Be $true
        }
    }
}
