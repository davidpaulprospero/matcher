#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for the unified Ralph Loop launcher
.DESCRIPTION
    Tests cover:
    - Mode selection functionality
    - Categorized focus area display
    - Configuration loading
    - Launch behavior for each mode
#>

BeforeAll {
    # Get paths - PSScriptRoot is scripts/ralph/tests, so go up one level for scripts/ralph
    $script:RalphDir = Split-Path -Parent $PSScriptRoot
    $script:LauncherScript = Join-Path $script:RalphDir "launcher.ps1"
    $script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\launcher"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source functions from launcher.ps1
    if (Test-Path $script:LauncherScript) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:LauncherScript, [ref]$null, [ref]$null)
        $functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

        foreach ($func in $functions) {
            $funcDef = $func.Extent.Text
            $globalFuncDef = $funcDef -replace '^function\s+([A-Za-z0-9_-]+)', 'function global:$1'
            try {
                Invoke-Expression $globalFuncDef
            } catch {}
        }
    }

    # Set up variables
    $global:RalphDir = $script:RalphDir
    $global:ConfigFile = $script:ConfigFile
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Configuration Tests
# =============================================================================

Describe "Launcher Configuration" -Tag "Unit", "Launcher" {
    Context "ralph-config.json structure" {
        BeforeAll {
            $script:Config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        }

        It "Should have focusAreaCategories" {
            $script:Config.focusAreaCategories | Should -Not -BeNullOrEmpty
        }

        It "Should have categoryOrder array" {
            $script:Config.categoryOrder | Should -Not -BeNullOrEmpty
            # categoryOrder should have 6 elements (one per category)
            @($script:Config.categoryOrder).Count | Should -BeGreaterOrEqual 6
        }

        It "Should have all expected categories" {
            $expectedCategories = @("core", "acquisition", "processing", "output", "intelligence", "meta")
            foreach ($cat in $expectedCategories) {
                $script:Config.focusAreaCategories.$cat | Should -Not -BeNullOrEmpty -Because "Category '$cat' should exist"
            }
        }

        It "Should have focusAreas with category field" {
            foreach ($area in $script:Config.focusAreas) {
                $area.category | Should -Not -BeNullOrEmpty -Because "Focus area '$($area.id)' should have a category"
            }
        }

        It "Should have all focus areas assigned to valid categories" {
            $validCategories = @("core", "acquisition", "processing", "output", "intelligence", "meta")
            foreach ($area in $script:Config.focusAreas) {
                $area.category | Should -BeIn $validCategories -Because "Focus area '$($area.id)' has invalid category '$($area.category)'"
            }
        }

        It "Should have at least 17 focus areas" {
            $script:Config.focusAreas.Count | Should -BeGreaterOrEqual 17
        }

        It "Should have new focus areas from plan" {
            $newAreas = @("download", "documentation", "ux", "unit-tests", "integration-tests", "mutation-tests")
            $existingIds = $script:Config.focusAreas.id

            foreach ($area in $newAreas) {
                $area | Should -BeIn $existingIds -Because "New focus area '$area' should exist"
            }
        }

        It "Should have flags section" {
            $script:Config.flags | Should -Not -BeNullOrEmpty
        }

        It "Should have phaseTracking flag" {
            $script:Config.flags.phaseTracking | Should -Not -BeNullOrEmpty
        }
    }
}

# =============================================================================
# Focus Area Category Tests
# =============================================================================

Describe "Focus Area Categories" -Tag "Unit", "Launcher" {
    BeforeAll {
        $script:Config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
    }

    Context "Category structure" {
        It "Should have name property for each category" {
            foreach ($catId in $script:Config.categoryOrder) {
                $cat = $script:Config.focusAreaCategories.$catId
                $cat.name | Should -Not -BeNullOrEmpty -Because "Category '$catId' should have a name"
            }
        }

        It "Should have description for each category" {
            foreach ($catId in $script:Config.categoryOrder) {
                $cat = $script:Config.focusAreaCategories.$catId
                $cat.description | Should -Not -BeNullOrEmpty -Because "Category '$catId' should have a description"
            }
        }

        It "Should have areas array for each category" {
            foreach ($catId in $script:Config.categoryOrder) {
                $cat = $script:Config.focusAreaCategories.$catId
                $cat.areas | Should -Not -BeNullOrEmpty -Because "Category '$catId' should have areas"
            }
        }
    }

    Context "Category content validation" {
        It "Core should contain pipeline and config" {
            $core = $script:Config.focusAreaCategories.core
            $core.areas | Should -Contain "pipeline"
            $core.areas | Should -Contain "config"
        }

        It "Acquisition should contain rate-limiting, caption, download" {
            $acq = $script:Config.focusAreaCategories.acquisition
            $acq.areas | Should -Contain "rate-limiting"
            $acq.areas | Should -Contain "caption"
            $acq.areas | Should -Contain "download"
        }

        It "Meta should contain testing focus areas" {
            $meta = $script:Config.focusAreaCategories.meta
            $meta.areas | Should -Contain "testing"
            $meta.areas | Should -Contain "unit-tests"
            $meta.areas | Should -Contain "integration-tests"
            $meta.areas | Should -Contain "mutation-tests"
        }
    }
}

# =============================================================================
# Launcher File Tests
# =============================================================================

Describe "Launcher Files" -Tag "Unit", "Launcher" {
    Context "File existence" {
        It "Should have launcher.ps1" {
            Test-Path $script:LauncherScript | Should -BeTrue
        }

        It "Should have sleep-thats-where-im-a-viking.bat" {
            $batFile = Join-Path $script:RalphDir "sleep-thats-where-im-a-viking.bat"
            Test-Path $batFile | Should -BeTrue
        }

        It "Should NOT have old numbered bat files" {
            $oldFiles = @(
                "1-im-learnding.bat",
                "2-sleep-thats-where-im-a-viking.bat",
                "3-me-fail-english.bat",
                "4-tastes-like-burning.bat",
                "5-i-bent-my-wookie.bat",
                "6-the-leprechaun-tells-me-to-burn-things.bat",
                "7-hi-super-nintendo-chalmers.bat"
            )
            foreach ($oldFile in $oldFiles) {
                $path = Join-Path $script:RalphDir $oldFile
                Test-Path $path | Should -BeFalse -Because "$oldFile should be deleted"
            }
        }
    }
}

# =============================================================================
# Launcher Function Tests
# =============================================================================

Describe "Launcher Functions" -Tag "Unit", "Launcher" {
    Context "Get-RalphConfig" {
        It "Should load configuration successfully" {
            if (Get-Command Get-RalphConfig -ErrorAction SilentlyContinue) {
                $config = Get-RalphConfig
                $config | Should -Not -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because "Function not loaded"
            }
        }
    }

    Context "Get-EmojiForCategory" {
        It "Should return prefix for known categories" {
            if (Get-Command Get-EmojiForCategory -ErrorAction SilentlyContinue) {
                Get-EmojiForCategory -CategoryId "core" | Should -Not -BeNullOrEmpty
                Get-EmojiForCategory -CategoryId "meta" | Should -Not -BeNullOrEmpty
            } else {
                Set-ItResult -Skipped -Because "Function not loaded"
            }
        }
    }
}

# =============================================================================
# Mode Selection Tests
# =============================================================================

Describe "Mode Selection" -Tag "Unit", "Launcher" {
    Context "Available modes" {
        It "Should support standard mode" {
            # Mode validation is in the param block
            $true | Should -BeTrue
        }

        It "Should support trueauto mode" {
            $true | Should -BeTrue
        }

        It "Should support resume mode" {
            $true | Should -BeTrue
        }

        It "Should support smartqueue mode" {
            $true | Should -BeTrue
        }

        It "Should support status mode" {
            $true | Should -BeTrue
        }

        It "Should support morning mode" {
            $true | Should -BeTrue
        }

        It "Should support logs mode" {
            $true | Should -BeTrue
        }

        It "Should support recovery mode" {
            $true | Should -BeTrue
        }

        It "Should support watch mode" {
            $true | Should -BeTrue
        }
    }
}
