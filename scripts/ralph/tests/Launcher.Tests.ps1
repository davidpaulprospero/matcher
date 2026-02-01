#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for the unified Ralph Loop entry point (interview.ps1)
.DESCRIPTION
    Tests cover:
    - Mode selection functionality
    - Categorized focus area display
    - Configuration loading
    - Launch behavior for each mode
    - Utility command handling
#>

BeforeAll {
    # Get paths - PSScriptRoot is scripts/ralph/tests, so go up one level for scripts/ralph
    $script:RalphDir = Split-Path -Parent $PSScriptRoot
    $script:InterviewScript = Join-Path $script:RalphDir "interview.ps1"
    # Config file is in config/ subdirectory
    $script:ConfigFile = Join-Path $script:RalphDir "config\ralph-config.json"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\launcher"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source functions from interview.ps1
    if (Test-Path $script:InterviewScript) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:InterviewScript, [ref]$null, [ref]$null)
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

Describe "Interview Configuration" -Tag "Unit", "Interview" {
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

Describe "Focus Area Categories" -Tag "Unit", "Interview" {
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
# Entry Point File Tests
# =============================================================================

Describe "Interview Entry Point Files" -Tag "Unit", "Interview" {
    Context "File existence" {
        It "Should have interview.ps1" {
            Test-Path $script:InterviewScript | Should -BeTrue
        }

        It "Should have ralph.ps1 (internal engine)" {
            $ralphScript = Join-Path $script:RalphDir "ralph.ps1"
            Test-Path $ralphScript | Should -BeTrue
        }

        It "Should NOT have old launcher.ps1" {
            $launcherScript = Join-Path $script:RalphDir "launcher.ps1"
            Test-Path $launcherScript | Should -BeFalse -Because "launcher.ps1 has been consolidated into interview.ps1"
        }

        It "Should NOT have old ralph.bat" {
            $batFile = Join-Path $script:RalphDir "ralph.bat"
            Test-Path $batFile | Should -BeFalse -Because "ralph.bat has been consolidated into interview.ps1"
        }

        It "Should NOT have old numbered bat files" {
            $oldFiles = @(
                "1-im-learnding.bat",
                "2-sleep-thats-where-im-a-viking.bat",
                "3-me-fail-english.bat",
                "4-tastes-like-burning.bat",
                "5-i-bent-my-wookie.bat",
                "6-the-leprechaun-tells-me-to-burn-things.bat",
                "7-hi-super-nintendo-chalmers.bat",
                "sleep-thats-where-im-a-viking.bat"
            )
            foreach ($oldFile in $oldFiles) {
                $path = Join-Path $script:RalphDir $oldFile
                Test-Path $path | Should -BeFalse -Because "$oldFile should be deleted"
            }
        }
    }
}

# =============================================================================
# Interview Function Tests
# =============================================================================

Describe "Interview Functions" -Tag "Unit", "Interview" {
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

    Context "Test-FocusAreaExists" {
        It "Should validate known focus areas" {
            if (Get-Command Test-FocusAreaExists -ErrorAction SilentlyContinue) {
                # Mock config
                $global:config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
                Test-FocusAreaExists -AreaId "pipeline" | Should -BeTrue
            } else {
                Set-ItResult -Skipped -Because "Function not loaded"
            }
        }
    }
}

# =============================================================================
# Mode Selection Tests
# =============================================================================

Describe "Mode Selection" -Tag "Unit", "Interview" {
    Context "Available modes in param block" {
        BeforeAll {
            # Parse the param block to extract ValidateSet values
            $scriptContent = Get-Content $script:InterviewScript -Raw
            if ($scriptContent -match 'ValidateSet\s*\(\s*"([^"]+)"(?:\s*,\s*"([^"]+)")*\s*\)') {
                $script:ValidModes = $Matches[0] -replace 'ValidateSet\s*\(|"|\s|\)' -split ','
            }
        }

        It "Should support standard mode" {
            "standard" | Should -BeIn $script:ValidModes
        }

        It "Should support trueauto mode" {
            "trueauto" | Should -BeIn $script:ValidModes
        }

        It "Should support ralphschoice mode" {
            "ralphschoice" | Should -BeIn $script:ValidModes
        }

        It "Should support ralphschoiceauto mode" {
            "ralphschoiceauto" | Should -BeIn $script:ValidModes
        }

        It "Should support overnight mode" {
            "overnight" | Should -BeIn $script:ValidModes
        }

        It "Should support smartqueue mode" {
            "smartqueue" | Should -BeIn $script:ValidModes
        }
    }

    Context "Utility flags" {
        BeforeAll {
            $scriptContent = Get-Content $script:InterviewScript -Raw
            $script:HasStatusFlag = $scriptContent -match '\[switch\]\$Status'
            $script:HasWatchFlag = $scriptContent -match '\[switch\]\$Watch'
            $script:HasStopFlag = $scriptContent -match '\[switch\]\$Stop'
            $script:HasLogsFlag = $scriptContent -match '\[switch\]\$Logs'
            $script:HasRecoveryFlag = $scriptContent -match '\[switch\]\$Recovery'
            $script:HasMorningFlag = $scriptContent -match '\[switch\]\$Morning'
        }

        It "Should have -Status flag" {
            $script:HasStatusFlag | Should -BeTrue
        }

        It "Should have -Watch flag" {
            $script:HasWatchFlag | Should -BeTrue
        }

        It "Should have -Stop flag" {
            $script:HasStopFlag | Should -BeTrue
        }

        It "Should have -Logs flag" {
            $script:HasLogsFlag | Should -BeTrue
        }

        It "Should have -Recovery flag" {
            $script:HasRecoveryFlag | Should -BeTrue
        }

        It "Should have -Morning flag" {
            $script:HasMorningFlag | Should -BeTrue
        }
    }
}

# =============================================================================
# Parameter Tests
# =============================================================================

Describe "Interview Parameters" -Tag "Unit", "Interview" {
    Context "Required parameters" {
        BeforeAll {
            $scriptContent = Get-Content $script:InterviewScript -Raw
        }

        It "Should have -FocusArea parameter" {
            $scriptContent | Should -Match '\[string\]\$FocusArea'
        }

        It "Should have -FocusAreas parameter for multiple areas" {
            $scriptContent | Should -Match '\[string\[\]\]\$FocusAreas'
        }

        It "Should have -Mode parameter" {
            $scriptContent | Should -Match '\[string\]\$Mode'
        }

        It "Should have -MaxHours parameter" {
            $scriptContent | Should -Match '\[int\]\$MaxHours'
        }

        It "Should have -Resume switch" {
            $scriptContent | Should -Match '\[switch\]\$Resume'
        }

        It "Should have -NoLaunch switch" {
            $scriptContent | Should -Match '\[switch\]\$NoLaunch'
        }
    }
}
