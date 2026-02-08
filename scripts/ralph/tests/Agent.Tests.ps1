#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for the Ralph agent abstraction layer (agent.ps1, providers)
.DESCRIPTION
    Tests cover:
    - Provider resolution (default, config-driven, legacy fallback)
    - Model mapping (tier-based, defaults, legacy)
    - Command building (Claude vs Codex args, prompt method)
    - Provider instructions (preamble text)
    - Backward compatibility (no agent section in config)
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot
    $script:LibDir = Join-Path $script:RalphDir "lib"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata"

    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Provide stubs for dependencies that the agent files reference
    # Get-RalphConfig is mocked per-test; Get-ClaudePath needs a basic stub
    if (-not (Get-Command Get-RalphConfig -ErrorAction SilentlyContinue)) {
        function global:Get-RalphConfig { return @{} }
    }
    if (-not (Get-Command Get-ClaudePath -ErrorAction SilentlyContinue)) {
        function global:Get-ClaudePath { return "claude" }
    }

    # Dot-source provider files first (no external deps), then agent.ps1
    . (Join-Path $script:LibDir "agents\claude_provider.ps1")
    . (Join-Path $script:LibDir "agents\codex_provider.ps1")
    . (Join-Path $script:LibDir "agent.ps1")
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# PROVIDER RESOLUTION
# =============================================================================

Describe "Get-AgentProvider" -Tag "Unit", "Agent" {

    It "Returns 'claude' when config has agent.provider = 'claude'" {
        Mock Get-RalphConfig { @{ agent = @{ provider = "claude" } } }
        Get-AgentProvider | Should -Be "claude"
    }

    It "Returns 'codex' when config has agent.provider = 'codex'" {
        Mock Get-RalphConfig { @{ agent = @{ provider = "codex" } } }
        Get-AgentProvider | Should -Be "codex"
    }

    It "Defaults to 'claude' when no agent section exists" {
        Mock Get-RalphConfig { @{ model = "opus"; claudePath = "claude" } }
        Get-AgentProvider | Should -Be "claude"
    }

    It "Defaults to 'claude' when agent section has no provider field" {
        Mock Get-RalphConfig { @{ agent = @{ providers = @{} } } }
        Get-AgentProvider | Should -Be "claude"
    }
}

# =============================================================================
# MODEL MAPPING
# =============================================================================

Describe "Get-AgentModel" -Tag "Unit", "Agent" {

    BeforeAll {
        $script:FullConfig = @{
            agent = @{
                provider = "claude"
                providers = @{
                    claude = @{
                        executable = "claude"
                        models = @{ high = "opus"; fast = "sonnet"; default = "opus" }
                    }
                    codex = @{
                        executable = "codex"
                        models = @{ high = "o4-mini"; fast = "o4-mini"; default = "o4-mini" }
                    }
                }
            }
            model = "opus"
        }
    }

    It "Maps claude 'high' tier to 'opus'" {
        Mock Get-RalphConfig { $script:FullConfig }
        Get-AgentModel -Provider "claude" -Tier "high" | Should -Be "opus"
    }

    It "Maps claude 'fast' tier to 'sonnet'" {
        Mock Get-RalphConfig { $script:FullConfig }
        Get-AgentModel -Provider "claude" -Tier "fast" | Should -Be "sonnet"
    }

    It "Maps codex 'high' tier to 'o4-mini'" {
        Mock Get-RalphConfig { $script:FullConfig }
        Get-AgentModel -Provider "codex" -Tier "high" | Should -Be "o4-mini"
    }

    It "Falls back to 'default' tier when unknown tier specified" {
        Mock Get-RalphConfig { $script:FullConfig }
        Get-AgentModel -Provider "claude" -Tier "unknown" | Should -Be "opus"
    }

    It "Falls back to legacy config.model for claude when no agent section" {
        Mock Get-RalphConfig { @{ model = "sonnet" } }
        Get-AgentModel -Provider "claude" | Should -Be "sonnet"
    }

    It "Returns 'opus' as ultimate fallback for claude" {
        Mock Get-RalphConfig { @{} }
        Get-AgentModel -Provider "claude" | Should -Be "opus"
    }

    It "Returns 'o4-mini' as ultimate fallback for codex" {
        Mock Get-RalphConfig { @{} }
        Get-AgentModel -Provider "codex" | Should -Be "o4-mini"
    }
}

# =============================================================================
# COMMAND BUILDING - CLAUDE
# =============================================================================

Describe "Build-ClaudeCommand" -Tag "Unit", "Agent" {

    It "Returns stdin prompt method" {
        $result = Build-ClaudeCommand -Model "opus"
        $result.PromptMethod | Should -Be "stdin"
    }

    It "Includes --print and --dangerously-skip-permissions" {
        $result = Build-ClaudeCommand -Model "opus"
        $result.Args | Should -Contain "--print"
        $result.Args | Should -Contain "--dangerously-skip-permissions"
    }

    It "Sets the model correctly" {
        $result = Build-ClaudeCommand -Model "sonnet"
        $modelIdx = [array]::IndexOf($result.Args, "--model")
        $result.Args[$modelIdx + 1] | Should -Be "sonnet"
    }

    It "Adds read-only allowed tools when ReadOnly option set" {
        $result = Build-ClaudeCommand -Model "opus" -Options @{ ReadOnly = $true }
        $result.Args -join ' ' | Should -BeLike "*--allowedTools=Read,Glob,Grep*"
    }

    It "Adds full allowed tools when AllowedTools option set" {
        $result = Build-ClaudeCommand -Model "opus" -Options @{ AllowedTools = $true }
        $result.Args -join ' ' | Should -BeLike "*--allowedTools=Bash,Read,Write,Edit,Glob,Grep,WebSearch*"
    }

    It "AllowedTools takes priority over ReadOnly" {
        $result = Build-ClaudeCommand -Model "opus" -Options @{ AllowedTools = $true; ReadOnly = $true }
        $result.Args -join ' ' | Should -BeLike "*--allowedTools=Bash,Read,Write,Edit,Glob,Grep,WebSearch*"
    }
}

# =============================================================================
# COMMAND BUILDING - CODEX
# =============================================================================

Describe "Build-CodexCommand" -Tag "Unit", "Agent" {

    It "Returns arg prompt method" {
        $result = Build-CodexCommand -Model "o4-mini" -Prompt "test"
        $result.PromptMethod | Should -Be "arg"
    }

    It "Includes exec and --full-auto" {
        $result = Build-CodexCommand -Model "o4-mini" -Prompt "test"
        $result.Args | Should -Contain "exec"
        $result.Args | Should -Contain "--full-auto"
    }

    It "Embeds short prompt directly in args" {
        $shortPrompt = "Fix the bug in main.py"
        $result = Build-CodexCommand -Model "o4-mini" -Prompt $shortPrompt
        $result.Args | Should -Contain $shortPrompt
        $result.TempFile | Should -BeNullOrEmpty
    }

    It "Adds --sandbox strict when ReadOnly set" {
        $result = Build-CodexCommand -Model "o4-mini" -Prompt "test" -Options @{ ReadOnly = $true }
        $result.Args | Should -Contain "--sandbox"
        $result.Args | Should -Contain "strict"
    }

    It "Adds workspace when specified" {
        $result = Build-CodexCommand -Model "o4-mini" -Prompt "test" -Options @{ Workspace = "C:\project" }
        $result.Args | Should -Contain "--workspace"
        $result.Args | Should -Contain "C:\project"
    }
}

# =============================================================================
# BUILD-AGENTCOMMAND DISPATCH
# =============================================================================

Describe "Build-AgentCommand" -Tag "Unit", "Agent" {

    It "Dispatches to Claude provider for 'claude'" {
        $result = Build-AgentCommand -Provider "claude" -Model "opus" -Prompt "test"
        $result.PromptMethod | Should -Be "stdin"
        $result.Args | Should -Contain "--print"
    }

    It "Dispatches to Codex provider for 'codex'" {
        $result = Build-AgentCommand -Provider "codex" -Model "o4-mini" -Prompt "test"
        $result.PromptMethod | Should -Be "arg"
        $result.Args | Should -Contain "exec"
    }

    It "Falls back to Claude for unknown provider" {
        $result = Build-AgentCommand -Provider "unknown-agent" -Model "test" -Prompt "test"
        $result.PromptMethod | Should -Be "stdin"
        $result.Args | Should -Contain "--print"
    }
}

# =============================================================================
# PROVIDER INSTRUCTIONS
# =============================================================================

Describe "Get-ProviderInstructions" -Tag "Unit", "Agent" {

    It "Returns empty string for claude (uses CLAUDE.md)" {
        Get-ProviderInstructions -Provider "claude" | Should -BeExactly ""
    }

    It "Returns non-empty preamble for codex" {
        $result = Get-ProviderInstructions -Provider "codex"
        $result | Should -Not -BeNullOrEmpty
        $result | Should -BeLike "*Codex*"
    }

    It "Codex instructions mention pytest" {
        $result = Get-ProviderInstructions -Provider "codex"
        $result | Should -BeLike "*pytest*"
    }

    It "Returns empty for unknown provider" {
        Get-ProviderInstructions -Provider "unknown" | Should -BeExactly ""
    }
}

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

Describe "Backward Compatibility" -Tag "Unit", "Agent" {

    It "Full pipeline defaults to Claude when no agent config" {
        Mock Get-RalphConfig { @{ claudePath = "claude"; model = "opus" } }
        $provider = Get-AgentProvider
        $provider | Should -Be "claude"
        $model = Get-AgentModel -Provider $provider
        $model | Should -Be "opus"
    }

    It "Build-AgentCommand works with defaults from legacy config" {
        $result = Build-AgentCommand -Provider "claude" -Model "opus" -Prompt "test prompt"
        $result.PromptMethod | Should -Be "stdin"
        $result.Args.Count | Should -BeGreaterThan 3
    }

    It "Legacy model field respected when agent section missing" {
        Mock Get-RalphConfig { @{ model = "haiku" } }
        Get-AgentModel -Provider "claude" | Should -Be "haiku"
    }
}
