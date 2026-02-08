# scripts/ralph/lib/agent.ps1
# Agent abstraction layer: provider-agnostic process spawning
#
# Supports multiple AI CLI backends (Claude Code, OpenAI Codex, etc.)
# via a clean provider interface. All monitoring, healing, and stall
# detection works identically regardless of which agent is selected.

# Load provider modules
$script:AgentProvidersPath = Join-Path $PSScriptRoot 'agents'

# ============================================================================
# PROVIDER RESOLUTION
# ============================================================================

function Get-AgentProvider {
    <#
    .SYNOPSIS
        Read the configured agent provider name.
        Checks config.agent.provider first, falls back to legacy "claude" default.
    .RETURNS
        Provider string: "claude", "codex", etc.
    #>

    $config = Get-RalphConfig

    # New config path: agent.provider
    if ($config.agent -and $config.agent.provider) {
        return $config.agent.provider
    }

    # Legacy default
    return "claude"
}

function Get-AgentExecutable {
    <#
    .SYNOPSIS
        Resolve the CLI executable path for a given provider.
        Checks config.agent.providers.<provider>.executable, then falls back
        to the provider name itself (assuming it's in PATH).
    .PARAMETER Provider
        The provider name (e.g., "claude", "codex")
    .RETURNS
        Full path or command name for the CLI executable
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Provider
    )

    $config = Get-RalphConfig

    # Check agent.providers.<provider>.executable
    if ($config.agent -and $config.agent.providers -and $config.agent.providers.$Provider) {
        $providerConfig = $config.agent.providers.$Provider
        if ($providerConfig.executable) {
            $exe = $providerConfig.executable

            # If it's a bare name, try to resolve it
            if ($exe -eq "claude") {
                return Get-ClaudePath
            }

            if ($exe -eq "codex") {
                # Try common Codex CLI locations
                $possiblePaths = @(
                    "$env:APPDATA\npm\codex.cmd",
                    "$env:USERPROFILE\.npm-global\codex.cmd",
                    "$env:LOCALAPPDATA\Programs\codex\codex.exe"
                )
                foreach ($path in $possiblePaths) {
                    if (Test-Path $path) {
                        return $path
                    }
                }
                return "codex"  # Fallback to PATH
            }

            return $exe
        }
    }

    # Legacy fallback: for "claude" provider, use Get-ClaudePath
    if ($Provider -eq "claude") {
        return Get-ClaudePath
    }

    # Default: just return the provider name (hope it's in PATH)
    return $Provider
}

function Get-AgentModel {
    <#
    .SYNOPSIS
        Map quality tier to provider-specific model name.
    .PARAMETER Provider
        The provider name
    .PARAMETER Tier
        Quality tier: "high", "fast", or "default"
    .RETURNS
        Model name string for the provider's CLI
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Provider,
        [string]$Tier = "default"
    )

    $config = Get-RalphConfig

    # Check agent.providers.<provider>.models.<tier>
    if ($config.agent -and $config.agent.providers -and $config.agent.providers.$Provider) {
        $providerConfig = $config.agent.providers.$Provider
        if ($providerConfig.models) {
            $model = $providerConfig.models.$Tier
            if ($model) { return $model }

            # Fallback to "default" tier
            $model = $providerConfig.models.default
            if ($model) { return $model }
        }
    }

    # Legacy fallback: use config.model for claude
    if ($Provider -eq "claude") {
        if ($config.model) { return $config.model }
        return "opus"
    }

    # Codex default
    if ($Provider -eq "codex") {
        return "o4-mini"
    }

    return "default"
}

function Build-AgentCommand {
    <#
    .SYNOPSIS
        Build CLI arguments and determine prompt delivery method for a provider.
        Dispatches to provider-specific Build-*Command functions.
    .PARAMETER Provider
        The provider name
    .PARAMETER Model
        The model to use
    .PARAMETER Prompt
        The prompt text (needed for arg-based providers)
    .PARAMETER Options
        Hashtable of options: AllowedTools (bool), ReadOnly (bool), Workspace (string)
    .RETURNS
        Hashtable: Args (string[]), PromptMethod ("stdin" or "arg")
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Provider,
        [Parameter(Mandatory)]
        [string]$Model,
        [string]$Prompt = "",
        [hashtable]$Options = @{}
    )

    switch ($Provider) {
        "claude" {
            return Build-ClaudeCommand -Model $Model -Options $Options
        }
        "codex" {
            return Build-CodexCommand -Model $Model -Prompt $Prompt -Options $Options
        }
        default {
            Write-Host "  Warning: Unknown agent provider '$Provider', falling back to claude" -ForegroundColor Yellow
            return Build-ClaudeCommand -Model $Model -Options $Options
        }
    }
}

function Get-ProviderInstructions {
    <#
    .SYNOPSIS
        Get provider-specific preamble text for prompts.
        Claude and Codex have different CLI behaviors and conventions.
    .PARAMETER Provider
        The provider name
    .RETURNS
        String with provider-specific context, or empty string
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Provider
    )

    switch ($Provider) {
        "codex" {
            return @"
NOTE: You are running as OpenAI Codex CLI in full-auto mode.
- Use standard shell commands for file operations
- Commit changes with descriptive messages referencing the story ID
- Run tests with: pytest tests/ --tb=short -q --non-interactive
- Update scripts/ralph/state/prd.json to set passes: true when the story is complete
"@
        }
        "claude" {
            # Claude Code gets its instructions from CLAUDE.md and prompt.md
            return ""
        }
        default {
            return ""
        }
    }
}
