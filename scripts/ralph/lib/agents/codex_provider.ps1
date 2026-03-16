# scripts/ralph/lib/agents/codex_provider.ps1
# OpenAI Codex CLI provider for Ralph agent abstraction

function Build-CodexCommand {
    <#
    .SYNOPSIS
        Build OpenAI Codex CLI arguments.
        Prompt is delivered as a CLI argument (PromptMethod = "arg").
        For long prompts (>20K chars), writes to a temp file.
    .PARAMETER Model
        Codex model name (e.g., "o4-mini")
    .PARAMETER Prompt
        The prompt text to embed in args
    .PARAMETER Options
        Hashtable: ReadOnly (bool), Workspace (string)
    .RETURNS
        Hashtable: Args (string[]), PromptMethod ("arg"), TempFile (string or $null)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Model,
        [string]$Prompt = "",
        [hashtable]$Options = @{}
    )

    $cmdArgs = @("exec", "--full-auto", "--model", $Model)

    if ($Options.ReadOnly) {
        $cmdArgs += @("--sandbox", "strict")
    }

    if ($Options.Workspace) {
        $cmdArgs += @("--workspace", $Options.Workspace)
    }

    # Handle prompt delivery
    $tempFile = $null
    if ($Prompt.Length -gt 20000) {
        # Long prompt: write to temp file to avoid Windows CLI length limits (~32K)
        $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) "ralph_codex_prompt_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        $Prompt | Set-Content $tempFile -Encoding UTF8 -NoNewline
        # Codex reads prompt from the last positional arg
        $cmdArgs += Get-Content $tempFile -Raw
        Write-Host "  [INFO] Long prompt ($($Prompt.Length) chars) - using temp file" -ForegroundColor DarkGray
    }
    elseif ($Prompt) {
        $cmdArgs += $Prompt
    }

    return @{
        Args = $cmdArgs
        PromptMethod = "arg"
        TempFile = $tempFile
    }
}
