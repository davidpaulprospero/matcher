# scripts/ralph/lib/agents/claude_provider.ps1
# Claude Code CLI provider for Ralph agent abstraction

function Build-ClaudeCommand {
    <#
    .SYNOPSIS
        Build Claude Code CLI arguments.
        Prompt is delivered via stdin (PromptMethod = "stdin").
    .PARAMETER Model
        Claude model name (e.g., "opus", "sonnet")
    .PARAMETER Options
        Hashtable: AllowedTools (bool), ReadOnly (bool)
    .RETURNS
        Hashtable: Args (string[]), PromptMethod ("stdin")
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Model,
        [hashtable]$Options = @{}
    )

    $cmdArgs = @("--print", "--dangerously-skip-permissions", "--model", $Model)

    if ($Options.AllowedTools) {
        $cmdArgs += "--allowedTools=Bash,Read,Write,Edit,Glob,Grep,WebSearch"
    }
    elseif ($Options.ReadOnly) {
        $cmdArgs += "--allowedTools=Read,Glob,Grep"
    }

    return @{
        Args = $cmdArgs
        PromptMethod = "stdin"
    }
}
