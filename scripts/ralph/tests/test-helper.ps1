# scripts/ralph/tests/test-helper.ps1
# Shared test helper for loading Ralph functions from modular lib/ files + ralph.ps1

function Import-RalphFunctions {
    param(
        [string]$RalphDir,
        [switch]$GlobalScope,
        [string[]]$OnlyFunctions
    )

    $libDir = Join-Path $RalphDir 'lib'
    $scripts = @()

    # Source lib modules first (order matters for dependencies)
    if (Test-Path $libDir) {
        $scripts += Get-ChildItem -Path $libDir -Filter '*.ps1' | ForEach-Object { $_.FullName }
    }

    # Then ralph.ps1 itself
    $ralphScript = Join-Path $RalphDir 'ralph.ps1'
    if (Test-Path $ralphScript) {
        $scripts += $ralphScript
    }

    foreach ($scriptPath in $scripts) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$null, [ref]$null)
        $functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

        foreach ($func in $functions) {
            if ($OnlyFunctions -and $OnlyFunctions -notcontains $func.Name) { continue }

            $funcDef = $func.Extent.Text
            if ($GlobalScope) {
                $globalFuncDef = $funcDef -replace '^function\s+([A-Za-z0-9_-]+)', 'function global:$1'
                Invoke-Expression $globalFuncDef
            } else {
                . ([scriptblock]::Create($funcDef))
            }
        }
    }
}
