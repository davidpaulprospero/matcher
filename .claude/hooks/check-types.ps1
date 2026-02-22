# check-types.ps1 - Run mypy on edited Python files (PostToolUse hook)

param()

# Read JSON input from stdin
$json = [System.Console]::In.ReadToEnd() | ConvertFrom-Json
$filePath = $json.tool_input.file_path

# Only process Python files
if (-not $filePath -or -not $filePath.EndsWith(".py")) {
    exit 0
}

# Verify file exists
if (-not (Test-Path $filePath)) {
    exit 0
}

# Run mypy on the edited file
$output = python -m mypy --ignore-missing-imports --no-error-summary $filePath 2>&1
$exitCode = $LASTEXITCODE

# Only show output if there are errors (non-zero exit)
if ($exitCode -ne 0 -and $output) {
    # Show first 15 lines of errors to avoid flooding
    $output | Select-Object -First 15 | ForEach-Object { Write-Host $_ }
    if (($output | Measure-Object).Count -gt 15) {
        Write-Host "... (truncated, run mypy manually for full output)"
    }
}

exit 0  # Don't block edits on type errors - informational only
