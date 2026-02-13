$ErrorActionPreference = "Stop"
$content = Get-Content 'D:\_Projects\voiceover-matcher-subtitle\scripts\ralph\lib\interview.ps1' -Raw
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($content, [ref]$null, [ref]$errors)
if ($errors) {
    $errors | ForEach-Object { Write-Host $_.Message -ForegroundColor Red }
    exit 1
} else {
    Write-Host 'Syntax OK' -ForegroundColor Green
    exit 0
}
