<#
.SYNOPSIS
    Run Ralph Loop test suite
.DESCRIPTION
    Executes Pester tests for the Ralph Loop PowerShell scripts.
    Supports filtering by tags and generating various output formats.
.PARAMETER Tag
    Filter tests by tag. Valid tags: Unit, Integration, Pure, Logging, Metrics, etc.
.PARAMETER ExcludeTag
    Exclude tests with these tags
.PARAMETER Output
    Output format: 'Minimal', 'Normal', 'Detailed', 'Diagnostic'
.PARAMETER CI
    Run in CI mode with JUnit output
.PARAMETER Coverage
    Generate code coverage report
.EXAMPLE
    .\Run-Tests.ps1
    Run all tests with normal output
.EXAMPLE
    .\Run-Tests.ps1 -Tag Unit
    Run only unit tests
.EXAMPLE
    .\Run-Tests.ps1 -Tag Integration -Output Detailed
    Run integration tests with detailed output
.EXAMPLE
    .\Run-Tests.ps1 -CI
    Run all tests with JUnit XML output for CI
#>

param(
    [string[]]$Tag,
    [string[]]$ExcludeTag,
    [ValidateSet('Minimal', 'Normal', 'Detailed', 'Diagnostic')]
    [string]$Output = 'Normal',
    [switch]$CI,
    [switch]$Coverage
)

$ErrorActionPreference = 'Stop'

# Check for Pester
$pester = Get-Module -ListAvailable -Name Pester | Where-Object { $_.Version -ge '5.0.0' }
if (-not $pester) {
    Write-Host "Pester 5.0+ is required. Installing..." -ForegroundColor Yellow
    Install-Module -Name Pester -Force -Scope CurrentUser -MinimumVersion 5.0.0
}

Import-Module Pester -MinimumVersion 5.0.0

# Test paths
$testPath = $PSScriptRoot
$outputPath = Join-Path $testPath "results"

# Create output directory
if (-not (Test-Path $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
}

# Build Pester configuration
$config = New-PesterConfiguration

# Test discovery
$config.Run.Path = $testPath
$config.Run.Exit = $true

# Filtering
if ($Tag) {
    $config.Filter.Tag = $Tag
}
if ($ExcludeTag) {
    $config.Filter.ExcludeTag = $ExcludeTag
}

# Output settings
$config.Output.Verbosity = $Output

# CI mode
if ($CI) {
    $config.TestResult.Enabled = $true
    $config.TestResult.OutputFormat = 'JUnitXml'
    $config.TestResult.OutputPath = Join-Path $outputPath "test-results.xml"
    $config.Output.Verbosity = 'Detailed'
}

# Coverage
if ($Coverage) {
    $config.CodeCoverage.Enabled = $true
    $config.CodeCoverage.Path = @(
        (Join-Path (Split-Path $testPath -Parent) "ralph.ps1"),
        (Join-Path (Split-Path $testPath -Parent) "watch.ps1"),
        (Join-Path (Split-Path $testPath -Parent) "interview.ps1")
    )
    $config.CodeCoverage.OutputPath = Join-Path $outputPath "coverage.xml"
    $config.CodeCoverage.OutputFormat = 'JaCoCo'
}

# Run timestamp
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  RALPH LOOP TEST SUITE" -ForegroundColor Cyan
Write-Host "  $timestamp" -ForegroundColor DarkGray
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

if ($Tag) {
    Write-Host "  Tags: $($Tag -join ', ')" -ForegroundColor Yellow
}
if ($ExcludeTag) {
    Write-Host "  Excluding: $($ExcludeTag -join ', ')" -ForegroundColor Yellow
}
Write-Host ""

# Run tests
$result = Invoke-Pester -Configuration $config

# Summary
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  TEST RESULTS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$passedColor = if ($result.PassedCount -gt 0) { 'Green' } else { 'Gray' }
$failedColor = if ($result.FailedCount -gt 0) { 'Red' } else { 'Gray' }
$skippedColor = if ($result.SkippedCount -gt 0) { 'Yellow' } else { 'Gray' }

Write-Host "  Passed:  $($result.PassedCount)" -ForegroundColor $passedColor
Write-Host "  Failed:  $($result.FailedCount)" -ForegroundColor $failedColor
Write-Host "  Skipped: $($result.SkippedCount)" -ForegroundColor $skippedColor
Write-Host "  Total:   $($result.TotalCount)" -ForegroundColor White
Write-Host ""
Write-Host "  Duration: $([math]::Round($result.Duration.TotalSeconds, 2)) seconds" -ForegroundColor DarkGray
Write-Host ""

if ($CI) {
    Write-Host "  JUnit XML: $($config.TestResult.OutputPath.Value)" -ForegroundColor DarkGray
}
if ($Coverage) {
    Write-Host "  Coverage:  $($config.CodeCoverage.OutputPath.Value)" -ForegroundColor DarkGray
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Exit with appropriate code
if ($result.FailedCount -gt 0) {
    exit 1
}
exit 0
