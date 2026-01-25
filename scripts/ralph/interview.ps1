# Ralph Interview Mode
# Usage: .\scripts\ralph\interview.ps1 [-Resume]

param(
    [switch]$Resume
)

# Set up paths
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"

# Load config if it exists
$config = $null
if (Test-Path $script:ConfigFile) {
    $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
}

function Write-RalphHeader {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "   'Hi, Super Nintendo Chalmers!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Interview Mode" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-ExistingContext {
    <#
    .SYNOPSIS
        Checks if there's resumable interview context in queue.json
    .DESCRIPTION
        Returns a hashtable with:
        - HasContext: $true if resumable context exists
        - Context: The interview context string
        - FocusAreas: Array of focus area objects
        - Completed: Number of completed focus areas
        - Total: Total number of focus areas
    #>

    # Check if queue.json exists
    if (-not (Test-Path $script:QueueFile)) {
        return @{ HasContext = $false }
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Check for interviewContext
        if (-not $queue.interviewContext) {
            return @{ HasContext = $false }
        }

        # Check for focusAreas with incomplete items
        if (-not $queue.focusAreas -or $queue.focusAreas.Count -eq 0) {
            return @{ HasContext = $false }
        }

        # Count completed vs incomplete focus areas
        $completed = 0
        $incomplete = 0

        foreach ($area in $queue.focusAreas) {
            if ($area.completed -eq $true) {
                $completed++
            } else {
                $incomplete++
            }
        }

        # Only return context if there are incomplete items
        if ($incomplete -eq 0) {
            return @{ HasContext = $false }
        }

        return @{
            HasContext = $true
            Context = $queue.interviewContext
            FocusAreas = $queue.focusAreas
            Completed = $completed
            Total = $queue.focusAreas.Count
        }
    }
    catch {
        # If we can't parse the file, no resumable context
        return @{ HasContext = $false }
    }
}

# Entry point
Write-RalphHeader
