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

function Show-ResumePrompt {
    <#
    .SYNOPSIS
        Displays resume prompt for interrupted interview sessions
    .PARAMETER ExistingContext
        The context hashtable from Get-ExistingContext
    .RETURNS
        $true if user wants to resume, $false for new interview
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$ExistingContext
    )

    Write-Host "  Found interrupted session:" -ForegroundColor Yellow
    Write-Host ""

    # Display context (truncate if too long)
    $contextText = $ExistingContext.Context
    if ($contextText.Length -gt 200) {
        $contextText = $contextText.Substring(0, 197) + "..."
    }
    Write-Host "  Context: $contextText" -ForegroundColor Gray
    Write-Host ""

    # Show progress
    $completed = $ExistingContext.Completed
    $total = $ExistingContext.Total
    Write-Host "  Progress: $completed/$total focus areas complete" -ForegroundColor Cyan
    Write-Host ""

    # List remaining focus areas
    $remaining = $ExistingContext.FocusAreas | Where-Object { -not $_.completed }
    if ($remaining.Count -gt 0) {
        Write-Host "  Remaining focus areas:" -ForegroundColor White
        foreach ($area in $remaining) {
            $areaName = if ($area.name) { $area.name } else { $area.area }
            Write-Host "    - $areaName" -ForegroundColor Gray
        }
        Write-Host ""
    }

    # Prompt user
    Write-Host "  Resume this session? [Y]es / [N]ew interview" -ForegroundColor Yellow -NoNewline
    $response = Read-Host " "

    return ($response -match "^[Yy]")
}

# ============================================================================
# QUESTION FUNCTIONS
# ============================================================================

function Ask-WorkType {
    <#
    .SYNOPSIS
        Asks user what kind of work they want to do
    .RETURNS
        "bug", "feature", "improvement", or "client"
    #>

    Write-Host "  What kind of work?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [B] Bug fix" -ForegroundColor White
    Write-Host "  [F] Feature" -ForegroundColor White
    Write-Host "  [I] Improvement (default)" -ForegroundColor White
    Write-Host "  [C] Client feedback" -ForegroundColor White
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Bb]" { return "bug" }
        "^[Ff]" { return "feature" }
        "^[Cc]" { return "client" }
        default { return "improvement" }
    }
}

function Ask-Details {
    <#
    .SYNOPSIS
        Asks for details based on work type
    .PARAMETER WorkType
        The type of work (bug, feature, improvement, client)
    .RETURNS
        User's text input describing the work
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$WorkType
    )

    $prompts = @{
        "bug" = "Describe the bug (or paste error message)"
        "feature" = "What feature do you want?"
        "improvement" = "What should be improved?"
        "client" = "What was the client feedback?"
    }

    $prompt = $prompts[$WorkType]
    if (-not $prompt) {
        $prompt = "Describe what you need"
    }

    Write-Host ""
    Write-Host "  $prompt" -ForegroundColor Cyan
    Write-Host ""

    $response = Read-Host "  "
    return $response
}

function Ask-Area {
    <#
    .SYNOPSIS
        Asks which focus area to work on
    .PARAMETER WorkType
        The type of work
    .PARAMETER Details
        The work details (for context)
    .RETURNS
        User's input (can be empty string to let Ralph decide)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$WorkType,
        [Parameter(Mandatory=$true)]
        [string]$Details
    )

    Write-Host ""
    Write-Host "  Which area? (optional - press Enter to let Ralph decide)" -ForegroundColor Cyan

    # Show available areas from config if available
    if ($config -and $config.focusAreas) {
        Write-Host ""
        Write-Host "  Available areas:" -ForegroundColor Gray
        foreach ($area in $config.focusAreas) {
            $areaId = if ($area.id) { $area.id } else { $area }
            Write-Host "    - $areaId" -ForegroundColor Gray
        }
    }

    Write-Host ""
    $response = Read-Host "  "
    return $response
}

function Ask-Client {
    <#
    .SYNOPSIS
        Asks which client this work is for
    .RETURNS
        User's input (can be empty string to skip)
    #>

    Write-Host ""
    Write-Host "  Which client? (optional - press Enter to skip)" -ForegroundColor Cyan

    # Show known clients if clients.json exists
    $clientsFile = Join-Path $script:RalphDir "clients.json"
    if (Test-Path $clientsFile) {
        try {
            $clientsData = Get-Content $clientsFile -Raw | ConvertFrom-Json
            if ($clientsData.clients) {
                $clientNames = @()
                $clientsData.clients.PSObject.Properties | ForEach-Object {
                    if ($_.Name -ne "default") {
                        $clientNames += $_.Name
                    }
                }
                if ($clientNames.Count -gt 0) {
                    Write-Host ""
                    Write-Host "  Known clients: $($clientNames -join ', ')" -ForegroundColor Gray
                }
            }
        }
        catch {
            # Ignore parse errors
        }
    }

    Write-Host ""
    $response = Read-Host "  "
    return $response
}

function Ask-Priority {
    <#
    .SYNOPSIS
        Asks for work priority
    .RETURNS
        "high", "normal", or "low"
    #>

    Write-Host ""
    Write-Host "  Priority?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [H] High" -ForegroundColor White
    Write-Host "  [N] Normal (default)" -ForegroundColor White
    Write-Host "  [L] Low" -ForegroundColor White
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Hh]" { return "high" }
        "^[Ll]" { return "low" }
        default { return "normal" }
    }
}

# ============================================================================
# ENTRY POINT
# ============================================================================

# Entry point
Write-RalphHeader

# Check for existing session to resume
$existing = Get-ExistingContext
if ($existing.HasContext -and -not $Resume) {
    $shouldResume = Show-ResumePrompt -ExistingContext $existing
    if ($shouldResume) {
        Write-Host "  Resuming previous session..." -ForegroundColor Green
        $script:ResumeMode = $true
        $script:FocusAreas = $existing.FocusAreas | Where-Object { -not $_.completed }
    } else {
        Write-Host "  Starting new interview..." -ForegroundColor Cyan
        $script:ResumeMode = $false
    }
} elseif ($Resume -and $existing.HasContext) {
    # -Resume flag was passed explicitly
    Write-Host "  Resuming previous session (via -Resume flag)..." -ForegroundColor Green
    $script:ResumeMode = $true
    $script:FocusAreas = $existing.FocusAreas | Where-Object { -not $_.completed }
} else {
    # No existing context or starting fresh
    $script:ResumeMode = $false
}
