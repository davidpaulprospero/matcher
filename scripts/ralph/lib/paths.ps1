# scripts/ralph/lib/paths.ps1
# Centralized path definitions for Ralph Loop
#
# USAGE: Source this file FIRST in ralph.ps1, then use $script:Paths.*
#
# Directory structure:
#   config/   - Static config files (ralph-config.json, clients.json)
#   state/    - Persistent state (prd.json, queue.json, sprint_history.json)
#   session/  - Volatile session data (prompt.md, metrics.csv, logs)

# ============================================================================
# PATH DEFINITIONS
# ============================================================================

function Initialize-RalphPaths {
    <#
    .SYNOPSIS
        Initialize all Ralph paths. Called once at script start.
    .DESCRIPTION
        Sets up $script:Paths hashtable with all path references.
        Automatically creates required directories if they don't exist.
    .PARAMETER RalphDir
        Base Ralph directory (defaults to script location)
    #>
    param(
        [string]$RalphDir = ""
    )

    if (-not $RalphDir) {
        # Determine RalphDir from caller context
        $RalphDir = if ($script:RalphDir) {
            $script:RalphDir
        } else {
            Split-Path -Parent $PSScriptRoot
        }
    }

    $script:Paths = @{
        # Base directories
        RalphDir     = $RalphDir
        ProjectRoot  = Split-Path -Parent (Split-Path -Parent $RalphDir)
        LibDir       = (Join-Path $RalphDir "lib") -replace '\\', '/'

        # New organized directories
        ConfigDir    = (Join-Path $RalphDir "config") -replace '\\', '/'
        StateDir     = (Join-Path $RalphDir "state") -replace '\\', '/'
        SessionDir   = (Join-Path $RalphDir "session") -replace '\\', '/'
        LogsDir      = (Join-Path $RalphDir "logs") -replace '\\', '/'
        ArchiveDir   = (Join-Path $RalphDir "archive") -replace '\\', '/'
        SprintsDir   = (Join-Path $RalphDir "archive/sprints") -replace '\\', '/'
        SessionsDir  = (Join-Path $RalphDir "archive/sessions") -replace '\\', '/'
        TestsDir     = (Join-Path $RalphDir "tests") -replace '\\', '/'
        DocsDir      = (Join-Path $RalphDir "docs") -replace '\\', '/'

        # Config files (static, rarely change)
        ConfigFile       = (Join-Path $RalphDir "config/ralph-config.json") -replace '\\', '/'
        ClientsFile      = (Join-Path $RalphDir "config/clients.json") -replace '\\', '/'
        FeedbackFile     = (Join-Path $RalphDir "config/feedback.json") -replace '\\', '/'
        TestBaselineFile = (Join-Path $RalphDir "config/test_baseline.json") -replace '\\', '/'

        # State files (mutable runtime state, persists across sessions)
        PrdFile              = (Join-Path $RalphDir "state/prd.json") -replace '\\', '/'
        QueueFile            = (Join-Path $RalphDir "state/queue.json") -replace '\\', '/'
        SprintHistoryFile    = (Join-Path $RalphDir "state/sprint_history.json") -replace '\\', '/'
        StoryProgressFile    = (Join-Path $RalphDir "state/story_progress.json") -replace '\\', '/'
        LearningDbFile       = (Join-Path $RalphDir "state/learning_db.json") -replace '\\', '/'
        HealthMetricsFile    = (Join-Path $RalphDir "state/health_metrics.json") -replace '\\', '/'
        HeartbeatFile        = (Join-Path $RalphDir "state/heartbeat.json") -replace '\\', '/'
        LastRetrospectiveFile = (Join-Path $RalphDir "state/last_retrospective.json") -replace '\\', '/'
        HardStoriesArchive   = (Join-Path $RalphDir "state/hard_stories_archive.json") -replace '\\', '/'
        CrashRecoveryFile    = (Join-Path $RalphDir "state/crash_recovery.json") -replace '\\', '/'
        HealingStateFile     = (Join-Path $RalphDir "state/healing_state.json") -replace '\\', '/'

        # Session files (volatile, per-session, gitignored)
        PromptFile             = (Join-Path $RalphDir "session/prompt.md") -replace '\\', '/'
        ExplorationContextFile = (Join-Path $RalphDir "session/exploration_context.md") -replace '\\', '/'
        ProgressFile           = (Join-Path $RalphDir "session/progress.txt") -replace '\\', '/'
        SessionLogFile         = (Join-Path $RalphDir "session/session.log") -replace '\\', '/'
        HealingLogFile         = (Join-Path $RalphDir "session/healing_log.jsonl") -replace '\\', '/'
        ApiTimeoutsFile        = (Join-Path $RalphDir "session/api_timeouts.jsonl") -replace '\\', '/'
        RalphsChoicesLog       = (Join-Path $RalphDir "session/ralphs_choices.log") -replace '\\', '/'
        MetricsFile            = (Join-Path $RalphDir "session/metrics.csv") -replace '\\', '/'
        SprintProgressFile     = (Join-Path $RalphDir "session/sprint_progress.md") -replace '\\', '/'
        GracefulStopSignal     = (Join-Path $RalphDir "session/graceful_stop.signal") -replace '\\', '/'
        PreStoryBaselineFile   = (Join-Path $RalphDir "session/pre_story_baseline.json") -replace '\\', '/'
    }

    # Ensure directories exist
    $dirsToCreate = @(
        $script:Paths.ConfigDir
        $script:Paths.StateDir
        $script:Paths.SessionDir
        $script:Paths.LogsDir
        $script:Paths.ArchiveDir
        $script:Paths.SprintsDir
        $script:Paths.SessionsDir
    )

    foreach ($dir in $dirsToCreate) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    return $script:Paths
}

# ============================================================================
# PATH RESOLUTION
# ============================================================================

function Resolve-RalphPath {
    <#
    .SYNOPSIS
        Resolve a path key to its full path.
    .PARAMETER PathKey
        Key from $script:Paths (e.g., 'PrdFile', 'QueueFile')
    .PARAMETER CreateIfMissing
        If true and path doesn't exist, create parent directory
    .RETURNS
        Full path for the given key
    #>
    param(
        [Parameter(Mandatory)]
        [string]$PathKey,
        [switch]$CreateIfMissing
    )

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    $path = $script:Paths[$PathKey]
    if (-not $path) {
        Write-Warning "Unknown path key: $PathKey"
        return $null
    }

    if ($CreateIfMissing -and -not (Test-Path $path)) {
        $parentDir = Split-Path -Parent $path
        if (-not (Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
        }
    }

    return $path
}

# ============================================================================
# CONVENIENCE ACCESSORS
# ============================================================================

function Get-RalphPath {
    <#
    .SYNOPSIS
        Get a path from the paths configuration.
    .DESCRIPTION
        Simple accessor for $script:Paths with automatic initialization.
    .PARAMETER Key
        Path key (e.g., 'PrdFile', 'ConfigFile')
    .PARAMETER Resolve
        If true, use Resolve-RalphPath for backward compatibility
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Key,
        [switch]$Resolve
    )

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    if ($Resolve) {
        return Resolve-RalphPath -PathKey $Key
    }

    return $script:Paths[$Key]
}

# Initialize paths on module load
Initialize-RalphPaths | Out-Null
