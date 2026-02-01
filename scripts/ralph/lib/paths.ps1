# scripts/ralph/lib/paths.ps1
# Centralized path definitions for Ralph Loop
#
# USAGE: Source this file FIRST in ralph.ps1, then use $script:Paths.*
#
# DESIGN: Provides backward-compatible path resolution.
# Resolve-RalphPath checks old location first for migration period.

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
        LibDir       = Join-Path $RalphDir "lib"

        # New organized directories
        ConfigDir    = Join-Path $RalphDir "config"
        StateDir     = Join-Path $RalphDir "state"
        SessionDir   = Join-Path $RalphDir "session"
        LogsDir      = Join-Path $RalphDir "logs"
        ArchiveDir   = Join-Path $RalphDir "archive"
        SprintsDir   = Join-Path $RalphDir "archive\sprints"
        SessionsDir  = Join-Path $RalphDir "archive\sessions"
        TestsDir     = Join-Path $RalphDir "tests"
        DocsDir      = Join-Path $RalphDir "docs"

        # Config files (static, rarely change)
        ConfigFile       = Join-Path $RalphDir "config\ralph-config.json"
        ClientsFile      = Join-Path $RalphDir "config\clients.json"
        FeedbackFile     = Join-Path $RalphDir "config\feedback.json"
        TestBaselineFile = Join-Path $RalphDir "config\test_baseline.json"

        # State files (mutable runtime state, persists across sessions)
        PrdFile              = Join-Path $RalphDir "state\prd.json"
        QueueFile            = Join-Path $RalphDir "state\queue.json"
        SprintHistoryFile    = Join-Path $RalphDir "state\sprint_history.json"
        StoryProgressFile    = Join-Path $RalphDir "state\story_progress.json"
        LearningDbFile       = Join-Path $RalphDir "state\learning_db.json"
        HealthMetricsFile    = Join-Path $RalphDir "state\health_metrics.json"
        HeartbeatFile        = Join-Path $RalphDir "state\heartbeat.json"
        LastRetrospectiveFile = Join-Path $RalphDir "state\last_retrospective.json"
        HardStoriesArchive   = Join-Path $RalphDir "state\hard_stories_archive.json"
        CrashRecoveryFile    = Join-Path $RalphDir "state\crash_recovery.json"
        HealingStateFile     = Join-Path $RalphDir "state\healing_state.json"

        # Session files (volatile, per-session, gitignored)
        PromptFile             = Join-Path $RalphDir "session\prompt.md"
        ExplorationContextFile = Join-Path $RalphDir "session\exploration_context.md"
        ProgressFile           = Join-Path $RalphDir "session\progress.txt"
        SessionLogFile         = Join-Path $RalphDir "session\session.log"
        HealingLogFile         = Join-Path $RalphDir "session\healing_log.jsonl"
        ApiTimeoutsFile        = Join-Path $RalphDir "session\api_timeouts.jsonl"
        RalphsChoicesLog       = Join-Path $RalphDir "session\ralphs_choices.log"
        MetricsFile            = Join-Path $RalphDir "session\metrics.csv"
        GracefulStopSignal     = Join-Path $RalphDir "session\graceful_stop.signal"
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
# BACKWARD COMPATIBILITY
# ============================================================================

function Resolve-RalphPath {
    <#
    .SYNOPSIS
        Resolve a path with backward compatibility fallback.
    .DESCRIPTION
        During migration, files may be in old or new locations.
        This function checks the new location first, then falls back to old.
    .PARAMETER PathKey
        Key from $script:Paths (e.g., 'PrdFile', 'QueueFile')
    .PARAMETER CreateIfMissing
        If true and neither location exists, create parent directory for new path
    .RETURNS
        Resolved path (new location if exists, else old location if exists, else new location)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$PathKey,
        [switch]$CreateIfMissing
    )

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    $newPath = $script:Paths[$PathKey]
    if (-not $newPath) {
        Write-Warning "Unknown path key: $PathKey"
        return $null
    }

    # Check new location first
    if (Test-Path $newPath) {
        return $newPath
    }

    # Build old path for fallback
    $oldPath = Get-LegacyPath -PathKey $PathKey
    if ($oldPath -and (Test-Path $oldPath)) {
        return $oldPath
    }

    # Neither exists - return new path
    if ($CreateIfMissing) {
        $parentDir = Split-Path -Parent $newPath
        if (-not (Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
        }
    }

    return $newPath
}

function Get-LegacyPath {
    <#
    .SYNOPSIS
        Get the legacy (old) path for a file during migration.
    .DESCRIPTION
        Maps path keys to their original root-level locations.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$PathKey
    )

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    $ralphDir = $script:Paths.RalphDir

    # Map path keys to their legacy locations (all in root)
    $legacyMap = @{
        # Config files (were in root)
        ConfigFile       = Join-Path $ralphDir "ralph-config.json"
        ClientsFile      = Join-Path $ralphDir "clients.json"
        FeedbackFile     = Join-Path $ralphDir "feedback.json"
        TestBaselineFile = Join-Path $ralphDir "test_baseline.json"

        # State files (were in root)
        PrdFile              = Join-Path $ralphDir "prd.json"
        QueueFile            = Join-Path $ralphDir "queue.json"
        SprintHistoryFile    = Join-Path $ralphDir "sprint_history.json"
        StoryProgressFile    = Join-Path $ralphDir "story_progress.json"
        LearningDbFile       = Join-Path $ralphDir "learning_db.json"
        HealthMetricsFile    = Join-Path $ralphDir "health_metrics.json"
        HeartbeatFile        = Join-Path $ralphDir "heartbeat.json"
        LastRetrospectiveFile = Join-Path $ralphDir "last_retrospective.json"
        HardStoriesArchive   = Join-Path $ralphDir "hard_stories_archive.json"
        CrashRecoveryFile    = Join-Path $ralphDir "crash_recovery.json"
        HealingStateFile     = Join-Path $ralphDir "healing_state.json"

        # Session files (were in root)
        PromptFile             = Join-Path $ralphDir "prompt.md"
        ExplorationContextFile = Join-Path $ralphDir "exploration_context.md"
        ProgressFile           = Join-Path $ralphDir "progress.txt"
        HealingLogFile         = Join-Path $ralphDir "healing_log.jsonl"
        ApiTimeoutsFile        = Join-Path $ralphDir "api_timeouts.jsonl"
        RalphsChoicesLog       = Join-Path $ralphDir "ralphs_choices.log"
        MetricsFile            = Join-Path $ralphDir "metrics.csv"
        GracefulStopSignal     = Join-Path $ralphDir "graceful_stop.signal"
    }

    return $legacyMap[$PathKey]
}

function Migrate-RalphFile {
    <#
    .SYNOPSIS
        Migrate a file from legacy location to new location.
    .PARAMETER PathKey
        Key from $script:Paths
    .PARAMETER Force
        Overwrite if destination exists
    .RETURNS
        $true if migrated, $false if no migration needed
    #>
    param(
        [Parameter(Mandatory)]
        [string]$PathKey,
        [switch]$Force
    )

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    $newPath = $script:Paths[$PathKey]
    $oldPath = Get-LegacyPath -PathKey $PathKey

    if (-not $oldPath -or -not (Test-Path $oldPath)) {
        return $false  # Nothing to migrate
    }

    if ((Test-Path $newPath) -and -not $Force) {
        return $false  # Already migrated
    }

    # Ensure parent directory exists
    $parentDir = Split-Path -Parent $newPath
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    # Move file
    Move-Item -Path $oldPath -Destination $newPath -Force
    Write-Host "  Migrated: $PathKey -> $newPath" -ForegroundColor DarkGray

    return $true
}

function Invoke-RalphMigration {
    <#
    .SYNOPSIS
        Migrate all files from legacy locations to new structure.
    .DESCRIPTION
        One-time migration script. Safe to run multiple times.
    #>

    if (-not $script:Paths) {
        Initialize-RalphPaths
    }

    $migratedCount = 0

    # Config files
    @('ConfigFile', 'ClientsFile', 'FeedbackFile', 'TestBaselineFile') | ForEach-Object {
        if (Migrate-RalphFile -PathKey $_) { $migratedCount++ }
    }

    # State files
    @('PrdFile', 'QueueFile', 'SprintHistoryFile', 'StoryProgressFile',
      'LearningDbFile', 'HealthMetricsFile', 'HeartbeatFile',
      'LastRetrospectiveFile', 'HardStoriesArchive', 'CrashRecoveryFile',
      'HealingStateFile') | ForEach-Object {
        if (Migrate-RalphFile -PathKey $_) { $migratedCount++ }
    }

    # Session files
    @('PromptFile', 'ExplorationContextFile', 'ProgressFile',
      'HealingLogFile', 'ApiTimeoutsFile', 'RalphsChoicesLog',
      'MetricsFile', 'GracefulStopSignal') | ForEach-Object {
        if (Migrate-RalphFile -PathKey $_) { $migratedCount++ }
    }

    if ($migratedCount -gt 0) {
        Write-Host "  Migration complete: $migratedCount files moved" -ForegroundColor Green
    }

    return $migratedCount
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
