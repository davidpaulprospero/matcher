# scripts/ralph/lib/compat.ps1
# Cross-platform compatibility utilities for Ralph
#
# This module provides cross-platform alternatives to Windows-specific commands.
# Source this at the start of any script that needs to run on both Windows and Linux.

# ============================================================================
# OS DETECTION
# ============================================================================

# Detect if running on Windows
function Get-IsWindows {
    if ($PSVersionTable.Platform -eq 'Win' -or $null -eq $32NTPSVersionTable.Platform) {
        return $true
    }
    return $false
}

# Detect if running on Linux
function Get-IsLinux {
    if ($PSVersionTable.Platform -eq 'Unix' -and (Test-Path '/proc/version') -and (Get-Content '/proc/version' -ErrorAction SilentlyContinue) -notmatch 'Microsoft') {
        return $true
    }
    return $false
}

# Detect if running on macOS
function Get-IsMacOS {
    if ($PSVersionTable.Platform -eq 'Unix' -and (Test-Path '/proc/version') -and (Get-Content '/proc/version' -ErrorAction SilentlyContinue) -match 'Darwin') {
        return $true
    }
    return $false
}

# Get the platform identifier
function Get-Platform {
    if (Get-IsWindows) { return 'windows' }
    if (Get-IsLinux) { return 'linux' }
    if (Get-IsMacOS) { return 'macos' }
    return 'unknown'
}

# ============================================================================
# PATH UTILITIES
# ============================================================================

# Get the home directory (cross-platform)
function Get-CrossPlatformHome {
    if (Get-IsWindows) {
        return $env:USERPROFILE
    }
    return $env:HOME
}

# Join path segments (cross-platform - uses / on all platforms for PowerShell Core)
function Join-CrossPlatformPath {
    param(
        [string[]]$PathSegments
    )

    # PowerShell Core handles / correctly on all platforms
    # Just normalize any backslashes to forward slashes
    $joined = Join-Path $PathSegments[0] $PathSegments[1]
    for ($i = 2; $i -lt $PathSegments.Count; $i++) {
        $joined = Join-Path $joined $PathSegments[$i]
    }

    # Normalize backslashes to forward slashes for cross-platform consistency
    return $joined -replace '\\', '/'
}

# Convert Windows path to cross-platform path
function ConvertTo-CrossPlatformPath {
    param(
        [string]$Path
    )

    return $Path -replace '\\', '/'
}

# ============================================================================
# ENVIRONMENT
# ============================================================================

# Get config file path (cross-platform)
function Get-MiniMaxEnvFile {
    $home = Get-CrossPlatformHome
    return Join-Path $home ".minimax.env"
}

# ============================================================================
# PROCESS STARTING
# ============================================================================

# Start a new process (cross-platform)
function Start-CrossPlatformProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [switch]$NoExit,
        [switch]$Wait
    )

    if (Get-IsWindows) {
        $args = @()
        if ($NoExit) {
            $args += "-NoExit"
        }
        $args += "-Command"
        $args += "Set-Location '$PWD'; & '$FilePath' $($ArgumentList -join ' ')"

        Start-Process powershell -ArgumentList $args -Wait:$Wait
    }
    else {
        # On Linux, just run the process directly
        $proc = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru -Wait:$Wait
        return $proc
    }
}

# ============================================================================
# FILE LOCKING (Cross-platform)
# ============================================================================

# Try to acquire a lock file (cross-platform)
function Test-FileLock {
    param(
        [string]$Path
    )

    try {
        $stream = [System.IO.File]::Open($Path, 'Open', 'ReadWrite', 'None')
        $stream.Close()
        return $false
    }
    catch {
        return $true
    }
}

# ============================================================================
# EXPORT MODULE MEMBERS
# ============================================================================

Export-ModuleMember -Function @(
    'Get-IsWindows',
    'Get-IsLinux',
    'Get-IsMacOS',
    'Get-Platform',
    'Get-CrossPlatformHome',
    'Join-CrossPlatformPath',
    'ConvertTo-CrossPlatformPath',
    'Get-MiniMaxEnvFile',
    'Start-CrossPlatformProcess',
    'Test-FileLock'
)
