#!/bin/bash
# Ralph Loop - Linux Launcher
# Usage: ./ralph.sh [options]
#
# This launcher installs PowerShell Core if needed and runs Ralph on Linux.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RALPH_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for PowerShell Core
check_pwsh() {
    if command -v pwsh &> /dev/null; then
        return 0
    fi
    return 1
}

# Install PowerShell Core
install_pwsh() {
    echo -e "${YELLOW}PowerShell Core not found. Installing...${NC}"

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check if Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            # Install prerequisites
            sudo apt-get update
            sudo apt-get install -y wget apt-transport-https software-properties-common

            # Download and install Microsoft repository
            wget -q https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
            sudo dpkg -i packages-microsoft-prod.deb
            rm packages-microsoft-prod.deb

            sudo apt-get update
            sudo apt-get install -y powershell
        # Check if Arch
        elif command -v pacman &> /dev/null; then
            sudo pacman -S powershell-bin
        # Check if Fedora
        elif command -v dnf &> /dev/null; then
            sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
            sudo dnf install -y https://packages.microsoft.com/config/fedora/$(rpm -E %fedora)/packages-microsoft-prod-release.noarch.rpm
            sudo dnf update
            sudo dnf install -y powershell
        else
            echo -e "${RED}Unsupported Linux distribution. Please install PowerShell Core manually.${NC}"
            echo "See: https://aka.ms/powershell"
            exit 1
        fi
    fi

    if ! check_pwsh; then
        echo -e "${RED}Failed to install PowerShell Core. Please install manually.${NC}"
        exit 1
    fi

    echo -e "${GREEN}PowerShell Core installed successfully!${NC}"
}

# Set up environment for cross-platform operation
setup_environment() {
    export RALPH_DIR="$RALPH_DIR"
    export HOME_DIR="$HOME"

    # Create session directory if it doesn't exist
    mkdir -p "$RALPH_DIR/session"
    mkdir -p "$RALPH_DIR/logs"
}

# Run Ralph
run_ralph() {
    # Change to Ralph directory
    cd "$RALPH_DIR"

    # Build argument string for PowerShell
    local args=""
    for arg in "$@"; do
        args="$args $arg"
    done

    # Run PowerShell with the interview script
    # Note: Don't overwrite HOME on Linux - it's read-only
    pwsh -NoProfile -ExecutionPolicy Bypass -Command "
        # Cross-platform path setup
        \$env:RALPH_DIR = '$RALPH_DIR'

        # Load and execute the interview script
        & '$RALPH_DIR/interview.ps1' $args
    "
}

# Main
main() {
    echo -e "${GREEN}Ralph Loop - Linux Launcher${NC}"

    if ! check_pwsh; then
        echo -e "${RED}PowerShell Core (pwsh) is required but not installed.${NC}"
        echo ""
        echo "To install on Ubuntu/Debian:"
        echo "  1. wget -q https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb"
        echo "  2. sudo dpkg -i packages-microsoft-prod.deb"
        echo "  3. sudo apt update"
        echo "  4. sudo apt install powershell"
        echo ""
        echo "Or visit: https://aka.ms/powershell"
        echo ""
        echo "After installation, run this script again."
        exit 1
    fi

    setup_environment
    run_ralph "$@"
}

main "$@"
