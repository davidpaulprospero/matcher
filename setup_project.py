#!/usr/bin/env python3
r"""
Project Setup Script for Voiceover-Matcher

Creates a new project folder with interactive folder selection.

Usage:
    python setup_project.py                    # Interactive mode
    python setup_project.py "E:\Full\Path"     # Direct path mode
    python setup_project.py --regenerate "E:\Project\Path"  # Fix existing run.bat
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


# Get the directory where this script lives (central install location)
INSTALL_DIR = Path(__file__).parent.resolve()

# Default base path for projects
DEFAULT_BASE_PATH = r"E:\Edit Job"

# Settings file to remember last used path
SETTINGS_FILE = INSTALL_DIR / ".project_settings.json"


def load_settings() -> dict:
    """Load saved settings (last used path, etc.)"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_settings(settings: dict):
    """Save settings for next run"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except:
        pass  # Non-critical


def get_install_dir() -> Path:
    """Get the central installation directory"""
    return INSTALL_DIR


def list_folders(path: Path) -> list:
    """List subdirectories in a path"""
    if not path.exists():
        return []
    
    folders = []
    try:
        for item in sorted(path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                folders.append(item.name)
    except PermissionError:
        pass
    return folders


def select_folder(base_path: Path, level_name: str = "folder") -> Path:
    """Interactive folder selection with option to create new"""
    print(f"\n  Current: {base_path}")
    
    folders = list_folders(base_path)
    
    if folders:
        print(f"\n  Available {level_name}s:")
        for i, folder in enumerate(folders, 1):
            print(f"    {i}. {folder}")
        print(f"    {len(folders) + 1}. [Create new {level_name}]")
        print(f"    0. [Use this folder]")
        
        while True:
            try:
                choice = input(f"\n  Select [0-{len(folders) + 1}]: ").strip()
                
                if not choice:
                    continue
                
                choice_num = int(choice)
                
                if choice_num == 0:
                    return base_path
                elif 1 <= choice_num <= len(folders):
                    return base_path / folders[choice_num - 1]
                elif choice_num == len(folders) + 1:
                    # Create new folder
                    new_name = input(f"  New {level_name} name: ").strip()
                    if new_name:
                        new_path = base_path / new_name
                        new_path.mkdir(parents=True, exist_ok=True)
                        print(f"  Created: {new_name}")
                        return new_path
                else:
                    print("  Invalid choice, try again.")
            except ValueError:
                # Maybe they typed a folder name directly
                if choice in folders:
                    return base_path / choice
                print("  Invalid input, enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                sys.exit(0)
    else:
        print(f"\n  No {level_name}s found. Create one?")
        new_name = input(f"  New {level_name} name (or Enter to use current): ").strip()
        if new_name:
            new_path = base_path / new_name
            new_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {new_name}")
            return new_path
        return base_path


def sanitize_name(name: str) -> str:
    """Sanitize project name for filesystem"""
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


def create_run_bat(project_dir: Path, install_dir: Path) -> Path:
    """
    Create Windows batch file to run the matcher.

    Includes checkpoint & saved keywords support (v3.2):
    - run                     Auto-detect: prompt if saves exist, fresh if not
    - run --resume            Resume interrupted run from checkpoint
    - run --fresh             Fresh start, ignore checkpoint
    - run --use-keywords      Use most recent saved keywords
    - run --match-only        Skip to matching stage using cached data
    - run --list              List saved keyword presets
    """
    bat_content = f'''@echo off
REM ============================================================
REM Voiceover-Matcher Runner
REM Project: {project_dir.name}
REM Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
REM ============================================================

REM Set FFmpeg path
set IMAGEIO_FFMPEG_EXE=C:\\ffmpeg\\bin\\ffmpeg.exe

REM Central installation location
set INSTALL_DIR={install_dir}

REM Project directory (this folder)
set PROJECT_DIR=%~dp0
if "%PROJECT_DIR:~-1%"=="\\" set PROJECT_DIR=%PROJECT_DIR:~0,-1%

REM Change to install directory
cd /d "%INSTALL_DIR%"

REM ============================================================
REM EXPLICIT COMMANDS (skip auto-detection)
REM ============================================================

if "%~1"=="--list" (
    python main.py --project "%PROJECT_DIR%" --list-keywords
    goto :done
)

if "%~1"=="--help" (
    echo.
    echo   VOICEOVER-MATCHER - Commands:
    echo   ------------------------------------------------------------
    echo   run                     Auto-detect [prompt if saves exist]
    echo   run --resume            Resume from checkpoint
    echo   run --fresh             Fresh start [new keywords]
    echo   run --use-keywords      Use saved keywords
    echo   run --match-only        Skip to matching [use cached data]
    echo   run --list              List saved keyword presets
    echo   run --help              Show this help
    echo.
    goto :done
)

if "%~1"=="--resume" (
    echo   Mode: RESUME from checkpoint
    python main.py --project "%PROJECT_DIR%" --resume
    goto :check_error
)

if "%~1"=="--fresh" (
    echo   Mode: FRESH start
    python main.py --project "%PROJECT_DIR%" --fresh --save-keywords
    goto :check_error
)

if "%~1"=="--use-keywords" (
    if "%~2"=="" (
        echo   Mode: Using LATEST saved keywords
        python main.py --project "%PROJECT_DIR%" --use-keywords
    ) else (
        echo   Mode: Using saved keywords [%~2]
        python main.py --project "%PROJECT_DIR%" --use-keywords %~2
    )
    goto :check_error
)

if "%~1"=="--match-only" (
    echo   Mode: MATCH ONLY [skip download/transcribe]
    python main.py --project "%PROJECT_DIR%" --match-only --use-keywords
    goto :check_error
)

REM If we got here with an argument, it's unknown
if not "%~1"=="" (
    echo   Unknown option: %~1
    echo   Use: run --help
    goto :done
)

REM ============================================================
REM AUTO-DETECTION (no arguments)
REM ============================================================

echo.
echo   ============================================================
echo   VOICEOVER-MATCHER
echo   ============================================================
echo   Project: %PROJECT_DIR%
echo.

REM Check what exists
set HAS_CHECKPOINT=0
set HAS_KEYWORDS=0
set HAS_CACHE=0

if exist "%PROJECT_DIR%\\checkpoint.json" set HAS_CHECKPOINT=1
if exist "%PROJECT_DIR%\\saved_keywords.json" set HAS_KEYWORDS=1

REM Check for cached transcriptions and embeddings (fast - just check directories exist)
if exist "%PROJECT_DIR%\\.cache\\transcriptions" if exist "%PROJECT_DIR%\\.cache\\embeddings" set HAS_CACHE=1

REM If NOTHING saved, run fresh automatically (no prompt)
if %HAS_CHECKPOINT%==0 if %HAS_KEYWORDS%==0 if %HAS_CACHE%==0 (
    echo   No saves found - starting fresh run...
    echo.
    python main.py --project "%PROJECT_DIR%" --save-keywords
    goto :check_error
)

REM Something exists - prompt user
echo   SAVED DATA FOUND:
if %HAS_CHECKPOINT%==1 echo     - Checkpoint [resume interrupted run]
if %HAS_KEYWORDS%==1 echo     - Saved keywords [reuse for same videos]
if %HAS_CACHE%==1 echo     - Transcription and embedding caches available
echo.
echo   Options:
if %HAS_CHECKPOINT%==1 echo     [R] Resume from checkpoint
if %HAS_KEYWORDS%==1 echo     [K] Use saved keywords
if %HAS_CACHE%==1 (
    echo     [M] Match only [skip to matching stage]
) else (
    echo     [M] Match only [unavailable - missing cached data]
)
echo     [F] Fresh start [new keywords]
echo     [Q] Quit
echo.

:ask
set /p "CHOICE=  Your choice: "
if /i "%CHOICE%"=="R" goto :do_resume
if /i "%CHOICE%"=="K" goto :do_keywords
if /i "%CHOICE%"=="M" goto :do_match_only
if /i "%CHOICE%"=="F" goto :do_fresh
if /i "%CHOICE%"=="Q" goto :done
echo   Invalid choice. Enter R, K, M, F, or Q.
goto :ask

:do_resume
if %HAS_CHECKPOINT%==0 (
    echo   No checkpoint found!
    goto :ask
)
echo   Mode: RESUME
python main.py --project "%PROJECT_DIR%" --resume
goto :check_error

:do_keywords
if %HAS_KEYWORDS%==0 (
    echo   No saved keywords found!
    goto :ask
)
echo   Mode: Using saved keywords
python main.py --project "%PROJECT_DIR%" --use-keywords
goto :check_error

:do_match_only
if %HAS_CACHE%==0 (
    echo   Cannot use Match only - cached data is missing!
    echo   Run a full pipeline first to build transcription/embedding caches.
    goto :ask
)
echo   Mode: MATCH ONLY
python main.py --project "%PROJECT_DIR%" --match-only --use-keywords
goto :check_error

:do_fresh
echo   Mode: FRESH start
python main.py --project "%PROJECT_DIR%" --fresh --save-keywords
goto :check_error

:check_error
echo.
if %ERRORLEVEL% NEQ 0 (
    echo   ============================================================
    echo   [ERROR] Pipeline failed - code %ERRORLEVEL%
    echo   ============================================================
    echo   Tip: run --resume to continue
) else (
    echo   ============================================================
    echo   [SUCCESS] Pipeline completed
    echo   ============================================================
)

:done
echo.
exit /b
'''
    
    bat_path = project_dir / "run.bat"
    bat_path.write_text(bat_content)
    return bat_path


def create_run_sh(project_dir: Path, install_dir: Path) -> Path:
    """Create Unix shell script to run the matcher with checkpoint support (v3.2)"""
    sh_content = f'''#!/bin/bash
# ============================================================
# Voiceover-Matcher Runner
# Project: {project_dir.name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# ============================================================
#
# Usage:
#   ./run.sh                     Normal run (saves keywords automatically)
#   ./run.sh --resume            Resume interrupted run from checkpoint
#   ./run.sh --fresh             Fresh start, ignore checkpoint
#   ./run.sh --use-keywords      Use most recent saved keywords
#   ./run.sh --use-keywords NAME Use specific saved keyword preset
#   ./run.sh --match-only        Skip to matching using cached data
#   ./run.sh --list              List saved keyword presets
#   ./run.sh --help              Show all options
#
# ============================================================

# Central installation location (where main.py lives)
INSTALL_DIR="{install_dir}"

# Project directory (this folder)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Change to install directory
cd "$INSTALL_DIR"

# ============================================================
# Handle special commands
# ============================================================

case "$1" in
    --list)
        echo ""
        echo "  Listing saved keyword presets..."
        echo ""
        python main.py --project "$PROJECT_DIR" --list-keywords
        exit 0
        ;;
    --help)
        echo ""
        echo "  ============================================================"
        echo "  VOICEOVER-MATCHER - Quick Commands"
        echo "  ============================================================"
        echo ""
        echo "  ./run.sh                     Normal run (saves keywords)"
        echo "  ./run.sh --resume            Resume from checkpoint"
        echo "  ./run.sh --fresh             Fresh start, ignore checkpoint"
        echo "  ./run.sh --use-keywords      Use most recent saved keywords"
        echo "  ./run.sh --use-keywords NAME Use specific keyword preset"
        echo "  ./run.sh --match-only        Skip to matching [use cached data]"
        echo "  ./run.sh --list              List saved keyword presets"
        echo ""
        python main.py --help
        exit 0
        ;;
    --resume)
        echo ""
        echo "  Mode: RESUME from checkpoint"
        echo ""
        python main.py --project "$PROJECT_DIR" --resume
        EXIT_CODE=$?
        ;;
    --fresh)
        echo ""
        echo "  Mode: FRESH start (ignoring checkpoint)"
        echo ""
        python main.py --project "$PROJECT_DIR" --fresh --save-keywords
        EXIT_CODE=$?
        ;;
    --use-keywords)
        echo ""
        if [ -z "$2" ]; then
            echo "  Mode: Using LATEST saved keywords"
            echo ""
            python main.py --project "$PROJECT_DIR" --use-keywords
        else
            echo "  Mode: Using saved keywords [$2]"
            echo ""
            python main.py --project "$PROJECT_DIR" --use-keywords "$2"
        fi
        EXIT_CODE=$?
        ;;
    --match-only)
        echo ""
        echo "  Mode: MATCH ONLY [skip download/transcribe]"
        echo ""
        python main.py --project "$PROJECT_DIR" --match-only --use-keywords
        EXIT_CODE=$?
        ;;
    "")
        # No arguments - show interactive menu
        echo ""
        echo "  ============================================================"
        echo "  VOICEOVER-MATCHER"
        echo "  ============================================================"
        echo "  Project: $PROJECT_DIR"
        echo ""

        # Check what exists
        HAS_CHECKPOINT=0
        HAS_KEYWORDS=0
        HAS_CACHE=0

        [ -f "$PROJECT_DIR/checkpoint.json" ] && HAS_CHECKPOINT=1
        [ -f "$PROJECT_DIR/saved_keywords.json" ] && HAS_KEYWORDS=1

        # Check for cached transcriptions and embeddings (fast - just check directories exist)
        [ -d "$PROJECT_DIR/.cache/transcriptions" ] && [ -d "$PROJECT_DIR/.cache/embeddings" ] && HAS_CACHE=1

        # If nothing saved, run fresh automatically
        if [ $HAS_CHECKPOINT -eq 0 ] && [ $HAS_KEYWORDS -eq 0 ] && [ $HAS_CACHE -eq 0 ]; then
            echo "  No saves found - starting fresh run..."
            echo ""
            python main.py --project "$PROJECT_DIR" --save-keywords
            EXIT_CODE=$?
        else
            # Show menu
            echo "  SAVED DATA FOUND:"
            [ $HAS_CHECKPOINT -eq 1 ] && echo "    - Checkpoint [resume interrupted run]"
            [ $HAS_KEYWORDS -eq 1 ] && echo "    - Saved keywords [reuse for same videos]"
            [ $HAS_CACHE -eq 1 ] && echo "    - Transcription and embedding caches available"
            echo ""
            echo "  Options:"
            [ $HAS_CHECKPOINT -eq 1 ] && echo "    [R] Resume from checkpoint"
            [ $HAS_KEYWORDS -eq 1 ] && echo "    [K] Use saved keywords"
            if [ $HAS_CACHE -eq 1 ]; then
                echo "    [M] Match only [skip to matching stage]"
            else
                echo "    [M] Match only [unavailable - missing cached data]"
            fi
            echo "    [F] Fresh start [new keywords]"
            echo "    [Q] Quit"
            echo ""

            while true; do
                read -p "  Your choice: " CHOICE
                case "$CHOICE" in
                    [Rr])
                        if [ $HAS_CHECKPOINT -eq 0 ]; then
                            echo "  No checkpoint found!"
                        else
                            echo "  Mode: RESUME"
                            python main.py --project "$PROJECT_DIR" --resume
                            EXIT_CODE=$?
                            break
                        fi
                        ;;
                    [Kk])
                        if [ $HAS_KEYWORDS -eq 0 ]; then
                            echo "  No saved keywords found!"
                        else
                            echo "  Mode: Using saved keywords"
                            python main.py --project "$PROJECT_DIR" --use-keywords
                            EXIT_CODE=$?
                            break
                        fi
                        ;;
                    [Mm])
                        if [ $HAS_CACHE -eq 0 ]; then
                            echo "  Cannot use Match only - cached data is missing!"
                            echo "  Run a full pipeline first to build transcription/embedding caches."
                        else
                            echo "  Mode: MATCH ONLY"
                            python main.py --project "$PROJECT_DIR" --match-only --use-keywords
                            EXIT_CODE=$?
                            break
                        fi
                        ;;
                    [Ff])
                        echo "  Mode: FRESH start"
                        python main.py --project "$PROJECT_DIR" --fresh --save-keywords
                        EXIT_CODE=$?
                        break
                        ;;
                    [Qq])
                        echo ""
                        exit 0
                        ;;
                    *)
                        echo "  Invalid choice. Enter R, K, M, F, or Q."
                        ;;
                esac
            done
        fi
        ;;
    *)
        echo ""
        echo "  Unknown option: $1"
        echo "  Use: ./run.sh --help"
        exit 1
        ;;
esac

# Check exit code
if [ "${{EXIT_CODE:-0}}" -ne 0 ]; then
    echo ""
    echo "  [ERROR] Pipeline failed"
    echo ""
    echo "  If interrupted, you can resume with: ./run.sh --resume"
else
    echo ""
    echo "  [SUCCESS] Pipeline completed"
fi
'''
    
    sh_path = project_dir / "run.sh"
    sh_path.write_text(sh_content)
    sh_path.chmod(0o755)
    return sh_path


def create_analyze_bat(project_dir: Path, install_dir: Path) -> Path:
    """
    Create analyze.bat for post-edit timeline analysis.

    Drag-drop an exported XML/OTIO timeline file to analyze which clips were kept,
    replaced, moved, or disabled. Useful for understanding editor preferences.
    """
    bat_content = f'''@echo off
REM ============================================================
REM Post-Edit Timeline Analyzer
REM Project: {project_dir.name}
REM Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
REM ============================================================
REM
REM Usage: Drag and drop an exported XML or OTIO file onto this script
REM        to analyze which clips were selected in the final edit.
REM
REM Output shows:
REM   - Clips kept (V1, original recommendations)
REM   - Clips replaced with alternatives (V2+)
REM   - Clips replaced with external footage
REM   - Clips moved from other segments
REM   - Disabled segments
REM ============================================================

REM Central installation location
set INSTALL_DIR={install_dir}

REM Project directory (this folder)
set PROJECT_DIR=%~dp0
if "%PROJECT_DIR:~-1%"=="\\" set PROJECT_DIR=%PROJECT_DIR:~0,-1%

:: Check if a file was dragged onto the script
if "%~1"=="" (
    echo.
    echo   ============================================================
    echo   POST-EDIT TIMELINE ANALYZER
    echo   ============================================================
    echo.
    echo   Usage: Drag an exported XML or OTIO file onto this script
    echo.
    echo   This analyzes your edited timeline to understand:
    echo     - Which clips you kept vs replaced
    echo     - Which alternatives (V2+) you preferred
    echo     - External clips you added
    echo     - Clips moved between segments
    echo     - Segments where all clips were disabled
    echo.
    echo   Workflow:
    echo     1. Export timeline from DaVinci as XML or OTIO
    echo     2. Drag the file onto this script
    echo     3. Review the analysis summary
    echo.
    pause
    exit /b
)

:: Input file
set INPUT=%~1
set INPUT_EXT=%~x1

:: Validate extension
if /i not "%INPUT_EXT%"==".xml" if /i not "%INPUT_EXT%"==".otio" if /i not "%INPUT_EXT%"==".fcpxml" (
    echo.
    echo   ERROR: Unsupported file type: %INPUT_EXT%
    echo   Supported formats: .xml, .fcpxml, .otio
    echo.
    pause
    exit /b 1
)

echo.
echo   ============================================================
echo   POST-EDIT TIMELINE ANALYZER
echo   ============================================================
echo.
echo   Project:  %PROJECT_DIR%
echo   Timeline: %~nx1
echo.
echo   Analyzing...
echo.

REM Change to install directory and run analysis
cd /d "%INSTALL_DIR%"

python -c "from src.post_edit_analysis import analyze_final_edit; analyze_final_edit(r'%INPUT%')"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [ERROR] Analysis failed - check file format
) else (
    echo.
    echo   [SUCCESS] Analysis complete
)

echo.
pause
'''

    bat_path = project_dir / "analyze.bat"
    bat_path.write_text(bat_content)
    return bat_path


def create_analyze_sh(project_dir: Path, install_dir: Path) -> Path:
    """
    Create analyze.sh for post-edit timeline analysis on Unix systems.

    Usage: ./analyze.sh timeline.xml
    """
    sh_content = f'''#!/bin/bash
# ============================================================
# Post-Edit Timeline Analyzer
# Project: {project_dir.name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# ============================================================
#
# Usage: ./analyze.sh <timeline.xml or timeline.otio>
#
# Analyzes which clips were selected in the final edit.
# ============================================================

# Central installation location
INSTALL_DIR="{install_dir}"

# Project directory (this folder)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if a file was provided
if [ -z "$1" ]; then
    echo ""
    echo "  ============================================================"
    echo "  POST-EDIT TIMELINE ANALYZER"
    echo "  ============================================================"
    echo ""
    echo "  Usage: ./analyze.sh <timeline.xml or timeline.otio>"
    echo ""
    echo "  This analyzes your edited timeline to understand:"
    echo "    - Which clips you kept vs replaced"
    echo "    - Which alternatives (V2+) you preferred"
    echo "    - External clips you added"
    echo "    - Clips moved between segments"
    echo "    - Segments where all clips were disabled"
    echo ""
    exit 0
fi

INPUT="$1"
EXT="${{INPUT##*.}}"

# Validate extension
if [[ ! "$EXT" =~ ^(xml|otio|fcpxml)$ ]]; then
    echo ""
    echo "  ERROR: Unsupported file type: .$EXT"
    echo "  Supported formats: .xml, .fcpxml, .otio"
    echo ""
    exit 1
fi

echo ""
echo "  ============================================================"
echo "  POST-EDIT TIMELINE ANALYZER"
echo "  ============================================================"
echo ""
echo "  Project:  $PROJECT_DIR"
echo "  Timeline: $(basename "$INPUT")"
echo ""
echo "  Analyzing..."
echo ""

# Change to install directory and run analysis
cd "$INSTALL_DIR"

python -c "from src.post_edit_analysis import analyze_final_edit; analyze_final_edit('$INPUT')"

if [ $? -ne 0 ]; then
    echo ""
    echo "  [ERROR] Analysis failed - check file format"
else
    echo ""
    echo "  [SUCCESS] Analysis complete"
fi
'''

    sh_path = project_dir / "analyze.sh"
    sh_path.write_text(sh_content)
    sh_path.chmod(0o755)
    return sh_path


def create_convert_bat(project_dir: Path) -> Path:
    """
    Create convert.bat for converting HEVC/CapCut videos to DaVinci-compatible ProRes.

    Drag-drop any video file onto this batch file to convert it to ProRes 4444.
    Useful for CapCut exports that use HEVC codec which DaVinci may not handle well.
    """
    bat_content = '''@echo off
REM ============================================================
REM Video Converter for DaVinci Resolve
REM Converts HEVC/H.265 videos to ProRes 4444 (DaVinci compatible)
REM Usage: Drag and drop a video file onto this script
REM ============================================================

:: Check if a file was dragged onto the script
if "%~1"=="" (
    echo.
    echo   ============================================================
    echo   VIDEO CONVERTER FOR DAVINCI RESOLVE
    echo   ============================================================
    echo.
    echo   Usage: Drag a video file onto this script to convert it
    echo          to ProRes 4444 format compatible with DaVinci Resolve.
    echo.
    echo   This is useful for:
    echo     - CapCut exports (HEVC/H.265)
    echo     - iPhone recordings
    echo     - Any video that shows artifacts in DaVinci
    echo.
    pause
    exit /b
)

:: Set FFmpeg path
set FFMPEG=C:\\ffmpeg\\bin\\ffmpeg.exe

:: Check if FFmpeg exists
if not exist "%FFMPEG%" (
    echo   ERROR: FFmpeg not found at %FFMPEG%
    echo   Please install FFmpeg or update the path in this script.
    pause
    exit /b 1
)

:: Input file
set INPUT=%~1
set INPUT_DIR=%~dp1
set INPUT_NAME=%~n1
set INPUT_EXT=%~x1

:: Output file (same directory, _prores suffix)
set OUTPUT=%INPUT_DIR%%INPUT_NAME%_prores.mov

echo.
echo   ============================================================
echo   VIDEO CONVERTER
echo   ============================================================
echo.
echo   Input:  %INPUT%
echo   Output: %OUTPUT%
echo.
echo   Converting to ProRes 4444...
echo.

:: Convert to ProRes 4444
"%FFMPEG%" -i "%INPUT%" -c:v prores_ks -profile:v 4 -c:a pcm_s16le -y "%OUTPUT%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo   SUCCESS! Converted file saved to:
    echo   %OUTPUT%
) else (
    echo.
    echo   ERROR: Conversion failed.
)

echo.
pause
'''
    
    bat_path = project_dir / "convert.bat"
    bat_path.write_text(bat_content)
    return bat_path


def create_project_config(project_dir: Path) -> Path:
    """Create project-specific config that overrides central config"""
    config_content = f'''# ============================================================
# Project-Specific Configuration
# Project: {project_dir.name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# ============================================================
#
# This file overrides settings from the central config.yaml
# Only include settings you want to change for this project.
#
# The central config is at: {INSTALL_DIR / 'config.yaml'}
# ============================================================

# Project identification
project:
  name: "{project_dir.name}"
  
# Uncomment and modify any settings you want to override:

# keywords:
#   num_keywords: 20
#   tier_config:
#     short:
#       per_keyword: 1
#     medium:
#       per_keyword: 1
#     long:
#       per_keyword: 1
#     longer:
#       per_keyword: 0  # Disabled for speed

# enhanced:
#   enabled: true
#   min_confidence: 0.70
#   non_interactive: true  # Skip prompts, use defaults

# image_search:
#   enabled: true
#   max_entities: 2

# pipeline:
#   skip_download: false
#   skip_image_search: false
'''
    
    config_path = project_dir / "project_config.yaml"
    config_path.write_text(config_content)
    return config_path


def create_project_structure(project_dir: Path) -> dict:
    """Create standard project folder structure"""
    folders = {
        'voiceover': project_dir / 'voiceover',
        'output': project_dir / 'output',
        'logs': project_dir / 'logs',
        '.cache': project_dir / '.cache',
        '.cache/transcriptions': project_dir / '.cache' / 'transcriptions',
        '.cache/embeddings': project_dir / '.cache' / 'embeddings',
        '.cache/scenes': project_dir / '.cache' / 'scenes',
        '.cache/keyframes': project_dir / '.cache' / 'keyframes',
        '.cache/audio': project_dir / '.cache' / 'audio',
        '.cache/index': project_dir / '.cache' / 'index',
        '.cache/llm_responses': project_dir / '.cache' / 'llm_responses',
    }
    
    for name, path in folders.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return folders


def setup_project(project_path: str, install_dir: Path = None):
    """Set up a new project with all necessary files and folders"""
    project_dir = Path(project_path).resolve()
    
    if install_dir is None:
        install_dir = get_install_dir()
    
    # Create project directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'project_dir': str(project_dir),
        'install_dir': str(install_dir),
        'created': []
    }
    
    print()
    print("-" * 60)
    print(f"  Creating Project: {project_dir.name}")
    print(f"{'-' * 60}")
    print(f"  Location: {project_dir}")
    print(f"  Install:  {install_dir}")
    print()
    
    folders = create_project_structure(project_dir)
    for name, path in folders.items():
        if not name.startswith('.'):
            print(f"  + {name}/")
        result['created'].append(str(path))
    
    # Create run scripts
    if sys.platform == 'win32':
        bat_path = create_run_bat(project_dir, install_dir)
        print(f"  + run.bat")
        result['run_script'] = str(bat_path)
        
        # Create convert.bat for HEVC->ProRes conversion
        convert_path = create_convert_bat(project_dir)
        print(f"  + convert.bat")
        result['convert_script'] = str(convert_path)

        # Create analyze.bat for post-edit timeline analysis
        analyze_path = create_analyze_bat(project_dir, install_dir)
        print(f"  + analyze.bat")
        result['analyze_script'] = str(analyze_path)
    else:
        sh_path = create_run_sh(project_dir, install_dir)
        print(f"  + run.sh")
        result['run_script'] = str(sh_path)

        # Create analyze.sh for post-edit timeline analysis
        analyze_path = create_analyze_sh(project_dir, install_dir)
        print(f"  + analyze.sh")
        result['analyze_script'] = str(analyze_path)
    
    # Create project config (if doesn't exist)
    config_path = project_dir / "project_config.yaml"
    if not config_path.exists():
        create_project_config(project_dir)
        print(f"  + project_config.yaml")
        result['config'] = str(config_path)
    else:
        print(f"  * project_config.yaml (exists)")
        result['config'] = str(config_path)
    
    print(f"\n{'-' * 60}")
    print(f"  Project Ready!")
    print(f"{'-' * 60}")
    print(f"\n  Next steps:")
    print(f"  1. Put your voiceover in: {project_dir.name}\\voiceover\\")
    print(f"  2. Double-click 'run.bat' to start")
    print(f"\n  Quick commands:")
    print(f"    run                  Normal run (auto-saves keywords)")
    print(f"    run --resume         Resume interrupted run")
    print(f"    run --use-keywords   Use saved keywords")
    print(f"    run --list           List saved keyword presets")
    
    return result


def interactive_setup():
    """Interactive project setup with folder navigation"""
    print()
    print("=" * 60)
    print("  VOICEOVER-MATCHER: New Project Setup")
    print("=" * 60)
    print(f"  Install: {get_install_dir()}")
    
    # Load settings
    settings = load_settings()
    last_path = settings.get('last_base_path', DEFAULT_BASE_PATH)
    
    # Step 1: Base path
    print(f"\n  Default base: {last_path}")
    
    try:
        custom_base = input("  Press Enter to use default, or enter new base path: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    if custom_base:
        custom_base = custom_base.strip('"\'')
        base_path = Path(custom_base)
    else:
        base_path = Path(last_path)
    
    # Create base if it doesn't exist
    if not base_path.exists():
        print(f"\n  Base path doesn't exist: {base_path}")
        try:
            create = input("  Create it? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
        
        if create in ('', 'y', 'yes'):
            base_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {base_path}")
        else:
            print("  Cancelled.")
            return
    
    # Step 2: Select client folder (level 1)
    print("\n" + "-" * 60)
    print("  STEP 1: Select Client/Category")
    print("-" * 60)
    
    client_path = select_folder(base_path, "client")
    
    # Step 3: Select series/show folder (level 2) - only if we went into a subfolder
    if client_path != base_path:
        print("\n" + "-" * 60)
        print("  STEP 2: Select Series/Show")
        print("-" * 60)
        
        series_path = select_folder(client_path, "series")
    else:
        series_path = client_path
    
    # Step 4: Project name
    print("\n" + "-" * 60)
    print("  STEP 3: Name Your Project")
    print("-" * 60)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n  Project will be created in: {series_path}")
    print(f"  Date will be added: __{today}")
    
    try:
        project_name = input("\n  Project name (e.g., 'JAPAN Earthquake'): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    if not project_name:
        print("  No name provided. Cancelled.")
        return
    
    # Sanitize and add date
    project_name = sanitize_name(project_name)
    project_name_with_date = f"{project_name}__{today}"
    
    project_path = series_path / project_name_with_date
    
    # Check if exists
    if project_path.exists():
        print(f"\n  Warning: Project already exists: {project_path}")
        try:
            overwrite = input("  Continue anyway? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
        
        if overwrite not in ('y', 'yes'):
            print("  Cancelled.")
            return
    
    # Confirm
    print(f"\n  Will create: {project_path}")
    try:
        confirm = input("  Continue? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    if confirm and confirm not in ('y', 'yes', ''):
        print("  Cancelled.")
        return
    
    # Save last used path
    settings['last_base_path'] = str(base_path)
    settings['last_client_path'] = str(client_path)
    save_settings(settings)
    
    # Create project
    setup_project(str(project_path))


def regenerate_run_script(project_path: str, install_dir: str = None):
    """
    Regenerate the run.bat/run.sh for an existing project.
    Useful if the script was corrupted or install location changed.
    
    Usage:
        python setup_project.py --regenerate "E:\\Projects\\MyDoc"
    """
    project_dir = Path(project_path).resolve()
    
    if install_dir:
        install_path = Path(install_dir).resolve()
    else:
        install_path = get_install_dir()
    
    if not project_dir.exists():
        print(f"  Error: Project directory not found: {project_dir}")
        return False
    
    print(f"\n  Regenerating run script for: {project_dir.name}")
    print(f"  Install directory: {install_path}")
    
    if sys.platform == 'win32':
        bat_path = create_run_bat(project_dir, install_path)
        print(f"  + Created: {bat_path}")
    else:
        sh_path = create_run_sh(project_dir, install_path)
        print(f"  + Created: {sh_path}")
    
    print(f"\n  Done! Quick commands:")
    print(f"    run                  Normal run (auto-saves keywords)")
    print(f"    run --resume         Resume interrupted run")
    print(f"    run --use-keywords   Use saved keywords")
    print(f"    run --list           List saved keyword presets")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Set up a new Voiceover-Matcher project folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python setup_project.py                           # Interactive mode
  python setup_project.py "E:\\Projects\\MyDoc"     # Direct path
  python setup_project.py --regenerate "E:\\Projects\\MyDoc"  # Fix run.bat
        '''
    )
    
    parser.add_argument(
        'project_path',
        nargs='?',
        help='Path to the project directory (interactive if not provided)'
    )
    
    parser.add_argument(
        '--install-dir',
        help='Override central install directory'
    )
    
    parser.add_argument(
        '--regenerate',
        metavar='PROJECT_PATH',
        help='Regenerate run.bat/run.sh for an existing project'
    )
    
    args = parser.parse_args()
    
    if args.regenerate:
        # Regenerate run script for existing project
        regenerate_run_script(args.regenerate, args.install_dir)
    elif args.project_path:
        # Direct path mode
        install_dir = Path(args.install_dir) if args.install_dir else None
        setup_project(args.project_path, install_dir)
    else:
        # Interactive mode
        interactive_setup()


if __name__ == '__main__':
    main()
