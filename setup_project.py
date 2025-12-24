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
    
    FIXED: Removed erroneous 'python main.py %*' that ran before cd.
    The script now correctly:
    1. Sets environment variables
    2. Captures the project directory
    3. Changes to install directory FIRST
    4. THEN runs main.py with --project argument
    """
    bat_content = f'''@echo off
REM ============================================================
REM Voiceover-Matcher Runner
REM Project: {project_dir.name}
REM Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
REM ============================================================

REM Set FFmpeg path (adjust if your ffmpeg is elsewhere)
set IMAGEIO_FFMPEG_EXE=C:\\ffmpeg\\bin\\ffmpeg.exe

REM Central installation location (where main.py lives)
set INSTALL_DIR={install_dir}

REM Project directory (this folder)
set PROJECT_DIR=%~dp0
REM Remove trailing backslash
if "%PROJECT_DIR:~-1%"=="\\" set PROJECT_DIR=%PROJECT_DIR:~0,-1%

echo.
echo   ============================================================
echo   VOICEOVER-MATCHER
echo   ============================================================
echo   Project: %PROJECT_DIR%
echo   Install: %INSTALL_DIR%
echo.

REM Change to install directory FIRST, then run main.py
cd /d "%INSTALL_DIR%"
python main.py --project "%PROJECT_DIR%" %*

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [ERROR] Pipeline failed with error code %ERRORLEVEL%
    pause
)
'''
    
    bat_path = project_dir / "run.bat"
    bat_path.write_text(bat_content)
    return bat_path


def create_run_sh(project_dir: Path, install_dir: Path) -> Path:
    """Create Unix shell script to run the matcher"""
    sh_content = f'''#!/bin/bash
# ============================================================
# Voiceover-Matcher Runner
# Project: {project_dir.name}
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
# ============================================================

# Central installation location (where main.py lives)
INSTALL_DIR="{install_dir}"

# Project directory (this folder)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ============================================================"
echo "  VOICEOVER-MATCHER"
echo "  ============================================================"
echo "  Project: $PROJECT_DIR"
echo "  Install: $INSTALL_DIR"
echo ""

# Change to install directory FIRST, then run main.py
cd "$INSTALL_DIR"
python main.py --project "$PROJECT_DIR" "$@"
'''
    
    sh_path = project_dir / "run.sh"
    sh_path.write_text(sh_content)
    sh_path.chmod(0o755)
    return sh_path


def create_project_config(project_dir: Path) -> Path:
    """Create project-specific config with common overrides commented out"""
    config_content = '''# Project-Specific Configuration Overrides
# =========================================
# Uncomment and modify settings to override global config.yaml
# Only include settings you want to change for THIS project.

# Example overrides:

# voiceover_path: "./voiceover/narration.srt"

# Embedding provider override
# embedding:
#   provider: "voyage"  # Use voyage instead of gemini for this project

# Download settings override
# download:
#   tiers:
#     short:
#       per_keyword: 5  # More short clips for this project

# Matching threshold override
# matching:
#   confidence_threshold: 0.4  # Lower threshold for difficult matches

# Output settings
# output:
#   num_alternatives: 3  # More alternatives for this project
'''
    
    config_path = project_dir / "project_config.yaml"
    config_path.write_text(config_content)
    return config_path


def create_project_structure(project_dir: Path) -> dict:
    """Create the standard project folder structure"""
    folders = {
        'voiceover': 'Place your voiceover .srt or .mp3 files here',
        'downloaded_videos': 'Downloaded stock footage will be saved here',
        'otio_output': 'Generated OTIO timeline files',
        '.cache': 'Cache for transcriptions, embeddings, scene detection',
        'logs': 'Run logs and match decision records'
    }
    
    created = {}
    for folder, description in folders.items():
        folder_path = project_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        created[folder] = folder_path
        
        # Create a README in each folder (except hidden ones)
        if not folder.startswith('.'):
            readme_path = folder_path / "README.txt"
            if not readme_path.exists():
                readme_path.write_text(f"{description}\n")
    
    return created


def setup_project(project_path: str, install_dir: Path = None) -> dict:
    """
    Set up a new project folder.
    
    Args:
        project_path: Path to the project directory
        install_dir: Central install location (auto-detected if None)
    
    Returns:
        Dict with paths to created files/folders
    """
    if install_dir is None:
        install_dir = get_install_dir()
    
    project_dir = Path(project_path).resolve()
    
    # Create project directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'project_dir': project_dir,
        'install_dir': install_dir,
        'created': []
    }
    
    # Create folder structure
    print(f"\n{'-' * 60}")
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
    else:
        sh_path = create_run_sh(project_dir, install_dir)
        print(f"  + run.sh")
        result['run_script'] = str(sh_path)
    
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
    print(f"\n  Or run from command line:")
    print(f"  > cd \"{project_dir}\"")
    print(f"  > run.bat --voiceover voiceover\\script.srt")
    
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
    
    print(f"\n  Done! You can now run the project with:")
    print(f"  > cd \"{project_dir}\"")
    if sys.platform == 'win32':
        print(f"  > run.bat --voiceover voiceover\\script.srt")
    else:
        print(f"  > ./run.sh --voiceover voiceover/script.srt")
    
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