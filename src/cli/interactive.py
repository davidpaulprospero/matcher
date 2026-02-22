"""
Interactive prompts for the matcher pipeline.

Extracted from main.py (Jan 2026).
"""

import sys
from pathlib import Path
from typing import Optional

# Add scripts/ to path for script_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from script_utils import print_ok, print_warn, print_error, print_info, print_header, set_verbosity

# Initialize verbosity to normal (1) so messages show by default
set_verbosity(1)


def find_voiceover_interactive(project_dir: Path) -> Optional[str]:
    """
    Find voiceover files in the project directory and let user select.

    Searches:
    - project_dir/voiceover/ (primary location)
    - project_dir/*.srt, *.mp3, *.wav, *.mp4 (fallback)

    Args:
        project_dir: Project directory to search

    Returns:
        Path to selected voiceover file, or None if cancelled/not found
    """
    voiceover_extensions = {'.srt', '.mp3', '.wav', '.mp4', '.m4a', '.aac', '.flac', '.ogg'}
    candidates = []

    # Search in voiceover subdirectory first
    voiceover_dir = project_dir / 'voiceover'
    if voiceover_dir.exists():
        for f in voiceover_dir.iterdir():
            if f.is_file() and f.suffix.lower() in voiceover_extensions:
                candidates.append(f)

    # Also search project root
    for f in project_dir.iterdir():
        if f.is_file() and f.suffix.lower() in voiceover_extensions:
            # Avoid duplicates
            if f not in candidates:
                candidates.append(f)

    # Sort by name
    candidates.sort(key=lambda x: x.name.lower())

    if not candidates:
        print_header("NO VOICEOVER FILES FOUND")
        print_info(f"Searched in:")
        print_info(f"  - {voiceover_dir}")
        print_info(f"  - {project_dir}")
        print_info(f"Supported formats: {', '.join(sorted(voiceover_extensions))}")
        print_info("To continue:")
        print_info("  1. Place your voiceover file in the 'voiceover' folder")
        print_info("  2. Run again, or specify with: run.bat --voiceover <file>")
        return None

    if len(candidates) == 1:
        # Auto-select if only one file
        selected = candidates[0]
        print_ok(f"Auto-detected voiceover: {selected.name}")
        return str(selected)

    # Multiple files - let user choose
    print_header("SELECT VOICEOVER FILE")

    for i, f in enumerate(candidates, 1):
        # Show relative path from project dir
        try:
            rel_path = f.relative_to(project_dir)
        except ValueError:
            rel_path = f.name

        # Show file size
        size_mb = f.stat().st_size / (1024 * 1024)

        # Show file type hint
        type_hint = {
            '.srt': 'subtitles',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.mp4': 'video',
            '.m4a': 'audio',
            '.aac': 'audio',
            '.flac': 'audio',
            '.ogg': 'audio',
        }.get(f.suffix.lower(), 'file')

        print_info(f"    {i}. {rel_path}")
        print_info(f"       ({type_hint}, {size_mb:.1f} MB)")

    print_info(f"\n  0. Cancel")

    while True:
        try:
            choice = input(f"\n  Select [1-{len(candidates)}]: ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == 0:
                print_warn("Cancelled.")
                return None
            elif 1 <= choice_num <= len(candidates):
                selected = candidates[choice_num - 1]
                print_ok(f"Selected: {selected.name}")
                return str(selected)
            else:
                print_warn(f"Invalid choice. Enter 1-{len(candidates)} or 0 to cancel.")

        except ValueError:
            # Maybe they typed a filename
            for f in candidates:
                if choice.lower() in f.name.lower():
                    print_ok(f"Selected: {f.name}")
                    return str(f)
            print_warn("Invalid input. Enter a number.")

        except (EOFError, KeyboardInterrupt):
            print_warn("Cancelled.")
            return None
