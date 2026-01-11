"""
Interactive prompts for the matcher pipeline.

Extracted from main.py (Jan 2026).
"""

from pathlib import Path
from typing import Optional


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
        print("\n  ─────────────────────────────────────────────────────────────")
        print("  NO VOICEOVER FILES FOUND")
        print("  ─────────────────────────────────────────────────────────────")
        print(f"\n  Searched in:")
        print(f"    • {voiceover_dir}")
        print(f"    • {project_dir}")
        print(f"\n  Supported formats: {', '.join(sorted(voiceover_extensions))}")
        print(f"\n  To continue:")
        print(f"    1. Place your voiceover file in the 'voiceover' folder")
        print(f"    2. Run again, or specify with: run.bat --voiceover <file>")
        return None

    if len(candidates) == 1:
        # Auto-select if only one file
        selected = candidates[0]
        print(f"\n  Auto-detected voiceover: {selected.name}")
        return str(selected)

    # Multiple files - let user choose
    print("\n  ─────────────────────────────────────────────────────────────")
    print("  SELECT VOICEOVER FILE")
    print("  ─────────────────────────────────────────────────────────────")
    print()

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

        print(f"    {i}. {rel_path}")
        print(f"       ({type_hint}, {size_mb:.1f} MB)")

    print(f"\n    0. Cancel")

    while True:
        try:
            choice = input(f"\n  Select [1-{len(candidates)}]: ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == 0:
                print("  Cancelled.")
                return None
            elif 1 <= choice_num <= len(candidates):
                selected = candidates[choice_num - 1]
                print(f"\n  Selected: {selected.name}")
                return str(selected)
            else:
                print(f"  Invalid choice. Enter 1-{len(candidates)} or 0 to cancel.")

        except ValueError:
            # Maybe they typed a filename
            for f in candidates:
                if choice.lower() in f.name.lower():
                    print(f"\n  Selected: {f.name}")
                    return str(f)
            print("  Invalid input. Enter a number.")

        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return None
