"""
Project metadata utilities for reading Trello card info and sanitizing filenames.
"""

import json
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Sanitize a string for use as a filename base.

    - Strips characters invalid on Windows: < > : " / \\ | ? *
    - Replaces spaces and dashes with underscores
    - Collapses repeated underscores
    - Truncates at word boundary to max_length

    Args:
        name: Raw name string
        max_length: Maximum length (default 80)

    Returns:
        Sanitized string, or empty string if input is None/empty
    """
    if not name:
        return ""

    # Strip filesystem-unsafe and shell-problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*\'`,!]', '', name)

    # Replace spaces, dashes (ASCII and unicode), and em/en-dashes with underscores
    sanitized = re.sub(r'[\s\-\u2013\u2014]+', '_', sanitized)

    # Collapse repeated underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Truncate at word boundary
    if len(sanitized) > max_length:
        truncated = sanitized[:max_length].rsplit('_', 1)[0].rstrip('_')
        sanitized = truncated or sanitized[:max_length]

    return sanitized


def get_card_title(project_dir: Path) -> Optional[str]:
    """
    Read the Trello card title from trello_card.json in the project directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        Sanitized card title, or None if not available
    """
    trello_card_path = project_dir / "trello_card.json"

    if not trello_card_path.exists():
        return None

    try:
        data = json.loads(trello_card_path.read_text(encoding='utf-8'))
        name = data.get('name')
        if not name:
            return None
        sanitized = sanitize_filename(name)
        return sanitized or None
    except (json.JSONDecodeError, OSError):
        return None
