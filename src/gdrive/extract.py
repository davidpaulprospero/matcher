"""Utilities for extracting file IDs and URLs from Google Drive links."""

import re


def extract_file_id(url: str) -> str:
    """
    Extract Google Drive file ID from various URL formats.

    Args:
        url: Google Drive URL or file ID

    Returns:
        File ID string or None if not found
    """
    patterns = [
        r'(?:https?://)?(?:www\.)?drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'(?:https?://)?(?:www\.)?drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def construct_download_urls(file_id: str) -> list:
    """Construct multiple download URL formats for the same file."""
    return [
        f"https://drive.google.com/uc?id={file_id}&export=download",
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
        f"https://drive.google.com/open?id={file_id}",
    ]
