"""Cache utility functions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union, List


def compute_hash(data: Union[str, bytes], length: int = 16) -> str:
    """
    Compute MD5 hash with configurable length.

    Args:
        data: String or bytes to hash
        length: Hash length (default: 16 characters)

    Returns:
        Hex digest truncated to length

    Examples:
        >>> compute_hash("test")
        '098f6bcd46212109'
        >>> compute_hash("test", length=8)
        '098f6bcd'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()[:length]


def file_content_hash(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute content hash from file (first + last chunks).

    Uses first and last 1MB chunks for efficiency on large files.
    This is the same strategy used by GlobalCacheManager.

    Args:
        file_path: Path to file
        chunk_size: Chunk size in bytes (default: 1MB)

    Returns:
        MD5 hash hex digest

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> file_content_hash("/path/to/video.mp4")
        'a1b2c3d4e5f6...'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size

    with open(file_path, 'rb') as f:
        first_chunk = f.read(chunk_size)

        # For files larger than 2MB, read last chunk
        if file_size > 2 * chunk_size:
            f.seek(-chunk_size, 2)
            last_chunk = f.read()
        else:
            last_chunk = b''

    # Include file size in hash to handle size changes
    hash_data = f"{file_size}:{first_chunk.hex()}:{last_chunk.hex()}".encode()
    return hashlib.md5(hash_data).hexdigest()


def file_metadata_hash(file_path: Path | str) -> str:
    """
    Compute hash from file metadata (name + size).

    Faster than content hash, but less reliable (won't detect content changes
    if size stays the same). Used by TranscriptCache.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash hex digest

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> file_metadata_hash("/path/to/video.mp4")
        'b1c2d3e4f5a6...'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size = file_path.stat().st_size
    return compute_hash(f"{file_path.name}:{size}")


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize path for cross-platform cache keys.

    Resolves to absolute path and converts backslashes to forward slashes.
    Used by TranscriptCache, TopicExtractor, EntityCache, GlobalCache.

    Args:
        path: File path

    Returns:
        Normalized path string with forward slashes

    Examples:
        >>> normalize_path("C:\\Users\\test\\file.txt")
        'C:/Users/test/file.txt'
        >>> normalize_path("/home/user/file.txt")
        '/home/user/file.txt'
    """
    return str(Path(path).resolve()).replace('\\', '/')


def batch_hash(items: List[str], length: int = 16) -> str:
    """
    Compute hash for a batch of items.

    Sorts items before hashing to ensure consistent hash regardless of order.
    Used by EmbeddingCache for batch operations.

    Args:
        items: List of strings
        length: Hash length (default: 16 characters)

    Returns:
        MD5 hash of concatenated items

    Examples:
        >>> batch_hash(["item1", "item2"])
        'f1e2d3c4b5a6...'
        >>> batch_hash(["item2", "item1"])  # Same hash (sorted)
        'f1e2d3c4b5a6...'
    """
    return compute_hash('|'.join(sorted(items)), length=length)


def text_hash(text: str, length: int = 12) -> str:
    """
    Compute hash from text content.

    Convenience function used by EmbeddingCache for per-text caching.

    Args:
        text: Text to hash
        length: Hash length (default: 12 characters)

    Returns:
        MD5 hash hex digest

    Examples:
        >>> text_hash("hello world")
        '5eb63bbbe01e'
    """
    return compute_hash(text, length=length)
