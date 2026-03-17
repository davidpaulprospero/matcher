"""
Edge Case Generators for Testing - US-006 Sprint 27

Provides reusable generator functions for edge case test data:
- Empty/minimal SRT segments (0 words, single char)
- Extremely long text segments (>10K chars)
- Unicode/emoji-heavy text content
- Malformed/corrupted checkpoint dicts

Usage:
    from tests.fixtures.edge_cases import (
        generate_empty_srt_segment,
        generate_minimal_srt_segment,
        generate_long_text_segment,
        generate_unicode_segment,
        generate_emoji_segment,
        generate_corrupted_checkpoint,
        generate_edge_case_batch,
    )

    def test_empty_segment_handling():
        segment = generate_empty_srt_segment()
        # Test with empty segment...
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import random
import string


# =============================================================================
# CATEGORY 1: Empty/Minimal SRT Segments
# =============================================================================


def generate_empty_srt_segment(
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 1.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with empty text (0 words).

    Args:
        index: Segment index (default: 0)
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: 1.0)
        source_file: Source file path (default: "test.srt")

    Returns:
        Dict with SRT segment fields and empty text

    Example:
        >>> seg = generate_empty_srt_segment()
        >>> assert seg["text"] == ""
        >>> assert seg["word_count"] == 0
    """
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": "",
        "source_file": source_file,
        "word_count": 0,
        "keywords": [],
    }


def generate_minimal_srt_segment(
    char: str = "A",
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 0.1,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with single character text.

    Args:
        char: Single character for text (default: "A")
        index: Segment index (default: 0)
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (default: 0.1)
        source_file: Source file path (default: "test.srt")

    Returns:
        Dict with SRT segment fields and single-char text

    Example:
        >>> seg = generate_minimal_srt_segment(char="X")
        >>> assert seg["text"] == "X"
        >>> assert len(seg["text"]) == 1
    """
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": char,
        "source_file": source_file,
        "word_count": 1,
        "keywords": [],
    }


def generate_whitespace_only_segment(
    whitespace: str = "   ",
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 1.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with only whitespace text.

    Args:
        whitespace: Whitespace string (default: "   " - 3 spaces)
        index: Segment index
        start_time: Start time in seconds
        end_time: End time in seconds
        source_file: Source file path

    Returns:
        Dict with SRT segment with whitespace-only text
    """
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": whitespace,
        "source_file": source_file,
        "word_count": 0,
        "keywords": [],
    }


def generate_zero_duration_segment(
    text: str = "Zero duration",
    index: int = 0,
    timestamp: float = 5.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with zero duration (start == end).

    Args:
        text: Segment text
        index: Segment index
        timestamp: Both start and end time (same value)
        source_file: Source file path

    Returns:
        Dict with SRT segment where duration is 0
    """
    return {
        "index": index,
        "start_time": timestamp,
        "end_time": timestamp,  # Same as start = 0 duration
        "text": text,
        "source_file": source_file,
        "word_count": len(text.split()),
        "keywords": [],
    }


def generate_negative_duration_segment(
    text: str = "Backwards",
    index: int = 0,
    start_time: float = 10.0,
    end_time: float = 5.0,  # Before start!
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with negative duration (end < start).

    Args:
        text: Segment text
        index: Segment index
        start_time: Start time (will be > end_time)
        end_time: End time (will be < start_time)
        source_file: Source file path

    Returns:
        Dict with malformed SRT segment (negative duration)
    """
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": len(text.split()),
        "keywords": [],
    }


# =============================================================================
# CATEGORY 2: Extremely Long Text Segments (>10K chars)
# =============================================================================


def generate_long_text_segment(
    length: int = 10001,
    pattern: str = "A",
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 3600.0,  # 1 hour for very long text
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with extremely long text (>10K chars).

    Args:
        length: Text length in characters (default: 10001 = just over 10K)
        pattern: Character pattern to repeat (default: "A")
        index: Segment index
        start_time: Start time in seconds
        end_time: End time in seconds
        source_file: Source file path

    Returns:
        Dict with SRT segment containing very long text

    Example:
        >>> seg = generate_long_text_segment(length=15000)
        >>> assert len(seg["text"]) == 15000
    """
    text = pattern * length
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": 1,  # One giant "word"
        "keywords": [],
    }


def generate_long_word_segment(
    word_count: int = 2000,
    word_length: int = 5,
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 3600.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with many words (high word count).

    Args:
        word_count: Number of words (default: 2000)
        word_length: Length of each word (default: 5)
        index: Segment index
        start_time: Start time
        end_time: End time
        source_file: Source file path

    Returns:
        Dict with SRT segment with many words
    """
    words = ["word" + str(i % 100).zfill(2) for i in range(word_count)]
    text = " ".join(words)
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": word_count,
        "keywords": [],
    }


def generate_repeated_text_segment(
    phrase: str = "Lorem ipsum dolor sit amet. ",
    repetitions: int = 500,
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 3600.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate segment with repeated phrase (realistic long text).

    Args:
        phrase: Phrase to repeat
        repetitions: Number of times to repeat
        index: Segment index
        start_time: Start time
        end_time: End time
        source_file: Source file path

    Returns:
        Dict with long realistic text
    """
    text = phrase * repetitions
    word_count = len(text.split())
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": word_count,
        "keywords": [],
    }


# =============================================================================
# CATEGORY 3: Unicode/Emoji-Heavy Text Content
# =============================================================================


# Common unicode character sets
UNICODE_SAMPLES = {
    "japanese": "日本語テキスト",
    "chinese": "中文文本测试",
    "korean": "한국어 텍스트",
    "arabic": "النص العربي",
    "greek": "Ελληνικό κείμενο",
    "russian": "Русский текст",
    "hebrew": "טקסט בעברית",
    "thai": "ข้อความภาษาไทย",
    "hindi": "हिंदी पाठ",
    "mixed": "日本語 中文 한국어 Русский Ελληνικά العربية",
}

EMOJI_SAMPLES = {
    "basic": "😀😃😄😁😆😅🤣😂🙂🙃",
    "faces": "😍🥰😘😗😙😚😋😛😜🤪😝",
    "objects": "🎬🎥📹📸🎤🎧🎹🎸🎷🎺",
    "symbols": "✅❌⭐💫✨🔥💯🎯🚀💪",
    "flags": "🇺🇸🇬🇧🇯🇵🇰🇷🇩🇪🇫🇷🇪🇸🇮🇹",
    "compound": "👨‍👩‍👧‍👦👩‍❤️‍👨🏳️‍🌈🧑‍💻👨‍🔬",
}


def generate_unicode_segment(
    language: str = "japanese",
    repetitions: int = 10,
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 10.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with unicode text.

    Args:
        language: Language key from UNICODE_SAMPLES (or custom text)
        repetitions: Number of times to repeat sample
        index: Segment index
        start_time: Start time
        end_time: End time
        source_file: Source file path

    Returns:
        Dict with unicode text

    Example:
        >>> seg = generate_unicode_segment("japanese")
        >>> assert "日本語" in seg["text"]
    """
    base_text = UNICODE_SAMPLES.get(language, language)
    text = (base_text + " ") * repetitions
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text.strip(),
        "source_file": source_file,
        "word_count": len(text.split()),
        "keywords": [],
    }


def generate_emoji_segment(
    category: str = "objects",
    repetitions: int = 10,
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 5.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate an SRT segment with emoji-heavy text.

    Args:
        category: Emoji category from EMOJI_SAMPLES
        repetitions: Number of times to repeat
        index: Segment index
        start_time: Start time
        end_time: End time
        source_file: Source file path

    Returns:
        Dict with emoji text

    Example:
        >>> seg = generate_emoji_segment("objects")
        >>> assert "🎬" in seg["text"]
    """
    base_text = EMOJI_SAMPLES.get(category, category)
    text = " ".join([base_text] * repetitions)
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": repetitions,
        "keywords": [],
    }


def generate_mixed_unicode_emoji_segment(
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 10.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate segment with mixed unicode and emoji content.

    Returns:
        Dict with diverse unicode/emoji text
    """
    components = [
        "Hello 你好 こんにちは 안녕하세요",
        "🎬 Video Production 📹",
        "Café résumé naïve",
        "∑∏∫∂∇ Math symbols",
        "→←↑↓↔ Arrows",
        "日本語 🇯🇵 Japanese",
        "中文 🇨🇳 Chinese",
        "한국어 🇰🇷 Korean",
    ]
    text = " | ".join(components)
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": len(text.split()),
        "keywords": [],
    }


def generate_special_chars_segment(
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 5.0,
    source_file: str = "test.srt",
) -> Dict[str, Any]:
    """
    Generate segment with XML/JSON-problematic special characters.

    Returns:
        Dict with special characters that may break parsers
    """
    text = (
        'Text with <tags> & "quotes" and \'apostrophes\' '
        "plus {braces} [brackets] and `backticks` "
        "and $dollars and @mentions and #hashtags "
        "and \\ backslashes and / forward slashes "
        "and | pipes and ~ tildes"
    )
    return {
        "index": index,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "source_file": source_file,
        "word_count": len(text.split()),
        "keywords": [],
    }


# =============================================================================
# CATEGORY 4: Malformed/Corrupted Checkpoint Dicts
# =============================================================================


def generate_corrupted_checkpoint(
    corruption_type: str = "missing_fields",
) -> Dict[str, Any]:
    """
    Generate a corrupted checkpoint dict for testing error handling.

    Args:
        corruption_type: Type of corruption to apply:
            - "missing_fields": Missing required fields
            - "wrong_types": Fields with wrong types
            - "circular_reference": Contains circular reference (will fail JSON)
            - "null_values": Critical fields set to None
            - "negative_values": Invalid negative numbers
            - "empty_nested": Empty nested structures
            - "extra_fields": Unknown extra fields
            - "malformed_stage": Stage data with wrong structure

    Returns:
        Dict with corrupted checkpoint data

    Example:
        >>> cp = generate_corrupted_checkpoint("missing_fields")
        >>> assert "version" not in cp or "last_completed_stage" not in cp
    """
    base_checkpoint = {
        "version": "1.0",
        "created_at": "2026-01-29T12:00:00",
        "last_completed_stage": "MATCH",
        "config_hash": "abc123",
        "voiceover_path": "/path/to/voiceover.srt",
        "voiceover_hash": "def456",
    }

    if corruption_type == "missing_fields":
        # Remove required fields
        del base_checkpoint["version"]
        del base_checkpoint["last_completed_stage"]
        return base_checkpoint

    elif corruption_type == "wrong_types":
        # Use wrong types for fields
        base_checkpoint["version"] = 123  # Should be string
        base_checkpoint["last_completed_stage"] = ["MATCH"]  # Should be string
        base_checkpoint["created_at"] = True  # Should be string
        return base_checkpoint

    elif corruption_type == "circular_reference":
        # Note: This will fail JSON serialization
        base_checkpoint["self_ref"] = base_checkpoint
        return base_checkpoint

    elif corruption_type == "null_values":
        base_checkpoint["version"] = None
        base_checkpoint["last_completed_stage"] = None
        base_checkpoint["voiceover_path"] = None
        return base_checkpoint

    elif corruption_type == "negative_values":
        base_checkpoint["analyze"] = {
            "segment_count": -5,
            "keyword_count": -10,
        }
        base_checkpoint["match"] = {
            "match_count": -1,
            "avg_confidence": -0.5,
        }
        return base_checkpoint

    elif corruption_type == "empty_nested":
        base_checkpoint["analyze"] = {}
        base_checkpoint["download"] = {}
        base_checkpoint["match"] = {"matches": []}
        return base_checkpoint

    elif corruption_type == "extra_fields":
        base_checkpoint["unknown_field_1"] = "value1"
        base_checkpoint["__internal__"] = {"debug": True}
        base_checkpoint["random_data"] = [1, 2, 3, {"nested": "value"}]
        return base_checkpoint

    elif corruption_type == "malformed_stage":
        base_checkpoint["match"] = {
            "matches": "not a list",  # Should be list
            "avg_confidence": "high",  # Should be float
            "completed": "yes",  # Should be bool
        }
        return base_checkpoint

    else:
        # Unknown type - return base with a note
        base_checkpoint["_corruption_type"] = f"unknown: {corruption_type}"
        return base_checkpoint


def generate_checkpoint_with_numpy_simulation() -> Dict[str, Any]:
    """
    Generate checkpoint dict simulating numpy array serialization issues.

    Note: This simulates what happens when numpy arrays are serialized
    via json.dumps with default=str (they become string representations).

    Returns:
        Dict with numpy-like string values
    """
    return {
        "version": "1.0",
        "created_at": "2026-01-29T12:00:00",
        "last_completed_stage": "TRANSCRIBE",
        "transcribe": {
            # Simulates numpy array -> string conversion
            "embeddings": "[0.1 0.2 0.3 0.4 0.5]",
            "matrix": "[[0.1 0.2]\n [0.3 0.4]]",
            "shape": "(5, 384)",  # String instead of tuple
        },
        "config_hash": "abc123",
    }


def generate_checkpoint_with_datetime_strings() -> Dict[str, Any]:
    """
    Generate checkpoint with datetime objects serialized as strings.

    Returns:
        Dict with datetime string values
    """
    now = datetime.now()
    return {
        "version": "1.0",
        "created_at": now.isoformat(),
        "updated_at": str(now),  # Different format
        "last_completed_stage": "OUTPUT",
        "output": {
            "started_at": "2026-01-29 10:00:00",  # Space format
            "completed_at": "2026-01-29T12:30:45",  # ISO format
            "duration_seconds": "3600.5",  # String instead of float
        },
        "config_hash": "abc123",
    }


def generate_partial_checkpoint(
    complete_stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate checkpoint with only partial stage data.

    Args:
        complete_stages: List of stage names to include data for

    Returns:
        Dict with partial checkpoint data
    """
    stages = complete_stages or ["ANALYZE", "DOWNLOAD"]

    checkpoint = {
        "version": "1.0",
        "created_at": "2026-01-29T12:00:00",
        "last_completed_stage": stages[-1] if stages else None,
        "config_hash": "abc123",
    }

    if "ANALYZE" in stages:
        checkpoint["analyze"] = {"keywords": ["test"], "segment_count": 5}

    if "DOWNLOAD" in stages:
        checkpoint["download"] = {"video_count": 3}

    # All other stages are missing (simulates interrupted run)
    return checkpoint


# =============================================================================
# CATEGORY 5: Batch Generators
# =============================================================================


def generate_edge_case_batch(
    batch_type: str = "all",
    count_per_type: int = 2,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of edge case segments for comprehensive testing.

    Args:
        batch_type: Type of batch to generate:
            - "all": All types of edge cases
            - "empty": Empty/minimal segments
            - "long": Long text segments
            - "unicode": Unicode/emoji segments
        count_per_type: Number of each type to generate

    Returns:
        List of edge case segment dicts

    Example:
        >>> batch = generate_edge_case_batch("all", count_per_type=1)
        >>> assert len(batch) >= 10
    """
    segments = []
    idx = 0

    if batch_type in ("all", "empty"):
        for _ in range(count_per_type):
            segments.append(generate_empty_srt_segment(index=idx))
            idx += 1
            segments.append(generate_minimal_srt_segment(index=idx))
            idx += 1
            segments.append(generate_whitespace_only_segment(index=idx))
            idx += 1
            segments.append(generate_zero_duration_segment(index=idx))
            idx += 1

    if batch_type in ("all", "long"):
        for _ in range(count_per_type):
            segments.append(generate_long_text_segment(index=idx))
            idx += 1
            segments.append(generate_long_word_segment(index=idx))
            idx += 1

    if batch_type in ("all", "unicode"):
        for lang in ["japanese", "chinese", "mixed"]:
            segments.append(generate_unicode_segment(language=lang, index=idx))
            idx += 1
        for category in ["objects", "faces"]:
            segments.append(generate_emoji_segment(category=category, index=idx))
            idx += 1
        segments.append(generate_mixed_unicode_emoji_segment(index=idx))
        idx += 1
        segments.append(generate_special_chars_segment(index=idx))
        idx += 1

    return segments


def generate_checkpoint_corruption_batch() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate all types of corrupted checkpoints for testing.

    Returns:
        List of (corruption_type, checkpoint_dict) tuples

    Example:
        >>> batch = generate_checkpoint_corruption_batch()
        >>> for corruption_type, checkpoint in batch:
        ...     print(f"Testing {corruption_type}...")
    """
    corruption_types = [
        "missing_fields",
        "wrong_types",
        "null_values",
        "negative_values",
        "empty_nested",
        "extra_fields",
        "malformed_stage",
    ]

    batch = []
    for ctype in corruption_types:
        batch.append((ctype, generate_corrupted_checkpoint(ctype)))

    # Add special cases
    batch.append(("numpy_simulation", generate_checkpoint_with_numpy_simulation()))
    batch.append(("datetime_strings", generate_checkpoint_with_datetime_strings()))
    batch.append(("partial", generate_partial_checkpoint()))

    return batch


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Category 1: Empty/Minimal
    "generate_empty_srt_segment",
    "generate_minimal_srt_segment",
    "generate_whitespace_only_segment",
    "generate_zero_duration_segment",
    "generate_negative_duration_segment",
    # Category 2: Long Text
    "generate_long_text_segment",
    "generate_long_word_segment",
    "generate_repeated_text_segment",
    # Category 3: Unicode/Emoji
    "generate_unicode_segment",
    "generate_emoji_segment",
    "generate_mixed_unicode_emoji_segment",
    "generate_special_chars_segment",
    "UNICODE_SAMPLES",
    "EMOJI_SAMPLES",
    # Category 4: Corrupted Checkpoints
    "generate_corrupted_checkpoint",
    "generate_checkpoint_with_numpy_simulation",
    "generate_checkpoint_with_datetime_strings",
    "generate_partial_checkpoint",
    # Category 5: Batch Generators
    "generate_edge_case_batch",
    "generate_checkpoint_corruption_batch",
]
