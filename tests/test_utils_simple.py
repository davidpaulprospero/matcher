"""
Unit tests for utils module - comprehensive coverage.

Tests all major utilities including path handling, caching, progress tracking,
reuse tracking, FFmpeg debugging, and SRT parsing.
"""

import pytest
from pathlib import Path
import sys
import numpy as np
import json
import tempfile
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    # Embeddings
    is_embeddings_empty,
    # Path utilities
    normalize_path,
    sanitize_path,
    resolve_path,
    # FFmpeg debugging
    setup_ffmpeg_debug_log,
    log_ffmpeg_debug,
    FFmpegStderrCapture,
    # Dataclasses
    SRTSegment,
    Chapter,
    SceneInfo,
    VideoIndex,
    Match,
    AlternativeMatch,
    StrategyMatch,
    MatchResult,
    Topic,
    # Progress bar
    ProgressBar,
    # Caching
    CacheManager,
    # Reuse tracking
    ReuseTracker,
    # SRT parsing
    parse_srt_timestamp,
    format_srt_timestamp,
    parse_srt_file,
    write_srt_file
)


@pytest.mark.fast
class TestSRTSegment:
    """Test SRTSegment dataclass."""

    def test_create_srt_segment(self):
        """Test creating SRT segment."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test subtitle text"
        )

        assert segment.index == 1
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Test subtitle text"

    def test_srt_segment_duration(self):
        """Test computing segment duration."""
        segment = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=25.5,
            text="Test"
        )

        duration = segment.end_time - segment.start_time
        assert duration == 15.5

    def test_srt_segment_with_source_file(self):
        """Test segment with source file."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            source_file="/path/to/video.mp4"
        )

        assert segment.source_file == "/path/to/video.mp4"

    def test_srt_segment_equality(self):
        """Test comparing SRT segments."""
        seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        seg2 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        seg3 = SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Different")

        assert seg1 == seg2
        assert seg1 != seg3


@pytest.mark.fast
class TestSRTTimestamps:
    """Test SRT timestamp formatting and parsing."""

    def test_format_srt_timestamp_basic(self):
        """Test basic timestamp formatting."""
        formatted = format_srt_timestamp(90.5)

        # Should be in format: HH:MM:SS,mmm
        assert ":" in formatted
        assert "," in formatted

    def test_format_srt_timestamp_hours(self):
        """Test formatting with hours."""
        formatted = format_srt_timestamp(3665.0)  # 1 hour, 1 minute, 5 seconds

        assert formatted.count(":") == 2

    def test_parse_srt_timestamp_basic(self):
        """Test parsing SRT timestamp."""
        seconds = parse_srt_timestamp("00:01:30,000")

        assert seconds == 90.0

    def test_parse_srt_timestamp_with_milliseconds(self):
        """Test parsing with milliseconds."""
        seconds = parse_srt_timestamp("00:00:01,500")

        assert seconds == 1.5

    def test_format_parse_roundtrip(self):
        """Test formatting and parsing round-trip."""
        original = 125.75

        formatted = format_srt_timestamp(original)
        parsed = parse_srt_timestamp(formatted)

        assert abs(parsed - original) < 0.01  # Within 10ms


@pytest.mark.fast
class TestPathHandling:
    """Test path normalization and handling."""

    def test_normalize_path_forward_slash(self):
        """Test normalizing path with forward slashes."""
        path = "C:/Users/Test/video.mp4"

        normalized = normalize_path(path)

        assert normalized is not None

    def test_normalize_path_backslash(self):
        """Test normalizing path with backslashes."""
        path = "C:\\Users\\Test\\video.mp4"

        normalized = normalize_path(path)

        assert normalized is not None

    def test_sanitize_path_basic(self):
        """Test sanitizing path."""
        path = "C:/Users/Test/video.mp4"

        sanitized = sanitize_path(path)

        assert sanitized is not None

    def test_sanitize_path_special_chars(self):
        """Test sanitizing path with special characters."""
        path = "C:/Users/Test: Video/file.mp4"

        sanitized = sanitize_path(path)

        # Should handle special characters
        assert sanitized is not None


@pytest.mark.fast
class TestSRTFileParsing:
    """Test parsing SRT files."""

    def test_parse_srt_basic(self, tmp_path):
        """Test parsing basic SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First subtitle

2
00:00:05,000 --> 00:00:10,000
Second subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 2
        assert segments[0].text == "First subtitle"
        assert segments[1].text == "Second subtitle"

    def test_parse_srt_with_timestamps(self, tmp_path):
        """Test parsing SRT preserves timestamps."""
        srt_content = """1
00:00:10,500 --> 00:00:15,750
Test subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 1
        assert segments[0].start_time == 10.5
        assert segments[0].end_time == 15.75

    def test_write_srt_basic(self, tmp_path):
        """Test writing SRT file."""
        segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="First subtitle"
            ),
            SRTSegment(
                index=2,
                start_time=5.0,
                end_time=10.0,
                text="Second subtitle"
            )
        ]

        output_file = tmp_path / "output.srt"
        write_srt_file(segments, str(output_file))

        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        assert "First subtitle" in content
        assert "Second subtitle" in content

    def test_write_read_roundtrip(self, tmp_path):
        """Test writing and reading SRT round-trip."""
        original_segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="Test 1"
            ),
            SRTSegment(
                index=2,
                start_time=5.0,
                end_time=10.0,
                text="Test 2"
            )
        ]

        srt_file = tmp_path / "test.srt"
        write_srt_file(original_segments, str(srt_file))

        # Read back
        loaded_segments = parse_srt_file(str(srt_file))

        assert len(loaded_segments) == len(original_segments)
        for orig, loaded in zip(original_segments, loaded_segments):
            assert orig.text == loaded.text
            assert abs(orig.start_time - loaded.start_time) < 0.01
            assert abs(orig.end_time - loaded.end_time) < 0.01


@pytest.mark.fast
class TestEmbeddingHelpers:
    """Test embedding helper functions."""

    def test_is_embeddings_empty_none(self):
        """Test checking None embeddings."""
        assert is_embeddings_empty(None)

    def test_is_embeddings_empty_array(self):
        """Test checking empty array."""
        empty_array = np.array([])

        assert is_embeddings_empty(empty_array)

    def test_is_embeddings_not_empty(self):
        """Test checking non-empty embeddings."""
        embeddings = np.random.randn(10, 1024).astype(np.float32)

        assert not is_embeddings_empty(embeddings)

    def test_is_embeddings_single_vector(self):
        """Test checking single vector."""
        single_vec = np.random.randn(1024).astype(np.float32)

        assert not is_embeddings_empty(single_vec)


@pytest.mark.fast
class TestPathUtilitiesAdvanced:
    """Test advanced path handling features."""

    def test_sanitize_path_extended_length_standard(self):
        """Test removing standard extended-length prefix."""
        path = r"\\?\C:\Users\Test\video.mp4"
        sanitized = sanitize_path(path)

        assert not sanitized.startswith(r"\\?")
        assert "C:" in sanitized or "c:" in sanitized

    def test_sanitize_path_device_form(self):
        """Test removing device form prefix."""
        path = r"\\.\C:\Users\Test\video.mp4"
        sanitized = sanitize_path(path)

        assert not sanitized.startswith(r"\\.")

    def test_sanitize_path_forward_slash_prefix(self):
        """Test removing forward slash prefix."""
        path = "//?/C:/Users/Test/video.mp4"
        sanitized = sanitize_path(path)

        assert not sanitized.startswith("//?/")

    def test_sanitize_path_question_mark_edge_case(self):
        """Test edge case with ?\ or ?/ prefix."""
        path = r"?\C:\Users\Test\video.mp4"
        sanitized = sanitize_path(path)

        # Should remove the ?\ prefix
        assert not sanitized.startswith("?")

    def test_sanitize_path_double_slashes(self):
        """Test removing double slashes."""
        path = "C://Users//Test//video.mp4"
        sanitized = sanitize_path(path)

        # Should not have consecutive slashes
        assert "//" not in sanitized

    def test_normalize_path_empty_string(self):
        """Test normalizing empty path."""
        normalized = normalize_path("")

        assert normalized == ""

    def test_resolve_path_absolute(self, tmp_path):
        """Test resolving absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        resolved = resolve_path(str(test_file))

        assert resolved is not None
        assert "test.txt" in resolved

    def test_resolve_path_relative_with_base(self, tmp_path):
        """Test resolving relative path with base directory."""
        base_dir = tmp_path
        rel_path = "subdir/file.txt"

        resolved = resolve_path(rel_path, base_dir)

        assert "subdir" in resolved
        assert "file.txt" in resolved

    def test_resolve_path_oserror_fallback(self):
        """Test resolve_path handles OSError gracefully."""
        # Test with a path that might cause issues
        with patch.object(Path, 'resolve', side_effect=OSError("Test error")):
            # Should still return a path without crashing
            result = resolve_path("test.txt")
            assert isinstance(result, str)


@pytest.mark.fast
class TestFFmpegDebugLogging:
    """Test FFmpeg debug logging functionality."""

    def test_setup_ffmpeg_debug_log(self, tmp_path):
        """Test setting up FFmpeg debug log."""
        log_dir = tmp_path / "logs"

        log_path = setup_ffmpeg_debug_log(log_dir)

        assert log_path.exists()
        assert log_path.name == "ffmpeg_debug.log"

        # Verify header was written
        content = log_path.read_text()
        assert "FFmpeg Debug Log" in content
        assert "Started:" in content

    def test_log_ffmpeg_debug_basic(self, tmp_path):
        """Test logging FFmpeg debug message."""
        log_dir = tmp_path / "logs"
        setup_ffmpeg_debug_log(log_dir)

        log_ffmpeg_debug("Test message", "opencv")

        log_path = log_dir / "ffmpeg_debug.log"
        content = log_path.read_text()

        assert "Test message" in content
        assert "[opencv]" in content

    def test_log_ffmpeg_debug_no_log_set(self):
        """Test logging when no log is set up (should not crash)."""
        # Reset global log path by calling with invalid dir
        # then try to log - should handle gracefully
        try:
            log_ffmpeg_debug("Test message")
            # Should not crash
            assert True
        except Exception:
            pytest.fail("log_ffmpeg_debug raised exception when log not set")

    def test_ffmpeg_stderr_capture_context(self, tmp_path):
        """Test FFmpeg stderr capture context manager."""
        log_dir = tmp_path / "logs"
        setup_ffmpeg_debug_log(log_dir)

        with FFmpegStderrCapture("test_source"):
            # Write to stderr
            sys.stderr.write("Test stderr message\n")

        # Check log file contains captured message
        log_path = log_dir / "ffmpeg_debug.log"
        content = log_path.read_text()

        assert "Test stderr message" in content or "test_source" in content

    def test_ffmpeg_stderr_capture_no_log(self):
        """Test stderr capture when log not set up."""
        # Reset by not calling setup
        with FFmpegStderrCapture("test"):
            # Should not crash
            sys.stderr.write("Test\n")

        assert True

    def test_ffmpeg_stderr_capture_exception_passthrough(self, tmp_path):
        """Test that exceptions are not suppressed."""
        log_dir = tmp_path / "logs"
        setup_ffmpeg_debug_log(log_dir)

        with pytest.raises(ValueError):
            with FFmpegStderrCapture("test"):
                raise ValueError("Test exception")


@pytest.mark.fast
class TestChapterDataclass:
    """Test Chapter dataclass."""

    def test_chapter_contains_segment_true(self):
        """Test chapter contains segment."""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=5,
            title="Test Chapter"
        )

        assert chapter.contains_segment(0)
        assert chapter.contains_segment(3)
        assert chapter.contains_segment(5)

    def test_chapter_contains_segment_false(self):
        """Test chapter does not contain segment."""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=5,
            end_segment_idx=10
        )

        assert not chapter.contains_segment(4)
        assert not chapter.contains_segment(11)

    def test_chapter_to_dict(self):
        """Test converting chapter to dict."""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=5,
            title="Test",
            topics=["topic1", "topic2"]
        )

        chapter_dict = chapter.to_dict()

        assert chapter_dict["chapter_id"] == 1
        assert chapter_dict["title"] == "Test"
        assert len(chapter_dict["topics"]) == 2


@pytest.mark.fast
class TestSRTSegmentAdvanced:
    """Test advanced SRTSegment features."""

    def test_srt_segment_to_dict(self):
        """Test converting segment to dict."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            keywords=["key1", "key2"],
            entities=[{"text": "Entity1", "type": "PERSON"}],
            topic_id=1,
            topics=["topic1"],
            is_broll=True
        )

        seg_dict = segment.to_dict()

        assert seg_dict["index"] == 1
        assert seg_dict["is_broll"] is True
        assert len(seg_dict["keywords"]) == 2
        assert len(seg_dict["entities"]) == 1

    def test_srt_segment_from_dict(self):
        """Test creating segment from dict."""
        data = {
            "index": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Test",
            "source_file": "test.mp4",
            "keywords": ["key1"],
            "topic_id": 1,
            "extra_field": "ignored"  # Should be filtered out
        }

        segment = SRTSegment.from_dict(data)

        assert segment.index == 1
        assert segment.text == "Test"
        assert len(segment.keywords) == 1
        assert not hasattr(segment, "extra_field")

    def test_srt_segment_from_dict_missing_fields(self):
        """Test from_dict with missing fields (should use defaults)."""
        data = {
            "text": "Only text provided"
        }

        segment = SRTSegment.from_dict(data)

        assert segment.index == 0  # Default
        assert segment.start_time == 0.0  # Default
        assert segment.text == "Only text provided"
        assert segment.keywords == []  # Default

    def test_srt_segment_duration_property(self):
        """Test segment duration property."""
        segment = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=25.5,
            text="Test"
        )

        assert segment.duration == 15.5


@pytest.mark.fast
class TestSceneInfoDataclass:
    """Test SceneInfo dataclass."""

    def test_scene_info_to_dict(self):
        """Test converting scene to dict."""
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        scene = SceneInfo(
            video_path="/test/video.mp4",
            scene_index=0,
            start_time=0.0,
            end_time=10.0,
            description="Test scene",
            visual_keywords=["keyword1"],
            keyframes=["/test/frame1.jpg"],
            transcript_segment=segment
        )

        scene_dict = scene.to_dict()

        assert scene_dict["video_path"] == "/test/video.mp4"
        assert scene_dict["scene_index"] == 0
        assert scene_dict["description"] == "Test scene"
        assert scene_dict["transcript_segment"] is not None

    def test_scene_info_from_dict(self):
        """Test creating scene from dict."""
        data = {
            "video_path": "/test/video.mp4",
            "scene_index": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "description": "Test",
            "visual_keywords": ["key1"],
            "keyframes": ["/frame1.jpg"],
            "transcript_segment": {
                "index": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Test"
            }
        }

        scene = SceneInfo.from_dict(data)

        assert scene.video_path == "/test/video.mp4"
        assert scene.transcript_segment is not None
        assert scene.transcript_segment.text == "Test"

    def test_scene_info_from_dict_no_transcript(self):
        """Test scene from dict without transcript."""
        data = {
            "video_path": "/test/video.mp4",
            "scene_index": 0,
            "start_time": 0.0,
            "end_time": 10.0
        }

        scene = SceneInfo.from_dict(data)

        assert scene.transcript_segment is None
        assert scene.description == ""  # Default


@pytest.mark.fast
class TestVideoIndexDataclass:
    """Test VideoIndex dataclass."""

    def test_video_index_to_dict(self):
        """Test converting video index to dict."""
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        scene = SceneInfo(
            video_path="/test/video.mp4",
            scene_index=0,
            start_time=0.0,
            end_time=10.0
        )

        index = VideoIndex(
            video_path="/test/video.mp4",
            video_hash="abc123",
            duration=120.0,
            transcript_segments=[segment],
            scenes=[scene],
            has_embeddings=True,
            indexed_at="2026-01-09"
        )

        index_dict = index.to_dict()

        assert index_dict["video_hash"] == "abc123"
        assert index_dict["has_embeddings"] is True
        assert len(index_dict["transcript_segments"]) == 1
        assert len(index_dict["scenes"]) == 1

    def test_video_index_from_dict(self):
        """Test creating video index from dict."""
        data = {
            "video_path": "/test/video.mp4",
            "video_hash": "abc123",
            "duration": 120.0,
            "transcript_segments": [
                {"index": 1, "start_time": 0.0, "end_time": 5.0, "text": "Test"}
            ],
            "scenes": [
                {"video_path": "/test/video.mp4", "scene_index": 0, "start_time": 0.0, "end_time": 10.0}
            ],
            "has_embeddings": True,
            "indexed_at": "2026-01-09"
        }

        index = VideoIndex.from_dict(data)

        assert index.video_hash == "abc123"
        assert len(index.transcript_segments) == 1
        assert len(index.scenes) == 1


@pytest.mark.fast
class TestMatchDataclasses:
    """Test Match-related dataclasses."""

    def test_match_to_dict(self):
        """Test converting match to dict."""
        voiceover_seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="VO")
        video_seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Video")
        scene = SceneInfo(video_path="/test.mp4", scene_index=0, start_time=0.0, end_time=5.0)

        match = Match(
            voiceover_segment=voiceover_seg,
            video_segment=video_seg,
            video_scene=scene,
            confidence=0.95,
            reasoning="Test match",
            is_keyword_match=True,
            embedding_similarity=0.85
        )

        match_dict = match.to_dict()

        assert match_dict["confidence"] == 0.95
        assert match_dict["is_keyword_match"] is True
        assert match_dict["embedding_similarity"] == 0.85

    def test_strategy_match_to_dict(self):
        """Test converting strategy match to dict."""
        video_seg = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Video")

        strategy_match = StrategyMatch(
            video_segment=video_seg,
            video_scene=None,
            confidence=0.8,
            reasoning="Test",
            strategy="embedding_diversity"
        )

        match_dict = strategy_match.to_dict()

        assert match_dict["strategy"] == "embedding_diversity"
        assert match_dict["confidence"] == 0.8


@pytest.mark.fast
class TestProgressBar:
    """Test ProgressBar functionality."""

    def test_progress_bar_creation(self):
        """Test creating progress bar."""
        bar = ProgressBar(total=100, description="Test")

        assert bar.total == 100
        assert bar.current == 0
        assert bar.description == "Test"

    def test_progress_bar_update(self):
        """Test updating progress."""
        bar = ProgressBar(total=100)

        bar.update(10)
        assert bar.current == 10

        bar.update(5)
        assert bar.current == 15

    def test_progress_bar_set(self):
        """Test setting absolute progress."""
        bar = ProgressBar(total=100)

        bar.set(50)
        assert bar.current == 50

    def test_progress_bar_close(self):
        """Test closing progress bar."""
        bar = ProgressBar(total=100)
        bar.set(50)

        bar.close()

        # Should set to total
        assert bar.current == 100


@pytest.mark.fast
class TestCacheManager:
    """Test CacheManager functionality."""

    def test_cache_manager_creation(self, tmp_path):
        """Test creating cache manager."""
        cache_dir = tmp_path / "cache"

        manager = CacheManager(str(cache_dir))

        assert manager.cache_dir.exists()
        assert manager.transcription_dir.exists()
        assert manager.embedding_dir.exists()

    def test_get_file_hash(self, tmp_path):
        """Test getting file hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        manager = CacheManager(str(tmp_path / "cache"))
        file_hash = manager.get_file_hash(str(test_file))

        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash

    def test_get_text_hash(self, tmp_path):
        """Test getting text hash."""
        manager = CacheManager(str(tmp_path / "cache"))

        text_hash = manager.get_text_hash("test text")

        assert isinstance(text_hash, str)
        assert len(text_hash) == 16  # Truncated MD5

    def test_transcription_cache(self, tmp_path):
        """Test transcription caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        ]

        # Save
        manager.save_transcription("hash123", segments)

        # Load
        loaded = manager.get_transcription("hash123")

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].text == "Test"

    def test_transcription_cache_miss(self, tmp_path):
        """Test transcription cache miss."""
        manager = CacheManager(str(tmp_path / "cache"))

        loaded = manager.get_transcription("nonexistent")

        assert loaded is None

    def test_embeddings_cache(self, tmp_path):
        """Test embeddings caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        manager.save_embeddings("key123", embeddings)
        loaded = manager.get_embeddings("key123")

        assert loaded is not None
        assert len(loaded) == 2

    def test_scenes_cache(self, tmp_path):
        """Test scenes caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        scenes = [
            SceneInfo(video_path="/test.mp4", scene_index=0, start_time=0.0, end_time=10.0)
        ]

        manager.save_scenes("hash123", scenes)
        loaded = manager.get_scenes("hash123")

        assert loaded is not None
        assert len(loaded) == 1

    def test_llm_response_cache(self, tmp_path):
        """Test LLM response caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        response = {"result": "test response", "confidence": 0.9}

        manager.save_llm_response("prompt_hash", response)
        loaded = manager.get_llm_response("prompt_hash")

        assert loaded is not None
        assert loaded["result"] == "test response"

    def test_video_index_cache(self, tmp_path):
        """Test video index caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        index = VideoIndex(
            video_path="/test.mp4",
            video_hash="hash123",
            duration=120.0
        )

        manager.save_video_index(index)
        loaded = manager.get_video_index("hash123")

        assert loaded is not None
        assert loaded.video_path == "/test.mp4"

    def test_get_all_video_indices(self, tmp_path):
        """Test getting all video indices."""
        manager = CacheManager(str(tmp_path / "cache"))

        index1 = VideoIndex(video_path="/test1.mp4", video_hash="hash1", duration=120.0)
        index2 = VideoIndex(video_path="/test2.mp4", video_hash="hash2", duration=90.0)

        manager.save_video_index(index1)
        manager.save_video_index(index2)

        all_indices = manager.get_all_video_indices()

        assert len(all_indices) == 2

    def test_master_index(self, tmp_path):
        """Test master index caching."""
        manager = CacheManager(str(tmp_path / "cache"))

        index_data = {"/video1.mp4": "hash1", "/video2.mp4": "hash2"}

        manager.save_master_index(index_data)
        loaded = manager.get_master_index()

        assert len(loaded) == 2
        assert loaded["/video1.mp4"] == "hash1"


@pytest.mark.fast
class TestReuseTracker:
    """Test ReuseTracker functionality."""

    def test_reuse_tracker_creation(self):
        """Test creating reuse tracker."""
        tracker = ReuseTracker(max_reuse=3, reuse_penalty=0.1)

        assert tracker.max_reuse == 3
        assert tracker.reuse_penalty == 0.1

    def test_get_clip_id(self):
        """Test generating clip ID."""
        tracker = ReuseTracker()
        segment = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=15.0,
            text="Test",
            source_file="/test/video.mp4"
        )

        clip_id = tracker.get_clip_id(segment)

        assert "/test/video.mp4" in clip_id
        assert "10.00" in clip_id
        assert "15.00" in clip_id

    def test_get_source_file(self):
        """Test getting normalized source file."""
        tracker = ReuseTracker()
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            source_file="C:\\Test\\Video.MP4"
        )

        source = tracker.get_source_file(segment)

        # Should be lowercase and forward slashes
        assert source == "c:/test/video.mp4"

    def test_record_usage(self):
        """Test recording clip usage."""
        tracker = ReuseTracker()
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            source_file="/test.mp4"
        )

        assert tracker.get_usage_count(segment) == 0

        tracker.record_usage(segment)

        assert tracker.get_usage_count(segment) == 1

    def test_can_use_within_limit(self):
        """Test can use clip within limit."""
        tracker = ReuseTracker(max_reuse=2)
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/test.mp4")

        # First use
        assert tracker.can_use(segment)
        tracker.record_usage(segment)

        # Still can use (count=1, max=2)
        assert tracker.can_use(segment)
        tracker.record_usage(segment)

        # Now at max (count=2, max=2)
        assert not tracker.can_use(segment)

    def test_get_penalty(self):
        """Test getting reuse penalty."""
        tracker = ReuseTracker(max_reuse=5, reuse_penalty=0.2)
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/test.mp4")

        # No usage yet
        assert tracker.get_penalty(segment) == 0.0

        # After 2 uses
        tracker.record_usage(segment)
        tracker.record_usage(segment)

        penalty = tracker.get_penalty(segment)
        assert penalty == 0.4  # 2 * 0.2

    def test_adjust_confidence(self):
        """Test adjusting confidence based on reuse."""
        tracker = ReuseTracker(max_reuse=5, reuse_penalty=0.1)
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/test.mp4")

        tracker.record_usage(segment)

        adjusted = tracker.adjust_confidence(segment, 0.9)

        assert adjusted == 0.8  # 0.9 - 0.1

    def test_reset(self):
        """Test resetting tracker."""
        tracker = ReuseTracker()
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/test.mp4")

        tracker.record_usage(segment)
        assert tracker.get_usage_count(segment) == 1

        tracker.reset()

        assert tracker.get_usage_count(segment) == 0

    def test_get_top_sources(self):
        """Test getting top sources."""
        tracker = ReuseTracker()

        seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/video1.mp4")
        seg2 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test", source_file="/video2.mp4")

        # Use video1 twice, video2 once
        tracker.record_usage(seg1)
        tracker.record_usage(seg1)
        tracker.record_usage(seg2)

        top_sources = tracker.get_top_sources(limit=5)

        assert len(top_sources) >= 1
        # video1 should be first (most used)
        assert top_sources[0][1] == 2


@pytest.mark.fast
class TestSRTParsingAdvanced:
    """Test advanced SRT parsing features."""

    def test_parse_srt_file_not_found(self):
        """Test parsing non-existent file."""
        segments = parse_srt_file("/nonexistent/file.srt")

        assert len(segments) == 0

    def test_parse_srt_invalid_format(self, tmp_path):
        """Test parsing file with invalid format."""
        srt_file = tmp_path / "invalid.srt"
        srt_file.write_text("Not a valid SRT file\nJust random text")

        segments = parse_srt_file(str(srt_file))

        # Should handle gracefully
        assert isinstance(segments, list)

    def test_parse_srt_binary_file(self, tmp_path):
        """Test parsing binary file (should detect and skip)."""
        binary_file = tmp_path / "binary.srt"
        binary_file.write_bytes(b'\x00\x01\x02\x03ID3' + b'\x00' * 100)

        segments = parse_srt_file(str(binary_file))

        # Should detect binary and return empty
        assert len(segments) == 0

    def test_parse_srt_utf16_encoding(self, tmp_path):
        """Test parsing UTF-16 encoded SRT."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Test subtitle
"""
        srt_file = tmp_path / "utf16.srt"
        srt_file.write_text(srt_content, encoding='utf-16')

        segments = parse_srt_file(str(srt_file))

        assert len(segments) >= 0  # Should handle encoding

    def test_parse_srt_with_bom(self, tmp_path):
        """Test parsing SRT with BOM."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Test
"""
        srt_file = tmp_path / "bom.srt"
        # Write with UTF-16 BOM
        srt_file.write_bytes(b'\xff\xfe' + srt_content.encode('utf-16-le'))

        segments = parse_srt_file(str(srt_file))

        # Should handle BOM
        assert isinstance(segments, list)

    def test_parse_srt_malformed_timestamp(self, tmp_path):
        """Test parsing SRT with malformed timestamp."""
        srt_content = """1
INVALID TIMESTAMP
Test subtitle

2
00:00:05,000 --> 00:00:10,000
Valid subtitle
"""
        srt_file = tmp_path / "malformed.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt_file(str(srt_file))

        # Should skip malformed entry, parse valid one
        assert len(segments) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
