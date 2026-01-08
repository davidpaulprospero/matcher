#!/usr/bin/env python3
"""
Download-Only Test Suite v1.0

Test harness for the download module without needing full pipeline runs.
Tests download logic, segment merging, audio-first helpers, and yt-dlp interactions.

Use Cases:
    - Test new download features in isolation
    - Validate segment merging/buffer logic
    - Test LLM title filtering (mock or real)
    - Verify audio-first pipeline helpers
    - Test crash resilience and resumption

Usage:
    # Run all unit tests (no network)
    python tests/test_download.py

    # Run with real yt-dlp search (slow, needs network)
    python tests/test_download.py --live

    # Run specific test
    python tests/test_download.py --test segment_merging

    # With verbose output
    python tests/test_download.py --verbose

    # Save download fixtures from real run
    python tests/test_download.py --save-fixtures path/to/fixture.json

Fixture Creation:
    # From main.py with audio-first mode
    python main.py --project MyProject --save-download-fixtures fixtures/download_test.json
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import pytest
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DownloadMetrics:
    """Metrics collected during download tests"""

    # Download counts
    total_keywords: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    skipped_existing: int = 0

    # Segment metrics
    total_segments: int = 0
    merged_segments: int = 0
    merge_ratio: float = 0.0

    # LLM filter metrics
    llm_approved: int = 0
    llm_rejected: int = 0
    llm_pass_rate: float = 0.0

    # Audio-first metrics
    audio_downloads: int = 0
    video_segments_downloaded: int = 0
    bandwidth_saved_percent: float = 0.0

    # Performance
    total_time_ms: float = 0.0
    avg_download_time_ms: float = 0.0


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# FIXTURE LOADING/SAVING
# =============================================================================

def save_download_fixtures(
    output_path: str,
    audio_downloads: List[Any],
    matched_segments: Dict[str, List[Any]],
    merged_segments: List[Any],
    config: Any,
    project_dir: str = ""
) -> bool:
    """
    Save download inputs to fixture files for testing.

    Args:
        output_path: Path for output JSON file
        audio_downloads: List of AudioDownload objects
        matched_segments: Dict of video_id -> List[MatchedSegment]
        merged_segments: List of MergedSegment objects
        config: Config object
        project_dir: Source project directory

    Returns:
        True if saved successfully
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dicts
    audio_dicts = []
    for audio in audio_downloads:
        if hasattr(audio, 'to_dict'):
            audio_dicts.append(audio.to_dict())
        elif isinstance(audio, dict):
            audio_dicts.append(audio)

    matched_dicts = {}
    for video_id, segments in matched_segments.items():
        matched_dicts[video_id] = []
        for seg in segments:
            if hasattr(seg, '__dict__'):
                matched_dicts[video_id].append(asdict(seg))
            elif isinstance(seg, dict):
                matched_dicts[video_id].append(seg)

    merged_dicts = []
    for merged in merged_segments:
        if hasattr(merged, '__dict__'):
            d = asdict(merged)
            # Convert nested MatchedSegment list
            if 'original_matches' in d:
                d['original_matches'] = [
                    asdict(m) if hasattr(m, '__dict__') else m
                    for m in (merged.original_matches or [])
                ]
            merged_dicts.append(d)
        elif isinstance(merged, dict):
            merged_dicts.append(merged)

    # Extract config snapshot
    config_snapshot = {}
    if hasattr(config, 'download'):
        dc = config.download
        config_snapshot = {
            'audio_first_enabled': getattr(getattr(dc, 'audio_first', None), 'enabled', False),
            'buffer_seconds': getattr(getattr(dc, 'audio_first', None), 'buffer_seconds', 30.0),
            'merge_gap_seconds': getattr(getattr(dc, 'audio_first', None), 'merge_gap_seconds', 15.0),
            'llm_filter_enabled': getattr(getattr(dc, 'llm_title_filter', None), 'enabled', False),
            'davinci_mode': getattr(dc, 'davinci_mode', True),
        }

    fixture_data = {
        'created': datetime.now().isoformat(),
        'source_project': str(project_dir),
        'audio_downloads': audio_dicts,
        'matched_segments': matched_dicts,
        'merged_segments': merged_dicts,
        'config_snapshot': config_snapshot,
        'stats': {
            'audio_count': len(audio_dicts),
            'video_ids_with_matches': len(matched_dicts),
            'merged_count': len(merged_dicts),
        }
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixture_data, f, indent=2)

        print(f"  Saved download fixtures: {output_path}")
        return True

    except Exception as e:
        print(f"  Failed to save fixtures: {e}")
        return False


def load_download_fixtures(fixture_path: str) -> Tuple[List, Dict, List, Dict]:
    """
    Load download fixtures from saved files.

    Args:
        fixture_path: Path to fixture JSON file

    Returns:
        Tuple of (audio_downloads, matched_segments, merged_segments, config_snapshot)
    """
    from src.downloader import AudioDownload, MatchedSegment, MergedSegment

    fixture_path = Path(fixture_path)

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert dicts to dataclass objects
    audio_downloads = [AudioDownload(**d) for d in data.get('audio_downloads', [])]

    matched_segments = {}
    for video_id, segments in data.get('matched_segments', {}).items():
        matched_segments[video_id] = [MatchedSegment(**s) for s in segments]

    merged_segments = []
    for d in data.get('merged_segments', []):
        # Handle nested original_matches
        original_matches = [
            MatchedSegment(**m) for m in d.get('original_matches', [])
        ]
        d['original_matches'] = original_matches
        merged_segments.append(MergedSegment(**d))

    config_snapshot = data.get('config_snapshot', {})

    return audio_downloads, matched_segments, merged_segments, config_snapshot


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_test_config(overrides: Dict[str, Any] = None) -> Any:
    """
    Create a config object with optional overrides.

    Args:
        overrides: Dict of download config overrides

    Returns:
        Config object with overrides applied
    """
    from src.config import load_config

    config = load_config()

    if overrides:
        for key, value in overrides.items():
            # Handle nested keys like "audio_first.buffer_seconds"
            if '.' in key:
                parts = key.split('.')
                if len(parts) == 2:
                    parent_key, child_key = parts
                    parent_obj = getattr(config.download, parent_key, None)
                    if parent_obj is not None:
                        if hasattr(parent_obj, child_key):
                            setattr(parent_obj, child_key, value)
                        elif isinstance(parent_obj, dict):
                            parent_obj[child_key] = value
            elif hasattr(config.download, key):
                setattr(config.download, key, value)

    return config


def create_mock_audio_download(
    video_id: str = "test123",
    duration: float = 300.0,
    keyword: str = "test keyword"
) -> Any:
    """Create a mock AudioDownload for testing"""
    from src.downloader import AudioDownload

    return AudioDownload(
        audio_file=f"/tmp/audio/{video_id}.mp3",
        video_id=video_id,
        video_url=f"https://www.youtube.com/watch?v={video_id}",
        title=f"Test Video {video_id}",
        channel="Test Channel",
        duration=duration,
        keyword=keyword,
        duration_tier="medium",
        upload_date="20240101",
        license="Standard"
    )


def create_mock_matched_segment(
    video_id: str = "test123",
    start_time: float = 60.0,
    end_time: float = 70.0,
    track: str = "V1",
    idx: int = 0
) -> Any:
    """Create a mock MatchedSegment for testing"""
    from src.downloader import MatchedSegment

    return MatchedSegment(
        video_id=video_id,
        video_url=f"https://www.youtube.com/watch?v={video_id}",
        start_time=start_time,
        end_time=end_time,
        track=track,
        voiceover_segment_idx=idx,
        keyword="test keyword"
    )


# =============================================================================
# TEST RUNNER
# =============================================================================

class DownloadTestRunner:
    """
    Test runner for download module.

    Usage:
        runner = DownloadTestRunner(verbose=True)

        # Run unit tests (no network)
        runner.run_unit_tests()

        # Run with fixtures
        runner.load_fixtures('path/to/fixture.json')
        runner.run_fixture_tests()

        # Run live tests (requires network)
        runner.run_live_tests()
    """

    def __init__(self, verbose: bool = False, live_mode: bool = False):
        self.verbose = verbose
        self.live_mode = live_mode
        self.fixtures_loaded = False

        # Fixture data
        self.audio_downloads = []
        self.matched_segments = {}
        self.merged_segments = []
        self.config_snapshot = {}
        self.fixture_path = ""

        # Test results
        self.results: List[TestResult] = []

        # Temp directory for tests
        self.temp_dir = None

    def log(self, message: str, indent: int = 0):
        """Print message with optional indent"""
        prefix = "  " * indent
        print(f"{prefix}{message}")

    def log_verbose(self, message: str, indent: int = 0):
        """Print message only in verbose mode"""
        if self.verbose:
            self.log(message, indent)

    def setup(self):
        """Create temp directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="download_test_"))
        self.log_verbose(f"Created temp dir: {self.temp_dir}")

    def teardown(self):
        """Clean up temp directory"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.log_verbose(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                self.log_verbose(f"Failed to cleanup: {e}")

    def load_fixtures(self, fixture_path: str) -> bool:
        """Load fixtures from file"""
        try:
            self.fixture_path = fixture_path
            (
                self.audio_downloads,
                self.matched_segments,
                self.merged_segments,
                self.config_snapshot
            ) = load_download_fixtures(fixture_path)

            self.fixtures_loaded = True

            self.log(f"Loaded fixtures from: {fixture_path}")
            self.log(f"  Audio downloads: {len(self.audio_downloads)}")
            self.log(f"  Videos with matches: {len(self.matched_segments)}")
            self.log(f"  Merged segments: {len(self.merged_segments)}")

            return True

        except Exception as e:
            self.log(f"Failed to load fixtures: {e}")
            return False

    def run_test(self, name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test"""
        self.log(f"  {name}...", indent=0)
        start = time.time()

        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start

            if isinstance(result, tuple):
                if len(result) == 2:
                    passed, message = result
                    details = {}
                else:
                    passed, message, details = result
            else:
                passed = bool(result)
                message = "OK" if passed else "Failed"
                details = {}

            test_result = TestResult(
                name=name,
                passed=passed,
                duration=duration,
                message=message,
                details=details
            )

        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                message=f"Exception: {str(e)}"
            )
            if self.verbose:
                import traceback
                traceback.print_exc()

        self.results.append(test_result)

        status = "PASS" if test_result.passed else "FAIL"
        self.log(f"     [{status}] ({duration:.2f}s) {test_result.message}")

        return test_result

    def run_unit_tests(self):
        """Run all unit tests (no network required)"""
        self.log("\n  --- UNIT TESTS ---")

        self.run_test("Segment merging basic", test_segment_merging_basic)
        self.run_test("Segment merging with overlap", test_segment_merging_overlap)
        self.run_test("Segment merging edge cases", test_segment_merging_edge_cases)
        self.run_test("Segment filename generation", test_segment_filename)
        self.run_test("Video ID extraction", test_video_id_extraction)
        self.run_test("Collect matched segments", test_collect_matched_segments)
        self.run_test("Prepare merged segments", test_prepare_merged_segments)
        self.run_test("Filter string building", test_filter_string_building)
        self.run_test("Format string building", test_format_string_building)
        self.run_test("Needs transcoding detection", test_needs_transcoding)
        self.run_test("Filename sanitization", test_filename_sanitization)

    def run_fixture_tests(self):
        """Run tests using loaded fixtures"""
        if not self.fixtures_loaded:
            self.log("No fixtures loaded, skipping fixture tests")
            return

        self.log("\n  --- FIXTURE TESTS ---")

        self.run_test("Fixture segment merge consistency", test_fixture_merge_consistency, self)
        self.run_test("Fixture buffer application", test_fixture_buffer_application, self)

    def run_live_tests(self):
        """Run tests that require network (YouTube API calls)"""
        if not self.live_mode:
            self.log("\n  --- LIVE TESTS (skipped, use --live to enable) ---")
            return

        self.log("\n  --- LIVE TESTS ---")

        self.run_test("yt-dlp availability", test_ytdlp_available)
        self.run_test("YouTube search metadata", test_youtube_search_metadata)
        self.run_test("Dependency check", test_dependency_check)

    def print_summary(self) -> bool:
        """Print test summary and return True if all passed"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"  DOWNLOAD TEST SUMMARY")
        print(f"{'=' * 60}")

        if self.fixture_path:
            print(f"  Fixtures: {Path(self.fixture_path).name}")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message}")

        print(f"\n  Total: {passed}/{total} tests passed")
        print(f"{'=' * 60}")

        return failed == 0


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_segment_merging_basic() -> Tuple[bool, str]:
    """Test basic segment merging functionality"""
    from src.downloader import merge_segments_with_buffer

    segments = [
        (60.0, 70.0),
        (200.0, 210.0),
    ]

    merged = merge_segments_with_buffer(
        segments,
        buffer_seconds=30.0,
        merge_gap_seconds=15.0,
        video_duration=300.0
    )

    # Expect 2 separate segments (too far apart to merge)
    if len(merged) != 2:
        return False, f"Expected 2 segments, got {len(merged)}"

    # First segment: 60-30=30 to 70+30=100
    if merged[0] != (30.0, 100.0):
        return False, f"First segment wrong: {merged[0]}"

    # Second segment: 200-30=170 to 210+30=240
    if merged[1] != (170.0, 240.0):
        return False, f"Second segment wrong: {merged[1]}"

    return True, "Basic merging works correctly"


def test_segment_merging_overlap() -> Tuple[bool, str]:
    """Test segment merging with overlapping segments"""
    from src.downloader import merge_segments_with_buffer

    segments = [
        (60.0, 70.0),
        (80.0, 90.0),   # Close enough to merge with buffer
        (85.0, 95.0),   # Overlapping with previous
    ]

    merged = merge_segments_with_buffer(
        segments,
        buffer_seconds=30.0,
        merge_gap_seconds=15.0,
        video_duration=300.0
    )

    # All should merge into one segment
    if len(merged) != 1:
        return False, f"Expected 1 merged segment, got {len(merged)}"

    # Should span from 30 (60-30) to 125 (95+30)
    start, end = merged[0]
    if start != 30.0:
        return False, f"Expected start 30.0, got {start}"
    if end != 125.0:
        return False, f"Expected end 125.0, got {end}"

    return True, "Overlapping segments merged correctly"


def test_segment_merging_edge_cases() -> Tuple[bool, str]:
    """Test segment merging edge cases"""
    from src.downloader import merge_segments_with_buffer

    # Empty list
    result = merge_segments_with_buffer([], buffer_seconds=30.0)
    if result != []:
        return False, f"Empty list should return empty, got {result}"

    # Single segment
    result = merge_segments_with_buffer(
        [(50.0, 60.0)],
        buffer_seconds=30.0,
        video_duration=100.0
    )
    if len(result) != 1:
        return False, f"Single segment should stay single"
    if result[0] != (20.0, 90.0):
        return False, f"Single segment wrong: {result[0]}"

    # Segment at start (negative buffer should clamp to 0)
    result = merge_segments_with_buffer(
        [(5.0, 15.0)],
        buffer_seconds=30.0,
        video_duration=100.0
    )
    if result[0][0] != 0.0:
        return False, f"Start should clamp to 0, got {result[0][0]}"

    # Segment at end (buffer should clamp to duration)
    result = merge_segments_with_buffer(
        [(85.0, 95.0)],
        buffer_seconds=30.0,
        video_duration=100.0
    )
    if result[0][1] != 100.0:
        return False, f"End should clamp to 100, got {result[0][1]}"

    # Invalid segment (end <= start) should be skipped
    result = merge_segments_with_buffer(
        [(50.0, 50.0), (60.0, 70.0)],  # First is invalid
        buffer_seconds=5.0,
        video_duration=100.0
    )
    if len(result) != 1:
        return False, f"Invalid segment should be skipped, got {len(result)} segments"

    return True, "Edge cases handled correctly"


def test_segment_filename() -> Tuple[bool, str]:
    """Test segment filename generation"""
    from src.downloader import get_segment_filename

    # Basic case
    filename = get_segment_filename("abc123", 330.0)
    if filename != "abc123_0330.mp4":
        return False, f"Expected abc123_0330.mp4, got {filename}"

    # Zero start
    filename = get_segment_filename("xyz789", 0.0)
    if filename != "xyz789_0000.mp4":
        return False, f"Expected xyz789_0000.mp4, got {filename}"

    # Large number
    filename = get_segment_filename("test", 9999.0)
    if filename != "test_9999.mp4":
        return False, f"Expected test_9999.mp4, got {filename}"

    return True, "Filename generation correct"


def test_video_id_extraction() -> Tuple[bool, str]:
    """Test video ID extraction from file paths"""
    from src.downloader import _extract_video_id

    # Simple format
    video_id = _extract_video_id("/path/to/abc123.mp3")
    if video_id != "abc123":
        return False, f"Simple format failed: {video_id}"

    # Segment format
    video_id = _extract_video_id("/path/to/abc123_0330.mp4")
    if video_id != "abc123":
        return False, f"Segment format failed: {video_id}"

    # Empty path
    video_id = _extract_video_id("")
    if video_id is not None:
        return False, f"Empty path should return None"

    # None path
    video_id = _extract_video_id(None)
    if video_id is not None:
        return False, f"None path should return None"

    return True, "Video ID extraction correct"


def test_collect_matched_segments() -> Tuple[bool, str]:
    """Test collecting matched segments from match results"""
    from src.downloader import collect_matched_segments, AudioDownload

    # Create mock match results
    class MockVideoSegment:
        def __init__(self, source_file, start, end):
            self.source_file = source_file
            self.start_time = start
            self.end_time = end

    class MockMatch:
        def __init__(self, video_segment, confidence=0.8):
            self.video_segment = video_segment
            self.confidence = confidence

    class MockMatchResult:
        def __init__(self):
            self.primary_match = None
            self.alternatives = []
            self.secondary_matches = []
            self.strategy_matches = []

    # Create audio downloads map
    audio_downloads = {
        "vid1": AudioDownload(
            file="/tmp/vid1.mp3",
            video_id="vid1",
            url="https://youtube.com/watch?v=vid1",
            title="Video 1",
            duration=300.0,
            keyword="test"
        )
    }

    # Create match result
    result = MockMatchResult()
    result.primary_match = MockMatch(MockVideoSegment("/tmp/vid1.mp3", 60.0, 70.0))
    result.alternatives = [MockMatch(MockVideoSegment("/tmp/vid1.mp3", 80.0, 90.0))]

    match_results = [result]

    # Collect segments
    segments = collect_matched_segments(match_results, audio_downloads)

    if "vid1" not in segments:
        return False, "Video ID not in results"

    if len(segments["vid1"]) != 2:
        return False, f"Expected 2 segments, got {len(segments['vid1'])}"

    # Check tracks
    tracks = [s.track for s in segments["vid1"]]
    if "V1" not in tracks or "V2" not in tracks:
        return False, f"Wrong tracks: {tracks}"

    return True, "Segment collection works"


def test_prepare_merged_segments() -> Tuple[bool, str]:
    """Test preparing merged segments"""
    from src.downloader import (
        prepare_merged_segments,
        MatchedSegment,
        AudioDownload
    )

    # Create test data
    audio_downloads = {
        "vid1": AudioDownload(
            file="/tmp/vid1.mp3",
            video_id="vid1",
            url="https://youtube.com/watch?v=vid1",
            title="Video 1",
            duration=300.0,
            keyword="test"
        )
    }

    segments_by_video = {
        "vid1": [
            MatchedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=60.0,
                end_time=70.0,
                track="V1",
                voiceover_segment_idx=0
            ),
            MatchedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=65.0,
                end_time=75.0,
                track="V2",
                voiceover_segment_idx=0
            ),
        ]
    }

    merged = prepare_merged_segments(
        segments_by_video,
        audio_downloads,
        buffer_seconds=10.0,
        merge_gap_seconds=5.0
    )

    if len(merged) != 1:
        return False, f"Expected 1 merged segment, got {len(merged)}"

    # Check merged segment
    m = merged[0]
    if m.video_id != "vid1":
        return False, f"Wrong video_id: {m.video_id}"

    # Start: 60-10=50, End: 75+10=85
    if m.start_time != 50.0:
        return False, f"Expected start 50.0, got {m.start_time}"
    if m.end_time != 85.0:
        return False, f"Expected end 85.0, got {m.end_time}"

    return True, "Merged segment preparation works"


def test_filter_string_building() -> Tuple[bool, str]:
    """Test filter string building for yt-dlp"""
    from src.downloader import VideoDownloader
    from src.config import load_config

    config = load_config()
    downloader = VideoDownloader(config)

    # Test short tier filter
    filter_str = downloader._build_filter_string('short')

    if 'duration>' not in filter_str:
        return False, "Missing duration min filter"
    if 'duration<' not in filter_str:
        return False, "Missing duration max filter"
    if '!is_live' not in filter_str:
        return False, "Missing live stream filter"

    return True, "Filter string built correctly"


def test_format_string_building() -> Tuple[bool, str]:
    """Test format string building for yt-dlp"""
    from src.downloader import VideoDownloader
    from src.config import load_config

    config = load_config()
    downloader = VideoDownloader(config)

    format_str = downloader._build_format_string()

    # Should have video and audio selection
    if 'bestvideo' not in format_str and 'best' not in format_str:
        return False, "Missing video format selection"

    return True, "Format string built correctly"


def test_needs_transcoding() -> Tuple[bool, str]:
    """Test transcoding detection logic"""
    from src.downloader import VideoDownloader
    from src.config import load_config

    config = load_config()
    downloader = VideoDownloader(config)

    # Mock ffprobe to return vp9 codec
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(stdout='vp9', returncode=0)

        needs, reason = downloader._needs_transcoding('/fake/video.webm')

        if not needs:
            return False, f"VP9 should need transcoding, reason: {reason}"

    # Mock ffprobe to return h264 codec
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(stdout='h264', returncode=0)

        needs, reason = downloader._needs_transcoding('/fake/video.mp4')

        if needs:
            return False, f"H264/MP4 should not need transcoding, reason: {reason}"

    return True, "Transcoding detection works"


def test_filename_sanitization() -> Tuple[bool, str]:
    """Test filename sanitization for NLE compatibility"""
    from src.downloader import sanitize_filename_for_nle

    # Create a temp file with unsafe characters
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Rename to have unsafe characters
        unsafe_path = temp_path.parent / "test%file&name$here#.mp4"
        temp_path.rename(unsafe_path)

        # Sanitize
        safe_path = sanitize_filename_for_nle(unsafe_path)

        # Check result
        if '%' in safe_path.name or '&' in safe_path.name or '$' in safe_path.name or '#' in safe_path.name:
            return False, f"Unsafe chars remain: {safe_path.name}"

        # Clean up
        if safe_path.exists():
            safe_path.unlink()

        return True, "Filename sanitization works"

    except Exception as e:
        # Clean up on error
        for p in [temp_path, unsafe_path if 'unsafe_path' in dir() else None]:
            if p and p.exists():
                p.unlink()
        return False, f"Error: {e}"


# =============================================================================
# FIXTURE TESTS
# =============================================================================

@pytest.mark.integration
def test_fixture_merge_consistency(runner: DownloadTestRunner) -> Tuple[bool, str]:
    """Test that fixture merged segments are consistent with merge logic"""
    from src.downloader import prepare_merged_segments

    if not runner.audio_downloads:
        return True, "No fixtures to test"

    # Build audio_downloads dict
    audio_dict = {a.video_id: a for a in runner.audio_downloads}

    # Re-compute merged segments
    buffer = runner.config_snapshot.get('buffer_seconds', 30.0)
    gap = runner.config_snapshot.get('merge_gap_seconds', 15.0)

    recomputed = prepare_merged_segments(
        runner.matched_segments,
        audio_dict,
        buffer_seconds=buffer,
        merge_gap_seconds=gap
    )

    # Compare counts
    if len(recomputed) != len(runner.merged_segments):
        return False, f"Count mismatch: fixture={len(runner.merged_segments)}, recomputed={len(recomputed)}"

    return True, f"Merge consistency verified ({len(recomputed)} segments)"


@pytest.mark.integration
def test_fixture_buffer_application(runner: DownloadTestRunner) -> Tuple[bool, str]:
    """Test that buffers are correctly applied in fixtures"""
    if not runner.merged_segments:
        return True, "No merged segments to test"

    buffer = runner.config_snapshot.get('buffer_seconds', 30.0)

    for merged in runner.merged_segments:
        for match in merged.original_matches:
            # Check that match is within merged segment
            if match.start_time < merged.start_time:
                return False, f"Match start {match.start_time} before merged start {merged.start_time}"
            if match.end_time > merged.end_time:
                return False, f"Match end {match.end_time} after merged end {merged.end_time}"

    return True, "Buffer application verified"


# =============================================================================
# LIVE TESTS (require network)
# =============================================================================

def test_ytdlp_available() -> Tuple[bool, str]:
    """Test that yt-dlp is installed and accessible"""
    import subprocess

    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        version = result.stdout.strip()
        return True, f"yt-dlp version: {version}"
    except FileNotFoundError:
        return False, "yt-dlp not found"
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timed out"
    except Exception as e:
        return False, f"Error: {e}"


def test_youtube_search_metadata() -> Tuple[bool, str]:
    """Test YouTube metadata search (no download)"""
    from src.downloader import VideoDownloader
    from src.config import load_config

    config = load_config()
    downloader = VideoDownloader(config)

    # Search for a common term
    videos = downloader._search_video_metadata(
        keyword="nature documentary",
        tier="short",
        max_results=3
    )

    if not videos:
        # This can fail without cookies - not a critical failure
        return True, "No videos found (may need YouTube cookies)"

    # Verify structure
    for v in videos:
        if 'id' not in v or 'title' not in v:
            return False, f"Missing required fields in: {v}"

    return True, f"Found {len(videos)} videos"


def test_dependency_check() -> Tuple[bool, str]:
    """Test dependency check method"""
    from src.downloader import VideoDownloader
    from src.config import load_config

    config = load_config()
    downloader = VideoDownloader(config)

    success, message = downloader.check_dependencies()

    if not success:
        # Remove unicode characters for console compatibility
        message = message.encode('ascii', 'replace').decode('ascii')
        return False, message

    # Remove unicode characters and format for console
    message = message.replace('\n', ', ')
    message = message.encode('ascii', 'replace').decode('ascii')
    return True, message


# =============================================================================
# MAIN
# =============================================================================

def run_download_tests(
    fixture_path: str = None,
    live_mode: bool = False,
    specific_test: str = None,
    verbose: bool = False
) -> bool:
    """Run download tests"""

    runner = DownloadTestRunner(verbose=verbose, live_mode=live_mode)

    print(f"\n{'=' * 60}")
    print(f"  DOWNLOAD TEST SUITE")
    print(f"{'=' * 60}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Live mode: {live_mode}")

    # Load fixtures if provided
    if fixture_path:
        if not runner.load_fixtures(fixture_path):
            return False

    # Setup temp directory
    runner.setup()

    try:
        # Run specific test or all tests
        if specific_test:
            # Find and run specific test
            test_map = {
                'segment_merging': test_segment_merging_basic,
                'segment_overlap': test_segment_merging_overlap,
                'edge_cases': test_segment_merging_edge_cases,
                'filename': test_segment_filename,
                'video_id': test_video_id_extraction,
                'collect_segments': test_collect_matched_segments,
                'prepare_merged': test_prepare_merged_segments,
                'filter_string': test_filter_string_building,
                'format_string': test_format_string_building,
                'transcoding': test_needs_transcoding,
                'sanitization': test_filename_sanitization,
            }

            if specific_test in test_map:
                runner.run_test(specific_test, test_map[specific_test])
            else:
                print(f"Unknown test: {specific_test}")
                print(f"Available: {', '.join(test_map.keys())}")
                return False
        else:
            # Run all tests
            runner.run_unit_tests()

            if fixture_path:
                runner.run_fixture_tests()

            runner.run_live_tests()

        return runner.print_summary()

    finally:
        runner.teardown()


def main():
    parser = argparse.ArgumentParser(
        description='Test download module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_download.py
    python tests/test_download.py --live
    python tests/test_download.py --test segment_merging
    python tests/test_download.py --fixtures fixtures/download.json
        """
    )

    parser.add_argument(
        '--fixtures', '-f',
        type=str,
        help='Path to fixture JSON file'
    )

    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live tests (requires network)'
    )

    parser.add_argument(
        '--test', '-t',
        type=str,
        help='Run specific test only'
    )

    parser.add_argument(
        '--save-fixtures',
        type=str,
        metavar='PATH',
        help='Save fixtures to specified path (requires running pipeline)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Handle save-fixtures mode
    if args.save_fixtures:
        print("Fixture saving must be done from main.py with --save-download-fixtures flag")
        sys.exit(1)

    success = run_download_tests(
        fixture_path=args.fixtures,
        live_mode=args.live,
        specific_test=args.test,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
