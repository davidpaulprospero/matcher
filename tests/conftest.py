"""
Pytest configuration for test suite.

Provides fixtures and marks for integration tests that require
external resources (videos, API keys, etc.).

Test Markers:
    - unit: Fast unit tests with no external dependencies (<1 second)
    - integration: Tests requiring external resources (API keys, videos, etc.)
    - slow: Tests that take longer than 5 seconds to run
    - flaky: Tests with known intermittent failures (network issues, timing, etc.)

Usage:
    pytest tests/ -m unit              # Run only unit tests
    pytest tests/ -m "not slow"       # Skip slow tests
    pytest tests/ -m "integration and not flaky"  # Run integration tests, skip flaky ones
"""

import json
import os
import pytest
from pathlib import Path
from typing import List, Optional, Dict, Any


# =============================================================================
# FIXTURE DATA LOADER (US-86-002, Sprint 86)
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixtures() -> Dict[str, Any]:
    """
    Load all test fixture data files.

    Returns a dictionary with keys:
        - voiceover: sample.srt content as string
        - video: sample_video.json as dict
        - caption: sample_captions.json as dict
        - match: sample_matches.json as dict

    Usage:
        def test_something(fixtures):
            voiceover_text = fixtures['voiceover']
            video_data = fixtures['video']
            caption_data = fixtures['caption']
            match_data = fixtures['match']
    """
    return {
        'voiceover': (FIXTURES_DIR / 'voiceover' / 'sample.srt').read_text(encoding='utf-8'),
        'video': json.loads((FIXTURES_DIR / 'video' / 'sample_video.json').read_text(encoding='utf-8')),
        'caption': json.loads((FIXTURES_DIR / 'caption' / 'sample_captions.json').read_text(encoding='utf-8')),
        'match': json.loads((FIXTURES_DIR / 'match' / 'sample_matches.json').read_text(encoding='utf-8')),
    }


@pytest.fixture
def fixture_data() -> Dict[str, Any]:
    """
    Fixture providing access to all test fixture data files.

    Returns:
        Dictionary with fixture data for voiceover, video, caption, and match.
    """
    return fixtures()


# =============================================================================
# EXTERNAL RESOURCE DETECTION (US-002, Sprint 27)
# =============================================================================
#
# These helpers detect available external resources at test collection time.
# Use them with @pytest.mark.skipif for conditional test execution based on
# environment availability rather than permanent skips.
# =============================================================================

def has_gemini_api_key() -> bool:
    """Check if Gemini API key is available."""
    return bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))


def has_voyage_api_key() -> bool:
    """Check if Voyage AI API key is available."""
    return bool(os.getenv('VOYAGE_API_KEY'))


def has_cookies_file() -> bool:
    """Check if cookies.txt file exists for yt-dlp."""
    # Check common locations
    locations = [
        Path.cwd() / 'cookies.txt',
        Path.home() / 'cookies.txt',
        Path.home() / '.config' / 'yt-dlp' / 'cookies.txt',
    ]
    return any(p.exists() for p in locations)


def has_sentence_transformers() -> bool:
    """Check if sentence-transformers library is available."""
    try:
        import sentence_transformers
        return True
    except (ImportError, TypeError):
        return False


def has_opentimelineio() -> bool:
    """Check if opentimelineio library is available."""
    try:
        import opentimelineio
        return True
    except ImportError:
        return False


def has_numpy() -> bool:
    """Check if numpy is available."""
    try:
        import numpy
        return True
    except ImportError:
        return False


def has_pillow() -> bool:
    """Check if PIL/Pillow is available."""
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


def has_test_project(path: Optional[str] = None) -> bool:
    """
    Check if a test project directory exists with required structure.

    Args:
        path: Optional path to check. If None, checks common test locations.
    """
    if path:
        test_path = Path(path)
        return (test_path.exists() and
                (test_path / 'checkpoint.json').exists())

    # Check common test project locations
    locations = [
        Path.cwd() / '_test_project',
        Path('E:/Edit Job/_ralph_test'),
        Path.home() / 'test_project',
    ]
    return any(
        (p.exists() and (p / 'checkpoint.json').exists())
        for p in locations
    )


def get_test_project_path() -> Optional[Path]:
    """Get path to test project if one exists."""
    locations = [
        Path.cwd() / '_test_project',
        Path('E:/Edit Job/_ralph_test'),
        Path.home() / 'test_project',
    ]
    for p in locations:
        if p.exists() and (p / 'checkpoint.json').exists():
            return p
    return None


# Module-level constants for skipif decorators
HAS_GEMINI_API = has_gemini_api_key()
HAS_VOYAGE_API = has_voyage_api_key()
HAS_COOKIES = has_cookies_file()
HAS_SENTENCE_TRANSFORMERS = has_sentence_transformers()
HAS_OTIO = has_opentimelineio()
HAS_NUMPY = has_numpy()
HAS_PILLOW = has_pillow()
HAS_TEST_PROJECT = has_test_project()


# Reusable skip conditions
SKIP_NO_GEMINI = pytest.mark.skipif(
    not HAS_GEMINI_API,
    reason="Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
)

SKIP_NO_VOYAGE = pytest.mark.skipif(
    not HAS_VOYAGE_API,
    reason="Requires VOYAGE_API_KEY environment variable"
)

SKIP_NO_COOKIES = pytest.mark.skipif(
    not HAS_COOKIES,
    reason="Requires cookies.txt for yt-dlp video downloads"
)

SKIP_NO_SENTENCE_TRANSFORMERS = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="Requires sentence-transformers library (pip install sentence-transformers)"
)

SKIP_NO_OTIO = pytest.mark.skipif(
    not HAS_OTIO,
    reason="Requires opentimelineio library (pip install opentimelineio)"
)

SKIP_NO_NUMPY = pytest.mark.skipif(
    not HAS_NUMPY,
    reason="Requires numpy library"
)

SKIP_NO_TEST_PROJECT = pytest.mark.skipif(
    not HAS_TEST_PROJECT,
    reason="Requires test project directory with checkpoint.json"
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as fast unit tests (no external resources, <1s)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require external resources)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers",
        "flaky: marks tests with known intermittent failures"
    )
    config.addinivalue_line(
        "markers",
        "recorded: marks tests that use recorded API responses (US-149-010)"
    )


@pytest.fixture
def srt_path(tmp_path) -> Path:
    """
    Fixture for SRT file path.

    Creates a sample SRT file in tmp_path for testing. If you need a real
    SRT file, use a test project directory or provide one via environment.

    Returns:
        Path to a sample SRT file for testing
    """
    srt_file = tmp_path / "test_voiceover.srt"
    # Create minimal valid SRT content
    srt_content = """1
00:00:00,000 --> 00:00:05,000
Welcome to the tutorial.

2
00:00:05,000 --> 00:00:10,000
We will learn about Python programming.

3
00:00:10,000 --> 00:00:15,000
Let's get started with the basics.
"""
    srt_file.write_text(srt_content, encoding='utf-8')
    return srt_file


@pytest.fixture
def config(tmp_path):
    """Fixture for config."""
    from unittest.mock import Mock, MagicMock

    mock_config = Mock()
    mock_config.cache = Mock(cache_dir=str(tmp_path / "cache"))
    mock_config.output = Mock(output_dir=str(tmp_path / "output"))
    mock_config.transcription = Mock(model="base", language="en")
    mock_config.matching = Mock(
        min_confidence=0.5,
        location_matching=Mock(enabled=False, geonames_username="")
    )
    mock_config.keyword = Mock(max_keywords=10, min_keyword_length=3)
    mock_config.download = Mock(root_dir=str(tmp_path / "downloads"))

    # Create directories
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "downloads").mkdir(exist_ok=True)

    return mock_config


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Fixture for output directory."""
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def cookies_path() -> Optional[str]:
    """
    Fixture for cookies path.

    Returns path to cookies.txt if available, None otherwise.
    Tests requiring cookies should use SKIP_NO_COOKIES marker.

    Returns:
        Path to cookies.txt if found, None otherwise
    """
    locations = [
        Path.cwd() / 'cookies.txt',
        Path.home() / 'cookies.txt',
        Path.home() / '.config' / 'yt-dlp' / 'cookies.txt',
    ]
    for loc in locations:
        if loc.exists():
            return str(loc)
    return None


@pytest.fixture
def video_paths(tmp_path) -> List[str]:
    """
    Fixture for video paths.

    Creates minimal mock video files for testing. For integration tests
    requiring real videos, use a test project directory.

    Returns:
        List of paths to mock video files
    """
    videos = []
    for i in range(3):
        video_file = tmp_path / f"test_video_{i}.mp4"
        video_file.write_bytes(b'\x00' * 1024)  # Mock file content
        videos.append(str(video_file))
    return videos


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    """Fixture for cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture
def video_path(tmp_path) -> str:
    """
    Fixture for single video path.

    Creates a minimal mock video file for testing. For integration tests
    requiring real video content, use a test project directory.

    Returns:
        Path to a mock video file
    """
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b'\x00' * 1024)  # Mock file content
    return str(video_file)


@pytest.fixture
def runner(tmp_path, config):
    """
    Fixture for test runner.

    Creates a mock runner object for testing. For integration tests
    requiring real pipeline execution, use ResilientRunner directly.

    Returns:
        Mock runner object with common attributes
    """
    from unittest.mock import Mock

    mock_runner = Mock()
    mock_runner.project_dir = str(tmp_path)
    mock_runner.config = config
    mock_runner.audio_downloads = []
    mock_runner.merged_segments = []
    mock_runner.checkpoint_path = tmp_path / 'checkpoint.json'
    return mock_runner


@pytest.fixture
def texts():
    """Fixture for text list."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ]


@pytest.fixture
def voiceover_segments():
    """Fixture for voiceover segments."""
    return [
        {"index": 0, "text": "Welcome to the tutorial", "start": 0.0, "end": 2.5},
        {"index": 1, "text": "We will learn about Python", "start": 2.5, "end": 5.0},
        {"index": 2, "text": "Let's get started", "start": 5.0, "end": 7.0},
    ]


@pytest.fixture
def video_segments():
    """Fixture for video segments."""
    return [
        {"file": "video1.mp4", "text": "Python tutorial intro", "start": 0.0, "end": 10.0, "duration_tier": "short"},
        {"file": "video2.mp4", "text": "Learning programming", "start": 0.0, "end": 15.0, "duration_tier": "medium"},
        {"file": "video3.mp4", "text": "Getting started guide", "start": 0.0, "end": 8.0, "duration_tier": "short"},
    ]


@pytest.fixture
def vo_embeddings():
    """Fixture for voiceover embeddings."""
    import numpy as np
    # Return mock 384-dimensional embeddings (common embedding size)
    np.random.seed(42)
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def video_embeddings():
    """Fixture for video embeddings."""
    import numpy as np
    # Return mock 384-dimensional embeddings (common embedding size)
    np.random.seed(123)
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def matches():
    """Fixture for match results."""
    from unittest.mock import Mock
    return [
        Mock(
            vo_index=0,
            video_file="video1.mp4",
            start=0.0,
            end=2.5,
            confidence=0.85,
            strategy="primary"
        ),
        Mock(
            vo_index=1,
            video_file="video2.mp4",
            start=0.0,
            end=5.0,
            confidence=0.72,
            strategy="primary"
        ),
        Mock(
            vo_index=2,
            video_file="video3.mp4",
            start=0.0,
            end=3.0,
            confidence=0.78,
            strategy="primary"
        ),
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    # Simple temp directory that doesn't require external resources
    return tmp_path


# =============================================================================
# FIXTURE FACTORIES (US-006, Sprint 19)
# =============================================================================
#
# Factory fixtures create customizable test objects. Use them like:
#
#     def test_something(srt_segment_factory):
#         segment = srt_segment_factory(text="Custom text", confidence=0.9)
#
# Factories accept keyword arguments to override defaults, making it easy
# to create objects with specific properties for focused test cases.
# =============================================================================


@pytest.fixture
def srt_segment_factory():
    """
    Factory fixture for creating SRTSegment instances.

    Usage:
        segment = srt_segment_factory()  # Defaults
        segment = srt_segment_factory(text="Custom", is_broll=True)
        segment = srt_segment_factory(index=5, start_time=10.0, end_time=15.0)

    Default values:
        - index: 0
        - start_time: 0.0
        - end_time: 5.0
        - text: "Sample voiceover text"
        - source_file: ""
        - keywords: []
        - entities: []
        - topic_id: None
        - topics: []
        - is_broll: False

    Returns:
        Callable that creates SRTSegment instances
    """
    from src.utils import SRTSegment

    def _make(
        index: int = 0,
        start_time: float = 0.0,
        end_time: float = 5.0,
        text: str = "Sample voiceover text",
        source_file: str = "",
        keywords: list = None,
        entities: list = None,
        topic_id: int = None,
        topics: list = None,
        is_broll: bool = False,
    ):
        return SRTSegment(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text,
            source_file=source_file,
            keywords=keywords or [],
            entities=entities or [],
            topic_id=topic_id,
            topics=topics or [],
            is_broll=is_broll,
        )

    return _make


@pytest.fixture
def config_factory(tmp_path):
    """
    Factory fixture for creating mock Config objects with nested section overrides.

    Usage:
        config = config_factory()  # Defaults
        config = config_factory(
            matching={'min_confidence': 0.8},
            download={'root_dir': '/custom/path'}
        )

    Sections:
        - cache: cache_dir path
        - output: output_dir path
        - transcription: model, language
        - matching: min_confidence, location_matching.enabled
        - keyword: max_keywords, min_keyword_length
        - download: root_dir

    Returns:
        Callable that creates mock Config objects
    """
    from unittest.mock import Mock

    def _make(
        cache: dict = None,
        output: dict = None,
        transcription: dict = None,
        matching: dict = None,
        keyword: dict = None,
        download: dict = None,
    ):
        # Default values
        cache_defaults = {'cache_dir': str(tmp_path / "cache")}
        output_defaults = {'output_dir': str(tmp_path / "output")}
        trans_defaults = {'model': 'base', 'language': 'en'}
        match_defaults = {
            'min_confidence': 0.5,
            'location_matching': Mock(enabled=False, geonames_username="")
        }
        kw_defaults = {'max_keywords': 10, 'min_keyword_length': 3}
        dl_defaults = {'root_dir': str(tmp_path / "downloads")}

        # Merge with user overrides
        cache_cfg = {**cache_defaults, **(cache or {})}
        output_cfg = {**output_defaults, **(output or {})}
        trans_cfg = {**trans_defaults, **(transcription or {})}
        match_cfg = {**match_defaults, **(matching or {})}
        kw_cfg = {**kw_defaults, **(keyword or {})}
        dl_cfg = {**dl_defaults, **(download or {})}

        mock_config = Mock()
        mock_config.cache = Mock(**cache_cfg)
        mock_config.output = Mock(**output_cfg)
        mock_config.transcription = Mock(**trans_cfg)
        mock_config.matching = Mock(**match_cfg)
        mock_config.keyword = Mock(**kw_cfg)
        mock_config.download = Mock(**dl_cfg)

        # Create directories
        Path(cache_cfg['cache_dir']).mkdir(exist_ok=True)
        Path(output_cfg['output_dir']).mkdir(exist_ok=True)
        Path(dl_cfg['root_dir']).mkdir(exist_ok=True)

        return mock_config

    return _make


@pytest.fixture
def pipeline_state_factory():
    """
    Factory fixture for creating PipelineState with pre-populated stage data.

    Usage:
        state = pipeline_state_factory()  # Empty state
        state = pipeline_state_factory(
            voiceover_segments=[seg1, seg2],
            keywords=['travel', 'nature'],
            matches=[match1, match2]
        )

    Commonly used fields:
        - voiceover_path: str
        - voiceover_segments: List[VoiceoverSegment]
        - keywords: List[str]
        - downloaded_videos: List[DownloadedVideo]
        - matches: List[Match]
        - text_metadata: List[Dict]

    Returns:
        Callable that creates PipelineState instances
    """
    from src.state import PipelineState

    def _make(**kwargs):
        return PipelineState(**kwargs)

    return _make


@pytest.fixture
def match_result_factory(srt_segment_factory):
    """
    Factory fixture for creating MatchResult with configurable confidence and alternatives.

    Usage:
        result = match_result_factory()  # Simple result with 0.85 confidence
        result = match_result_factory(confidence=0.92, video_source_file="custom.mp4")
        result = match_result_factory(has_gap=True, gap_reason="No suitable match")
        result = match_result_factory(num_alternatives=3)  # Add 3 alternatives

    Parameters:
        - confidence: float (default 0.85) - primary match confidence
        - video_source_file: str (default "video.mp4") - source file for video segment
        - vo_text: str (default "Voiceover text") - voiceover text
        - video_text: str (default "Video transcript") - video transcript
        - has_gap: bool (default False) - whether this is a gap
        - gap_reason: str (default "") - reason for gap
        - num_alternatives: int (default 0) - number of alternatives to generate

    Returns:
        Callable that creates MatchResult instances
    """
    from src.utils import MatchResult, Match, AlternativeMatch

    def _make(
        confidence: float = 0.85,
        video_source_file: str = "video.mp4",
        vo_text: str = "Voiceover text",
        video_text: str = "Video transcript",
        has_gap: bool = False,
        gap_reason: str = "",
        num_alternatives: int = 0,
        matched_keywords: list = None,
    ):
        vo_segment = srt_segment_factory(text=vo_text)
        video_segment = srt_segment_factory(
            text=video_text,
            source_file=video_source_file
        )

        primary = Match(
            voiceover_segment=vo_segment,
            video_segment=video_segment,
            video_scene=None,
            confidence=confidence,
            reasoning="Test match"
        )

        alternatives = []
        for i in range(num_alternatives):
            alt_video_segment = srt_segment_factory(
                text=f"Alternative {i+1}",
                source_file=f"alt_video_{i+1}.mp4"
            )
            alt = AlternativeMatch(
                video_segment=alt_video_segment,
                video_scene=None,
                confidence=confidence - 0.05 * (i + 1),
                reasoning=f"Alternative match {i+1}"
            )
            alternatives.append(alt)

        return MatchResult(
            primary_match=primary,
            alternatives=alternatives,
            has_gap=has_gap,
            gap_reason=gap_reason,
            matched_keywords=matched_keywords or [],
        )

    return _make


@pytest.fixture
def match_factory(srt_segment_factory):
    """
    Factory fixture for creating individual Match objects.

    Usage:
        match = match_factory()  # Defaults
        match = match_factory(confidence=0.95, video_source_file="best.mp4")

    Parameters:
        - vo_text: str (default "Voiceover text")
        - video_text: str (default "Video transcript")
        - video_source_file: str (default "video.mp4")
        - confidence: float (default 0.85)
        - reasoning: str (default "Test match")

    Returns:
        Callable that creates Match instances
    """
    from src.utils import Match

    def _make(
        vo_text: str = "Voiceover text",
        video_text: str = "Video transcript",
        video_source_file: str = "video.mp4",
        confidence: float = 0.85,
        reasoning: str = "Test match",
    ):
        vo_segment = srt_segment_factory(text=vo_text)
        video_segment = srt_segment_factory(
            text=video_text,
            source_file=video_source_file
        )

        return Match(
            voiceover_segment=vo_segment,
            video_segment=video_segment,
            video_scene=None,
            confidence=confidence,
            reasoning=reasoning,
        )

    return _make


# =============================================================================
# DOWNLOAD-SEGMENTS FACTORIES (US-010, Sprint 55)
# =============================================================================
#
# Factory fixtures for download_segments stage testing. Reduces boilerplate
# across download/escalation/circuit-breaker test files.
# =============================================================================


@pytest.fixture
def download_config_factory():
    """
    Factory fixture for creating mock download config objects.

    Usage:
        cfg = download_config_factory()  # Defaults
        cfg = download_config_factory(socket_timeout=60, cookies_from_browser="firefox")
        cfg = download_config_factory(escalation={'enabled': True, 'max_tier': 3})

    Parameters:
        - cookies_from_browser: str (default "chrome")
        - cookiefile: str or None (default None)
        - socket_timeout: int (default 30)
        - max_retries: int (default 3)
        - escalation: dict (default {'enabled': True}) - merged onto mock

    Returns:
        Callable that creates mock download config objects
    """
    from unittest.mock import Mock

    def _make(
        cookies_from_browser: str = "chrome",
        cookiefile: str = None,
        socket_timeout: int = 30,
        max_retries: int = 3,
        escalation: dict = None,
        **extra_fields,
    ):
        esc_defaults = {'enabled': True}
        esc_cfg = {**esc_defaults, **(escalation or {})}

        mock_cfg = Mock()
        mock_cfg.cookies_from_browser = cookies_from_browser
        mock_cfg.cookiefile = cookiefile
        mock_cfg.socket_timeout = socket_timeout
        mock_cfg.max_retries = max_retries
        mock_cfg.escalation = Mock(**esc_cfg)

        # Apply any extra fields
        for key, value in extra_fields.items():
            setattr(mock_cfg, key, value)

        return mock_cfg

    return _make


@pytest.fixture
def escalation_result_factory():
    """
    Factory fixture for creating EscalationResult objects.

    Usage:
        result = escalation_result_factory()  # Tier 1 defaults
        result = escalation_result_factory(tier=2, rotate_cookies=True)
        result = escalation_result_factory(
            args=['--impersonate', 'Chrome-131:Windows-11', '--cookies-from-browser', 'chrome']
        )

    Parameters:
        - args: List[str] (default ['--impersonate', 'Chrome-131:Windows-11'])
        - tier: int (default 1) - maps to EscalationTier enum value
        - rotate_cookies: bool (default False)
        - rotate_vpn: bool (default False)

    Returns:
        Callable that creates EscalationResult instances
    """
    from src.downloader.escalation_manager import EscalationResult
    from src.downloader.types import EscalationTier

    def _make(
        args: list = None,
        tier: int = 1,
        rotate_cookies: bool = False,
        rotate_vpn: bool = False,
    ):
        return EscalationResult(
            args=args if args is not None else ['--impersonate', 'Chrome-131:Windows-11'],
            tier=EscalationTier(tier),
            rotate_cookies=rotate_cookies,
            rotate_vpn=rotate_vpn,
        )

    return _make


@pytest.fixture
def segment_download_stats_factory():
    """
    Factory fixture for creating SegmentDownloadStats instances.

    Usage:
        stats = segment_download_stats_factory()  # All zeros
        stats = segment_download_stats_factory(succeeded=5, failed=2, total=10)
        stats = segment_download_stats_factory(
            error_categories={'network': 3, 'bot_detection': 1}
        )

    Parameters:
        - succeeded: int (default 0)
        - failed: int (default 0)
        - cached: int (default 0)
        - attempted: int (default 0)
        - total: int (default 0)
        - retry_count: int (default 0)
        - error_categories: dict (default {})
        - segment_durations: list (default [])
        - total_bytes: int (default 0)

    Returns:
        Callable that creates SegmentDownloadStats instances
    """
    from src.stages.download_segments import SegmentDownloadStats

    def _make(
        succeeded: int = 0,
        failed: int = 0,
        cached: int = 0,
        attempted: int = 0,
        total: int = 0,
        retry_count: int = 0,
        error_categories: dict = None,
        segment_durations: list = None,
        total_bytes: int = 0,
    ):
        stats = SegmentDownloadStats()
        stats.succeeded = succeeded
        stats.failed = failed
        stats.cached = cached
        stats.attempted = attempted
        stats.total = total
        stats.retry_count = retry_count
        if error_categories:
            stats.error_categories = dict(error_categories)
        if segment_durations:
            stats.segment_durations = list(segment_durations)
        stats.total_bytes = total_bytes
        return stats

    return _make


# =============================================================================
# PIPELINE STATE FIXTURES (US-86-003, Sprint 86)
# =============================================================================
#
# Fixtures providing populated dataclass instances for common pipeline state objects.
# These fixtures create fully initialized objects with realistic test data.
# =============================================================================


@pytest.fixture
def pipeline_state() -> 'PipelineState':
    """
    Fixture providing a fully populated PipelineState with all required fields.

    Returns:
        PipelineState with populated voiceover_segments, video_search_results,
        matches, and other common fields for testing.
    """
    from src.state import PipelineState, VoiceoverSegment, VideoSearchResult, Match

    voiceover_segments = [
        VoiceoverSegment(index=0, start=0.0, end=5.0, text="Welcome to the tutorial", duration=5.0),
        VoiceoverSegment(index=1, start=5.0, end=10.0, text="We will learn about Python", duration=5.0),
        VoiceoverSegment(index=2, start=10.0, end=15.0, text="Let's get started with the basics", duration=5.0),
    ]

    video_search_results = [
        VideoSearchResult(
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            title="Python Tutorial",
            channel="Tech Channel",
            duration=600.0,
            duration_tier="medium",
            keyword="python",
        ),
        VideoSearchResult(
            video_id="def456",
            url="https://youtube.com/watch?v=def456",
            title="Learn Programming",
            channel="Code Channel",
            duration=300.0,
            duration_tier="short",
            keyword="programming",
        ),
    ]

    matches = [
        Match(
            segment_index=0,
            video_file="abc123",
            video_start=0.0,
            video_end=5.0,
            confidence=0.85,
            strategy="primary",
            reason="Keyword match",
        ),
        Match(
            segment_index=1,
            video_file="def456",
            video_start=10.0,
            video_end=15.0,
            confidence=0.72,
            strategy="primary",
            reason="Semantic match",
        ),
    ]

    return PipelineState(
        voiceover_path="/path/to/voiceover.srt",
        voiceover_segments=voiceover_segments,
        keywords=["python", "tutorial", "programming"],
        topic_context="Technology tutorial",
        video_ids=["abc123", "def456"],
        video_search_results=video_search_results,
        matches=matches,
    )


@pytest.fixture
def downloaded_video() -> 'DownloadedVideo':
    """
    Fixture providing a fully populated DownloadedVideo dataclass.

    Returns:
        DownloadedVideo with typical test values.
    """
    from src.state import DownloadedVideo

    return DownloadedVideo(
        file="abc123.mp4",
        url="https://youtube.com/watch?v=abc123",
        title="Python Tutorial",
        channel="Tech Channel",
        upload_date="2024-01-15",
        duration=600.0,
        duration_tier="medium",
        keyword="python",
        download_date="2024-01-20",
        license="Creative Commons",
        source="download",
        video_hash="abc123hash",
        face_score=0.75,
        description="Learn Python programming basics",
    )


@pytest.fixture
def video_search_result() -> 'VideoSearchResult':
    """
    Fixture providing a fully populated VideoSearchResult dataclass.

    Returns:
        VideoSearchResult with typical test values.
    """
    from src.state import VideoSearchResult

    return VideoSearchResult(
        video_id="abc123",
        url="https://youtube.com/watch?v=abc123",
        title="Python Tutorial",
        channel="Tech Channel",
        duration=600.0,
        duration_tier="medium",
        keyword="python",
        description="Learn Python programming basics",
    )


@pytest.fixture
def voiceover_segment_list() -> List['VoiceoverSegment']:
    """
    Fixture providing a list of VoiceoverSegment dataclasses.

    Returns:
        List of VoiceoverSegment with typical test values.
    """
    from src.state import VoiceoverSegment

    return [
        VoiceoverSegment(index=0, start=0.0, end=5.0, text="Welcome to the tutorial", duration=5.0),
        VoiceoverSegment(index=1, start=5.0, end=10.0, text="We will learn about Python", duration=5.0),
        VoiceoverSegment(index=2, start=10.0, end=15.0, text="Let's get started with the basics", duration=5.0),
        VoiceoverSegment(index=3, start=15.0, end=20.0, text="Now let's write some code", duration=5.0),
    ]


@pytest.fixture
def caption_result() -> 'CaptionResult':
    """
    Fixture providing a fully populated CaptionResult with segments.

    Returns:
        CaptionResult with typical test values and caption segments.
    """
    from src.caption.models import CaptionResult, CaptionSegment, CaptionStatus

    segments = [
        CaptionSegment(index=0, start_time=0.0, end_time=5.0, text="Welcome to this tutorial"),
        CaptionSegment(index=1, start_time=5.0, end_time=10.0, text="We will learn about Python"),
        CaptionSegment(index=2, start_time=10.0, end_time=15.0, text="Let's get started"),
    ]

    return CaptionResult(
        video_id="abc123",
        segments=segments,
        language="en",
        is_auto_generated=False,
        format_source="vtt",
        video_duration=600.0,
        status=CaptionStatus.SUCCESS,
    )


@pytest.fixture
def match_result() -> 'Match':
    """
    Fixture providing a fully populated Match dataclass with confidence scores.

    Returns:
        Match with typical test values and confidence score.
    """
    from src.state import Match

    return Match(
        segment_index=0,
        video_file="abc123.mp4",
        video_start=0.0,
        video_end=5.0,
        confidence=0.85,
        strategy="primary",
        reason="Keyword and semantic match for Python tutorial content",
        face_score=0.75,
    )


# =============================================================================
# MOCK LLM CLIENT FIXTURE (US-86-005, Sprint 86)
# =============================================================================
#
# Mock LLM client for deterministic testing without API calls.
# Provides configurable responses for generate() and deterministic embeddings for embed().
# =============================================================================


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.

    Provides:
    - mock_generate(): Returns configurable responses based on prompt patterns
    - mock_embed(): Returns deterministic embedding vectors

    Usage:
        mock = MockLLMClient()
        mock.set_response("keyword", {"keywords": ["test", "example"]})
        mock.set_embedding("test", [0.1, 0.2, 0.3])

        # Use in tests
        from unittest.mock import patch
        with patch('src.llm_client.create_client', return_value=mock):
            # Test code that uses LLM client
    """

    def __init__(self):
        self._responses: Dict[str, Any] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._call_log: List[Dict[str, Any]] = []
        self.provider_name = "mock"
        self.model = "mock-model"

    def set_response(self, pattern: str, response: Any):
        """
        Configure a response for a prompt pattern.

        Args:
            pattern: Substring to match in prompt (case-insensitive)
            response: Response to return (will be wrapped in LLMResponse)
        """
        self._responses[pattern.lower()] = response

    def set_text_response(self, pattern: str, text: str):
        """
        Configure a text response for a prompt pattern.

        Args:
            pattern: Substring to match in prompt
            text: Raw text response
        """
        self._responses[pattern.lower()] = {"_text": text}

    def set_embedding(self, text: str, embedding: List[float]):
        """
        Configure a deterministic embedding for text.

        Args:
            text: Text to match (case-insensitive)
            embedding: List of floats representing the embedding vector
        """
        self._embeddings[text.lower()] = embedding

    def generate(self, request) -> 'LLMResponse':
        """
        Generate a mock response based on prompt pattern matching.

        Args:
            request: LLMRequest object

        Returns:
            LLMResponse with configured response or default
        """
        from src.llm_client import LLMResponse

        prompt_lower = request.prompt.lower()

        # Log the call
        self._call_log.append({
            'method': 'generate',
            'prompt': request.prompt,
            'prompt_length': len(request.prompt),
            'response_format': request.response_format.value,
        })

        # Find matching pattern
        for pattern, response in self._responses.items():
            if pattern in prompt_lower:
                if isinstance(response, dict) and "_text" in response:
                    # Text response
                    return LLMResponse(
                        text=response["_text"],
                        parsed_data=None,
                        provider=self.provider_name,
                        model=self.model,
                        cached=False,
                        request_time_ms=1.0,
                    )
                else:
                    # Dict response (parsed as JSON)
                    import json
                    text = json.dumps(response)
                    return LLMResponse(
                        text=text,
                        parsed_data=response,
                        provider=self.provider_name,
                        model=self.model,
                        cached=False,
                        request_time_ms=1.0,
                    )

        # Default response
        return LLMResponse(
            text="Mock response",
            parsed_data={"result": "default"},
            provider=self.provider_name,
            model=self.model,
            cached=False,
            request_time_ms=1.0,
        )

    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        """
        Return deterministic embeddings for texts.

        Args:
            texts: List of texts to embed
            embed_mode: Embed mode (document or query) - ignored in mock

        Returns:
            List of embedding vectors
        """
        # Log the call
        self._call_log.append({
            'method': 'embed',
            'texts': texts,
            'embed_mode': embed_mode,
        })

        results = []
        for text in texts:
            text_lower = text.lower()
            if text_lower in self._embeddings:
                results.append(self._embeddings[text_lower])
            else:
                # Generate deterministic embedding based on text hash
                embedding_dim = 384  # Common dimension
                results.append(self._generate_deterministic_embedding(text_lower, embedding_dim))

        return results

    def _generate_deterministic_embedding(self, text: str, dim: int) -> List[float]:
        """Generate a deterministic embedding from text."""
        import hashlib
        # Use hash to generate seed for reproducible vectors
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        import random
        random.seed(seed)
        return [random.random() for _ in range(dim)]

    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get log of all calls made to this mock."""
        return list(self._call_log)

    def clear_call_log(self):
        """Clear the call log."""
        self._call_log.clear()

    def clear_responses(self):
        """Clear all configured responses."""
        self._responses.clear()

    def clear_embeddings(self):
        """Clear all configured embeddings."""
        self._embeddings.clear()


@pytest.fixture
def mock_llm_client():
    """
    Fixture providing a MockLLMClient for testing LLM-dependent code.

    The mock is pre-configured with common responses and embeddings, but
    can be further customized using:
    - mock.set_response(pattern, response) - configure generate() response
    - mock.set_embedding(text, embedding) - configure embed() response

    Usage:
        def test_keyword_extraction(mock_llm_client):
            # Configure the mock
            mock_llm_client.set_response("keywords", {"keywords": ["python", "tutorial"]})

            # Use in code that calls LLM
            result = extract_keywords("Sample text about Python")

            # Verify
            assert result == ["python", "tutorial"]
            assert mock_llm_client.get_call_log()

    Returns:
        MockLLMClient instance
    """
    mock = MockLLMClient()

    # Pre-configure common responses
    mock.set_response("keyword", {"keywords": ["test", "example"]})
    mock.set_response("extract", {"result": "extracted"})
    mock.set_response("match", {"confidence": 0.85, "reasoning": "test match"})
    mock.set_text_response("hello", "Hello! I'm a mock LLM.")

    # Pre-configure common embeddings
    mock.set_embedding("python tutorial", [0.1] * 384)
    mock.set_embedding("test text", [0.2] * 384)
    mock.set_embedding("example", [0.3] * 384)

    yield mock

    # Cleanup after test
    mock.clear_call_log()


@pytest.fixture
def mock_llm_client_with_response():
    """
    Fixture providing a MockLLMClient pre-configured with a specific response.

    Use this when you need a predictable response for your test.

    Usage:
        def test_something(mock_llm_client_with_response):
            mock = mock_llm_client_with_response
            mock.set_response("extract", {"entities": ["Python", "AI"]})

            # Test code that uses LLM
            ...

    Returns:
        MockLLMClient instance (same as mock_llm_client)
    """
    return MockLLMClient()


# =============================================================================
# MOCK DOWNLOADER FIXTURE (US-86-006, Sprint 86)
# =============================================================================
#
# Mock yt-dlp-based downloader for deterministic testing without network.
# Provides mock video info and simulated downloads.
# =============================================================================


class MockYTDL:
    """
    Mock yt-dlp YoutubeDL object for testing without network.

    Provides:
    - extract_info(): Returns mock video metadata for search/download
    - simulate_download(): Creates mock video files

    Usage:
        mock_ydl = MockYTDL()
        mock_ydl.set_video_info("abc123", {"title": "Test Video", "duration": 120})
        mock_ydl.set_download_result("abc123", success=True)

        # Use in tests
        from unittest.mock import patch
        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
            # Test code that uses yt-dlp
    """

    def __init__(self, options: dict = None):
        """Initialize mock yt-dlp with optional options."""
        self.options = options or {}
        self._video_info: Dict[str, Any] = {}
        self._download_results: Dict[str, bool] = {}
        self._call_log: List[Dict[str, Any]] = []

    def set_video_info(self, video_id: str, info: Dict[str, Any]):
        """
        Configure mock video metadata for a video ID.

        Args:
            video_id: YouTube video ID
            info: Video metadata dict (title, duration, channel, etc.)
        """
        # Ensure required fields
        info.setdefault('id', video_id)
        info.setdefault('title', f'Video {video_id}')
        info.setdefault('duration', 120)
        info.setdefault('channel', 'Test Channel')
        info.setdefault('uploader', 'Test Channel')
        info.setdefault('upload_date', '20240101')
        info.setdefault('thumbnail', f'https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg')
        info.setdefault('description', f'Test description for {video_id}')
        self._video_info[video_id] = info

    def set_search_results(self, query: str, results: List[Dict[str, Any]]):
        """
        Configure mock search results for a query.

        Args:
            query: Search query string
            results: List of video info dicts
        """
        self._video_info[f'search:{query}'] = results

    def set_download_result(self, video_id: str, success: bool = True):
        """
        Configure the result for a download operation.

        Args:
            video_id: YouTube video ID
            success: Whether download should succeed (default: True)
        """
        self._download_results[video_id] = success

    def extract_info(self, url: str, download: bool = False) -> Optional[Dict[str, Any]]:
        """
        Mock extract_info - returns video metadata.

        Args:
            url: Video URL or search URL
            download: Whether to download (ignored in mock)

        Returns:
            Video info dict or list of entries for search
        """
        self._call_log.append({
            'method': 'extract_info',
            'url': url,
            'download': download,
        })

        # Handle search URLs (ytsearchN:query)
        if url.startswith('ytsearch'):
            # Parse: ytsearch10:python tutorial
            match = re.match(r'ytsearch(\d+):(.+)', url)
            if match:
                count = int(match.group(1))
                query = match.group(2)
                results = self._video_info.get(f'search:{query}', [])
                return {'entries': results[:count]}

        # Handle video URLs (https://youtube.com/watch?v=XXX or just video ID)
        video_id = self._extract_video_id(url)
        if video_id and video_id in self._video_info:
            return self._video_info[video_id]

        # Return None if not found
        return None

    def download(self, video_id_or_url: str, output_path: str = None) -> bool:
        """
        Mock download - simulates download without actual network call.

        Args:
            video_id_or_url: Video ID or URL
            output_path: Optional output path (mock creates file)

        Returns:
            True if successful, False otherwise
        """
        self._call_log.append({
            'method': 'download',
            'video_id_or_url': video_id_or_url,
            'output_path': output_path,
        })

        video_id = self._extract_video_id(video_id_or_url)

        # Check if download should succeed
        if video_id in self._download_results:
            success = self._download_results[video_id]
        else:
            # Default: succeed if we have video info
            success = video_id in self._video_info

        # If success and output_path provided, create mock file
        if success and output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                # Write minimal valid MP4 header
                f.write(b'\x00\x00\x00\x1cftypisom\x00\x00\x0200isomiso2mp41')

        return success

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from URL or return as-is if already ID."""
        import re
        # YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',  # Already just an ID
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get log of all calls made to this mock."""
        return list(self._call_log)

    def clear_call_log(self):
        """Clear the call log."""
        self._call_log.clear()


class MockDownloader:
    """
    Mock VideoDownloader for testing without network.

    Provides:
    - download_video(): Simulates downloading a single video
    - get_video_info(): Returns mock metadata
    - search_videos(): Returns mock search results

    Usage:
        mock = MockDownloader()
        mock.set_video_info("abc123", {"title": "Test", "duration": 120})

        with patch('src.downloader.core.VideoDownloader', return_value=mock):
            # Test code that uses VideoDownloader
    """

    def __init__(self, config=None):
        """Initialize mock downloader with optional config."""
        self.config = config
        self._video_info: Dict[str, Any] = {}
        self._search_results: Dict[str, List[Dict[str, Any]]] = {}
        self._downloads: List[Dict[str, Any]] = []
        self._call_log: List[Dict[str, Any]] = []

    def set_video_info(self, video_id: str, info: Dict[str, Any]):
        """Configure mock video metadata."""
        info.setdefault('video_id', video_id)
        info.setdefault('title', f'Video {video_id}')
        info.setdefault('duration', 120)
        info.setdefault('channel', 'Test Channel')
        info.setdefault('url', f'https://youtube.com/watch?v={video_id}')
        self._video_info[video_id] = info

    def set_search_results(self, keyword: str, results: List[Dict[str, Any]]):
        """Configure mock search results for a keyword."""
        self._search_results[keyword] = results

    def download_video(self, video_id: str, output_path: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Simulate downloading a video.

        Args:
            video_id: YouTube video ID
            output_path: Output file path
            **kwargs: Additional download options

        Returns:
            DownloadedVideo dict or None on failure
        """
        self._call_log.append({
            'method': 'download_video',
            'video_id': video_id,
            'output_path': output_path,
        })

        if video_id not in self._video_info:
            return None

        info = self._video_info[video_id].copy()
        info['file'] = output_path or f'mock/{video_id}.mp4'

        self._downloads.append(info)
        return info

    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get mock video metadata.

        Args:
            video_id: YouTube video ID

        Returns:
            Video info dict or None if not found
        """
        self._call_log.append({
            'method': 'get_video_info',
            'video_id': video_id,
        })
        return self._video_info.get(video_id)

    def search_videos(self, keyword: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for mock videos by keyword.

        Args:
            keyword: Search keyword
            max_results: Maximum results to return

        Returns:
            List of video info dicts
        """
        self._call_log.append({
            'method': 'search_videos',
            'keyword': keyword,
            'max_results': max_results,
        })

        results = self._search_results.get(keyword, [])
        return results[:max_results]

    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get log of all calls made to this mock."""
        return list(self._call_log)

    def get_downloads(self) -> List[Dict[str, Any]]:
        """Get list of completed downloads."""
        return list(self._downloads)

    def clear_call_log(self):
        """Clear the call log."""
        self._call_log.clear()


import re


@pytest.fixture
def mock_youtube_dl():
    """
    Fixture providing a MockYTDL for testing yt-dlp-dependent code.

    The mock is pre-configured with common video metadata but can be
    customized using:
    - mock.set_video_info(id, info) - configure video metadata
    - mock.set_search_results(query, results) - configure search results
    - mock.set_download_result(id, success) - configure download result

    Usage:
        def test_video_search(mock_youtube_dl):
            # Configure mock
            mock_youtube_dl.set_search_results("python", [
                {"id": "abc123", "title": "Python Tutorial", "duration": 300}
            ])

            # Use in code that calls yt-dlp
            from unittest.mock import patch
            with patch('yt_dlp.YoutubeDL', return_value=mock_youtube_dl):
                results = search_videos("python")

            assert len(results) == 1
            assert mock_youtube_dl.get_call_log()

    Returns:
        MockYTDL instance
    """
    mock = MockYTDL()

    # Pre-configure common videos
    mock.set_video_info("dQw4w9WgXcQ", {
        "title": "Never Gonna Give You Up",
        "duration": 213,
        "channel": "RickAstleyVEVO",
    })
    mock.set_video_info("abc123def456", {
        "title": "Python Tutorial for Beginners",
        "duration": 600,
        "channel": "Tech Tutorials",
    })
    mock.set_video_info("xyz789uvw012", {
        "title": "Travel B-roll Collection",
        "duration": 180,
        "channel": "Stock Footage",
    })

    # Pre-configure common search results
    mock.set_search_results("python tutorial", [
        {"id": "abc123def456", "title": "Python Tutorial for Beginners", "duration": 600},
        {"id": "def789ghi012", "title": "Advanced Python Tips", "duration": 300},
    ])

    yield mock

    # Cleanup after test
    mock.clear_call_log()


@pytest.fixture
def mock_downloader():
    """
    Fixture providing a MockDownloader for testing VideoDownloader-dependent code.

    The mock provides a simpler interface than MockYTDL for testing
    the VideoDownloader class directly.

    Usage:
        def test_download(mock_downloader):
            # Configure mock
            mock_downloader.set_video_info("abc123", {
                "title": "Test Video",
                "duration": 120,
            })

            # Use in code that calls VideoDownloader
            from unittest.mock import patch
            with patch('src.downloader.core.VideoDownloader', return_value=mock_downloader):
                result = download_video("abc123", "output.mp4")

            assert mock_downloader.get_downloads()

    Returns:
        MockDownloader instance
    """
    mock = MockDownloader()

    # Pre-configure common videos
    mock.set_video_info("dQw4w9WgXcQ", {
        "title": "Never Gonna Give You Up",
        "duration": 213,
        "channel": "RickAstleyVEVO",
        "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    })
    mock.set_video_info("abc123def456", {
        "title": "Python Tutorial for Beginners",
        "duration": 600,
        "channel": "Tech Tutorials",
        "url": "https://youtube.com/watch?v=abc123def456",
    })

    # Pre-configure search results
    mock.set_search_results("python", [
        {"video_id": "abc123def456", "title": "Python Tutorial", "duration": 600},
    ])
    mock.set_search_results("travel", [
        {"video_id": "xyz789uvw012", "title": "Travel B-roll", "duration": 180},
    ])

    yield mock

    # Cleanup
    mock.clear_call_log()


@pytest.fixture
def tmp_video_dir(tmp_path):
    """
    Fixture providing a temporary directory for video downloads.

    Creates a clean temporary directory structure suitable for video
    downloads during tests. Automatically cleaned up after test.

    Directory structure created:
        tmp_video_dir/
        ├── videos/
        └── cache/

    Usage:
        def test_download(tmp_video_dir):
            video_path = tmp_video_dir / "videos" / "test.mp4"
            download_video("abc123", str(video_path))
            assert video_path.exists()

    Returns:
        Path: Path to temporary video directory
    """
    video_dir = tmp_path / "videos"
    cache_dir = tmp_path / "cache"

    video_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return tmp_path


# Helper function to create mock video file
def create_mock_video_file(path: str, duration_seconds: int = 1):
    """
    Create a minimal mock video file for testing.

    Args:
        path: File path to create
        duration_seconds: Approximate duration (affects file size)
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write minimal MP4 header + some padding
    with open(path, 'wb') as f:
        # ftyp box
        f.write(b'\x00\x00\x00\x1cftypisom\x00\x00\x0200isomiso2mp41')
        # mdat box with some data
        size = 1024 * duration_seconds  # ~1KB per second
        f.write(size.to_bytes(4, 'big') + b'mdat')
        f.write(b'\x00' * (size - 8))


# =============================================================================
# TEST ISOLATION MONITORING (US-005, Sprint 26)
# =============================================================================
#
# These fixtures help detect and warn about test isolation issues.
# Tests should use tmp_path for file operations to avoid:
# - Tests affecting each other
# - Artifacts left after test runs
# - Flaky tests due to pre-existing files
# =============================================================================


@pytest.fixture(scope="session")
def _isolation_tracker(request):
    """
    Session-scoped tracker for file creations outside tmp_path.

    This is an internal fixture used by the isolation monitoring system.
    """
    from collections import defaultdict
    tracker = {
        'violations': defaultdict(list),
        'warned': set(),
    }
    # Store on config for terminal summary access
    request.config._isolation_tracker = tracker
    return tracker


@pytest.fixture(autouse=True)
def _check_test_isolation(request, tmp_path, _isolation_tracker):
    """
    Auto-use fixture that warns when tests create files outside tmp_path.

    This fixture monitors for common isolation anti-patterns:
    - Direct use of tempfile.mkdtemp() without cleanup
    - Files created in project directories
    - Hardcoded paths to external directories

    Enabled via: pytest --check-isolation (default: off)
    """
    # Only run if --check-isolation flag is set
    if not request.config.getoption("--check-isolation", default=False):
        yield
        return

    import tempfile
    from unittest.mock import patch

    test_name = request.node.name
    created_paths = []
    pytest_tmp_root = tmp_path.parent

    # Track mkdtemp calls
    original_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_paths.append(('mkdtemp', path))
        return path

    with patch.object(tempfile, 'mkdtemp', tracking_mkdtemp):
        yield

    # Report violations (warnings, not failures)
    if created_paths:
        for op_type, path in created_paths:
            if path not in _isolation_tracker['warned']:
                _isolation_tracker['violations'][test_name].append((op_type, path))
                _isolation_tracker['warned'].add(path)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--check-isolation",
        action="store_true",
        default=False,
        help="Enable test isolation checking (warns about file creation outside tmp_path)"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report isolation violations at end of test run."""
    if not config.getoption("--check-isolation", default=False):
        return

    tracker = getattr(config, '_isolation_tracker', None)
    if not tracker or not tracker['violations']:
        return

    terminalreporter.write_sep("=", "Test Isolation Warnings")
    terminalreporter.write_line(
        "\nThe following tests used tempfile.mkdtemp() - consider using tmp_path:"
    )

    for test_name, violations in tracker['violations'].items():
        terminalreporter.write_line(f"\n  {test_name}:")
        for op_type, path in violations:
            terminalreporter.write_line(f"    [{op_type}] {path}")

    terminalreporter.write_line(
        "\nNote: mkdtemp() works but tmp_path is preferred for automatic cleanup. "
        "See tests/README.md for best practices."
    )


# =============================================================================
# YOUTUBE API INTEGRATION TEST FIXTURES (US-149-010)
# =============================================================================
#
# Fixtures for YouTube API integration tests with recorded responses.
# These fixtures enable testing without live API calls by using recorded data.
# =============================================================================

from unittest.mock import MagicMock


# Determine recordings directory for YouTube API fixtures
RECORDINGS_DIR = FIXTURES_DIR / "recordings"


@pytest.fixture(scope="session")
def youtube_api_recorded_responses() -> Dict[str, Any]:
    """
    Load all recorded YouTube API responses from the recordings directory.

    Returns:
        Dict mapping response categories to response data
    """
    responses = {}

    # Create recordings directory if it doesn't exist
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load each recording file
    recording_files = {
        "search": "search_responses.json",
        "video_details": "video_details_responses.json",
        "channel_metadata": "channel_metadata_responses.json",
        "captions": "caption_responses.json",
        "quota_exhausted": "quota_exhausted_responses.json",
        "rate_limited": "rate_limited_responses.json",
    }

    for key, filename in recording_files.items():
        filepath = RECORDINGS_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                responses[key] = json.load(f)
        else:
            responses[key] = {}

    return responses


@pytest.fixture
def load_recorded_response(youtube_api_recorded_responses):
    """
    Factory fixture to load a specific recorded response.

    Usage:
        def test_something(load_recorded_response):
            search_response = load_recorded_response("search", "nature_doc")
            assert search_response["items"]
    """
    def _load(category: str, key: str = "default") -> Optional[Dict[str, Any]]:
        """Load a specific response from the recorded responses."""
        if category not in youtube_api_recorded_responses:
            return None
        return youtube_api_recorded_responses[category].get(key)

    return _load


@pytest.fixture
def mock_youtube_api_server(youtube_api_recorded_responses):
    """
    Create a mock HTTP server that returns recorded responses.

    Returns a mock session that can be configured to return specific responses.
    """
    class MockYouTubeAPIServer:
        def __init__(self):
            self.responses = youtube_api_recorded_responses
            self.call_count = 0
            self.last_request = None

        def get_response(self, endpoint: str, params: Dict[str, Any]) -> MagicMock:
            """Get a mock response for the given endpoint and parameters."""
            self.call_count += 1
            self.last_request = {"endpoint": endpoint, "params": params}

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            # Determine which recorded response to return
            if "search" in endpoint:
                query = params.get("q", params.get("searchTerms", ""))
                response_data = self.responses.get("search", {}).get(query, {})

                # Default to "default" key if specific query not found
                if not response_data:
                    response_data = self.responses.get("search", {}).get("default", {})

                mock_response.json.return_value = response_data

            elif "videos" in endpoint:
                video_id = params.get("id", "")
                response_data = self.responses.get("video_details", {}).get(video_id, {})

                if not response_data:
                    response_data = self.responses.get("video_details", {}).get("default", {})

                mock_response.json.return_value = response_data

            elif "channels" in endpoint:
                channel_id = params.get("id", "")
                response_data = self.responses.get("channel_metadata", {}).get(channel_id, {})

                if not response_data:
                    response_data = self.responses.get("channel_metadata", {}).get("default", {})

                mock_response.json.return_value = response_data

            elif "captions" in endpoint:
                video_id = params.get("videoId", "")
                response_data = self.responses.get("captions", {}).get(video_id, {})

                if not response_data:
                    response_data = self.responses.get("captions", {}).get("default", {})

                mock_response.json.return_value = response_data

            else:
                mock_response.json.return_value = {"items": []}

            return mock_response

        def get_error_response(self, error_type: str = "quota_exhausted") -> MagicMock:
            """Get a mock error response."""
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            error_responses = self.responses.get(error_type, {})

            if error_type == "quota_exhausted":
                mock_response.status_code = 403
                mock_response.json.return_value = error_responses.get("default", {
                    "error": {
                        "code": 403,
                        "message": "Quota exceeded",
                        "errors": [{"reason": "quotaExceeded"}]
                    }
                })
            elif error_type == "rate_limited":
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "60"}
                mock_response.json.return_value = error_responses.get("default", {
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "errors": [{"reason": "rateLimitExceeded"}]
                    }
                })
            else:
                mock_response.status_code = 500
                mock_response.json.return_value = {"error": {"code": 500, "message": "Server error"}}

            return mock_response

        def reset(self):
            """Reset the mock server state."""
            self.call_count = 0
            self.last_request = None

    return MockYouTubeAPIServer()


@pytest.fixture
def api_client_with_recordings(mock_youtube_api_server):
    """
    Create a YouTubeAPIClient configured to use recorded responses.

    This fixture patches the requests.Session to return recorded responses
    instead of making live API calls.
    """
    from src.downloader.youtube_api_client import YouTubeAPIClient

    def _create_client(api_keys=None, **kwargs):
        with patch("requests.Session") as mock_session:
            # Configure mock session
            instance = MagicMock()
            mock_session.return_value = instance

            # Set up side effect to use mock_server
            def get_side_effect(url, **request_kwargs):
                # Parse URL to determine endpoint
                if "search" in url:
                    params = request_kwargs.get("params", {})
                    return mock_youtube_api_server.get_response("search", params)
                elif "videos" in url:
                    params = request_kwargs.get("params", {})
                    return mock_youtube_api_server.get_response("videos", params)
                elif "channels" in url:
                    params = request_kwargs.get("params", {})
                    return mock_youtube_api_server.get_response("channels", params)
                elif "captions" in url:
                    params = request_kwargs.get("params", {})
                    return mock_youtube_api_server.get_response("captions", params)
                else:
                    return mock_youtube_api_server.get_response("unknown", {})

            instance.get.side_effect = get_side_effect

            # Create client with the mocked session
            client = YouTubeAPIClient(
                api_key=api_keys[0] if api_keys and isinstance(api_keys, list) else (api_keys or "test_key"),
                auto_scale_quota=False,
                **kwargs
            )

            return client, mock_youtube_api_server

    return _create_client


@pytest.fixture
def fallback_handler_with_recordings(mock_youtube_api_server, youtube_api_recorded_responses):
    """
    Create a fallback handler configured with recorded quota_exhausted responses.

    This fixture tests the fallback handler with recorded quota exceeded errors.
    """
    from src.downloader.api_fallback_handler import YouTubeAPIFallbackHandler

    def _create_handler():
        handler = YouTubeAPIFallbackHandler()

        # Inject recorded quota_exhausted responses
        handler._recorded_responses = youtube_api_recorded_responses.get("quota_exhausted", {})

        return handler, mock_youtube_api_server

    return _create_handler


@pytest.fixture
def key_rotation_with_recordings(mock_youtube_api_server, youtube_api_recorded_responses):
    """
    Create a key rotation scenario with recorded 403 responses.

    This fixture tests key rotation logic with recorded quota exceeded/403 errors.
    """
    from src.downloader.youtube_api_client import YouTubeAPIClient

    def _create_scenario(num_keys=2):
        """Create a key rotation scenario with multiple API keys."""
        # Track which keys are exhausted
        exhausted_keys = []

        with patch("requests.Session") as mock_session:
            instance = MagicMock()
            mock_session.return_value = instance

            def get_side_effect(url, **request_kwargs):
                # Check if all keys are exhausted
                if len(exhausted_keys) >= num_keys:
                    return mock_youtube_api_server.get_error_response("quota_exhausted")

                # Return quota exceeded for each key
                return mock_youtube_api_server.get_error_response("quota_exhausted")

            instance.get.side_effect = get_side_effect

            client = YouTubeAPIClient(
                api_keys=[f"test_key_{i}" for i in range(num_keys)],
                auto_scale_quota=False,
            )

            return client, exhausted_keys, mock_youtube_api_server

    return _create_scenario


@pytest.fixture
def sample_search_queries():
    """Provide sample search queries for testing."""
    return [
        "nature documentary",
        "wildlife africa",
        "ocean exploration",
        "mountain climbing",
        "space exploration",
    ]


@pytest.fixture
def sample_video_ids():
    """Provide sample video IDs for testing."""
    return [
        "vid123",
        "vid456",
        "vid789",
        "vid_abc",
        "vid_def",
    ]


@pytest.fixture
def sample_channel_ids():
    """Provide sample channel IDs for testing."""
    return [
        "UC123456",
        "UC789012",
        "UC345678",
    ]


@pytest.fixture
def mock_api_responses():
    """Return mock API responses for testing (inline version)."""
    return {
        "search_success": {
            "items": [
                {
                    "id": {"videoId": "vid123", "kind": "youtube#video"},
                    "snippet": {
                        "title": "Test Video 1",
                        "channelId": "ch1",
                        "channelTitle": "Test Channel 1",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "description": "Test description 1",
                        "thumbnails": {"high": {"url": "https://example.com/thumb1.jpg"}},
                    },
                },
                {
                    "id": {"videoId": "vid456", "kind": "youtube#video"},
                    "snippet": {
                        "title": "Test Video 2",
                        "channelId": "ch2",
                        "channelTitle": "Test Channel 2",
                        "publishedAt": "2024-01-02T00:00:00Z",
                        "description": "Test description 2",
                        "thumbnails": {"medium": {"url": "https://example.com/thumb2.jpg"}},
                    },
                },
            ],
            "nextPageToken": None,
        },
        "video_details": {
            "items": [
                {
                    "id": "vid123",
                    "contentDetails": {
                        "duration": "PT10M30S",
                        "caption": "true",
                        "tags": ["tag1", "tag2"],
                        "categoryId": "22",
                        "dimension": "2d",
                        "definition": "hd",
                    },
                    "statistics": {
                        "viewCount": "1000000",
                        "likeCount": "50000",
                        "commentCount": "10000",
                    },
                    "topicDetails": {
                        "topicCategories": ["https://en.wikipedia.org/wiki/Topic:Technology"],
                        "relevantTopicIds": [],
                    },
                },
            ],
        },
        "captions_available": {
            "items": [
                {
                    "snippet": {
                        "language": "en",
                        "trackId": "track_en",
                        "trackKind": "standard",
                    },
                },
                {
                    "snippet": {
                        "language": "es",
                        "trackId": "track_es",
                        "trackKind": "ASR",
                    },
                },
            ],
        },
        "captions_unavailable": {
            "items": [],
        },
        "quota_exceeded": {
            "error": {
                "code": 403,
                "message": "Quota exceeded for this project",
                "errors": [{"reason": "quotaExceeded"}],
            },
        },
        "rate_limited": {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded",
                "errors": [{"reason": "rateLimitExceeded"}],
            },
        },
        "server_error": {
            "error": {
                "code": 500,
                "message": "Internal server error",
            },
        },
    }
