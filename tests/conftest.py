"""
Pytest configuration for test suite.

Provides fixtures and marks for integration tests that require
external resources (videos, API keys, etc.).
"""

import os
import pytest
from pathlib import Path
from typing import List, Optional


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
    except ImportError:
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
