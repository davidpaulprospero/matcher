"""
Shared test fixtures module.

US-010: Add shared test fixtures module

Provides reusable test data generators for consistent mocking across test files:
- create_mock_config(): Fully mocked Config object
- create_mock_state(): PipelineState with test data
- create_test_checkpoint(): Valid checkpoint dict
"""

from unittest.mock import Mock, MagicMock
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


def create_mock_config(
    tmp_path: Optional[Path] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    download_root: Optional[str] = None,
    **overrides
) -> Mock:
    """
    Create a fully mocked Config object for testing.

    Args:
        tmp_path: Optional temporary path to use for directories
        cache_dir: Override cache directory path
        output_dir: Override output directory path
        download_root: Override download root directory path
        **overrides: Additional config section overrides

    Returns:
        Mock Config object with all standard sections
    """
    base_path = tmp_path or Path("/tmp/test")

    mock_config = Mock()

    # Cache config
    mock_config.cache = Mock(
        cache_dir=cache_dir or str(base_path / "cache"),
        embeddings_dir=str(base_path / "cache" / "embeddings"),
        transcriptions_dir=str(base_path / "cache" / "transcriptions")
    )

    # Output config
    mock_config.output = Mock(
        output_dir=output_dir or str(base_path / "output"),
        otio_output_dir=str(base_path / "output" / "otio"),
        frame_rate=29.97,
        gap_mode="gap",
        include_disabled_tracks=True
    )

    # Transcription config
    mock_config.transcription = Mock(
        model="base",
        language="en",
        device="cpu",
        compute_type="int8",
        vad_filter=True,
        batch_size=16
    )

    # Matching config
    mock_config.matching = Mock(
        min_confidence=0.5,
        high_confidence_threshold=0.85,
        low_confidence_threshold=0.3,
        embedding_candidates=50,
        llm_rerank_count=5,
        max_clip_reuse=1,
        reuse_penalty=0.5,
        primary_provider="gemini",
        location_matching=Mock(enabled=False, geonames_username="")
    )

    # Keyword config
    mock_config.keyword = Mock(
        max_keywords=10,
        min_keyword_length=3,
        extract_entities=True,
        entity_types=["PERSON", "ORG", "GPE", "LOC"]
    )

    # Download config
    mock_config.download = Mock(
        root_dir=download_root or str(base_path / "downloads"),
        folder_name="videos",
        quality="best",
        max_retries=3,
        retry_delay=2.0,
        retry_backoff=2.0,
        rate_limit_budget=None,
        audio_first=Mock(enabled=False),
        caption_first=Mock(enabled=False),
        fallback=Mock(
            proxy=Mock(enabled=False),
            caption=Mock(enabled=True)
        )
    )

    # Duration tiers config
    mock_config.duration_tiers = Mock(
        short=Mock(min_seconds=0, max_seconds=120, videos_per_keyword=5),
        medium=Mock(min_seconds=120, max_seconds=600, videos_per_keyword=3),
        long=Mock(min_seconds=600, max_seconds=1500, videos_per_keyword=2),
        longer=Mock(min_seconds=1500, max_seconds=3600, videos_per_keyword=1)
    )

    # Pipeline config
    mock_config.pipeline = Mock(
        skip_download=False,
        skip_transcribe=False,
        skip_image_search=False,
        checkpoint_enabled=True
    )

    # Image search config
    mock_config.image_search = Mock(
        enabled=True,
        use_stock_apis=True,
        enable_consolidation=True,
        root_dir=str(base_path / "images")
    )

    # B-roll config
    mock_config.broll = Mock(
        enabled=True,
        min_words_threshold=10,
        embedding_weight=0.4,
        keyword_weight=0.35,
        entity_weight=0.25
    )

    # Scene detection config
    mock_config.scene_detection = Mock(
        enabled=True,
        detect_faces_per_scene=True,
        min_scene_duration=2.0,
        threshold=27.0
    )

    # Healing config (for self-healing pipeline)
    mock_config.healing = Mock(
        enabled=True,
        strategy="conservative",
        max_attempts_per_stage=3,
        max_total_heals=20,
        print_report=True
    )

    # Watcher config
    mock_config.watcher = Mock(
        enabled=True,
        recheck_interval_seconds=300.0,
        max_failures=5
    )

    # Apply any overrides
    for section, values in overrides.items():
        if hasattr(mock_config, section):
            section_mock = getattr(mock_config, section)
            if isinstance(values, dict):
                for key, value in values.items():
                    setattr(section_mock, key, value)
            else:
                setattr(mock_config, section, values)

    return mock_config


def create_mock_state(
    voiceover_segments: Optional[List[Dict]] = None,
    text_metadata: Optional[List[Dict]] = None,
    video_candidates: Optional[List] = None,
    embeddings: Optional[np.ndarray] = None,
    matches: Optional[List[Dict]] = None,
    **overrides
) -> Mock:
    """
    Create a mock PipelineState with test data.

    Args:
        voiceover_segments: List of voiceover segment dicts
        text_metadata: List of video metadata dicts
        video_candidates: List of video candidate objects
        embeddings: Numpy array of embeddings
        matches: List of match result dicts
        **overrides: Additional state attribute overrides

    Returns:
        Mock PipelineState object
    """
    mock_state = Mock()

    # Default voiceover segments
    mock_state.voiceover_segments = voiceover_segments or [
        {"index": 0, "text": "Welcome to the tutorial", "start": 0.0, "end": 2.5},
        {"index": 1, "text": "We will learn about Python", "start": 2.5, "end": 5.0},
        {"index": 2, "text": "Let's get started", "start": 5.0, "end": 7.0},
    ]

    # Default text metadata (video transcripts)
    mock_state.text_metadata = text_metadata or [
        {
            "video_path": "video1.mp4",
            "text": "Python tutorial introduction basics",
            "duration": 10.0,
            "word_count": 4
        },
        {
            "video_path": "video2.mp4",
            "text": "Learning programming concepts fundamentals",
            "duration": 15.0,
            "word_count": 4
        },
        {
            "video_path": "video3.mp4",
            "text": "Getting started guide for beginners",
            "duration": 8.0,
            "word_count": 5
        },
    ]

    # Default video candidates
    if video_candidates is not None:
        mock_state.video_candidates = video_candidates
    else:
        from unittest.mock import Mock as VideoMock
        mock_state.video_candidates = [
            VideoMock(file="video1.mp4", duration_tier="short", duration=10.0),
            VideoMock(file="video2.mp4", duration_tier="medium", duration=15.0),
            VideoMock(file="video3.mp4", duration_tier="short", duration=8.0),
        ]

    # Default embeddings (384-dimensional, common embedding size)
    if embeddings is not None:
        mock_state.embeddings = embeddings
    else:
        np.random.seed(42)
        mock_state.embeddings = np.random.rand(3, 384).astype(np.float32)

    # Default voiceover embeddings
    np.random.seed(123)
    mock_state.voiceover_embeddings = np.random.rand(3, 384).astype(np.float32)

    # Default matches
    mock_state.matches = matches or [
        {
            "segment_index": 0,
            "video_file": "video1.mp4",
            "confidence": 0.85,
            "start": 0.0,
            "end": 2.5,
            "strategy": "primary"
        },
        {
            "segment_index": 1,
            "video_file": "video2.mp4",
            "confidence": 0.72,
            "start": 0.0,
            "end": 5.0,
            "strategy": "primary"
        },
        {
            "segment_index": 2,
            "video_file": "video3.mp4",
            "confidence": 0.78,
            "start": 0.0,
            "end": 3.0,
            "strategy": "primary"
        },
    ]

    # Additional default state attributes
    mock_state.keywords = ["python", "tutorial", "programming", "basics"]
    mock_state.entities = []
    mock_state.entity_images = {}
    mock_state.entity_videos = {}
    mock_state.project_dir = None
    mock_state.voiceover_path = None

    # Apply any overrides
    for key, value in overrides.items():
        setattr(mock_state, key, value)

    return mock_state


def create_test_checkpoint(
    last_completed_stage: str = "MATCH",
    stages: Optional[Dict[str, Any]] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Create a valid checkpoint dict for testing.

    Args:
        last_completed_stage: Name of last completed stage
        stages: Dict of stage-specific data
        **overrides: Additional checkpoint field overrides

    Returns:
        Valid checkpoint dict
    """
    checkpoint = {
        "version": "2.0",
        "last_completed_stage": last_completed_stage,
        "timestamp": "2026-01-25T12:00:00",
        "stages": stages or {
            "ANALYZE": {
                "keywords": ["python", "tutorial", "programming"],
                "entities": [],
                "completed": True
            },
            "DOWNLOAD": {
                "videos_downloaded": 3,
                "completed": True
            },
            "TRANSCRIBE": {
                "videos_transcribed": 3,
                "completed": True
            },
            "MATCH": {
                "matches": [
                    {
                        "segment_index": 0,
                        "video_file": "video1.mp4",
                        "confidence": 0.85,
                        "start": 0.0,
                        "end": 2.5
                    },
                    {
                        "segment_index": 1,
                        "video_file": "video2.mp4",
                        "confidence": 0.72,
                        "start": 0.0,
                        "end": 5.0
                    },
                    {
                        "segment_index": 2,
                        "video_file": "video3.mp4",
                        "confidence": 0.78,
                        "start": 0.0,
                        "end": 3.0
                    }
                ],
                "completed": True
            }
        },
        "config_hash": "abc123",
        "voiceover_hash": "def456"
    }

    # Apply any overrides
    checkpoint.update(overrides)

    return checkpoint


# Convenience aliases for backward compatibility
mock_config = create_mock_config
mock_state = create_mock_state
test_checkpoint = create_test_checkpoint
