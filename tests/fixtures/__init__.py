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


# =============================================================================
# US-010: Complex Coordination Scenario Fixtures
# =============================================================================


def create_checkpoint_with_populated_stages(
    project_dir: Optional[Path] = None,
    last_completed_stage: str = "MATCH",
    include_all_stages: bool = True,
    voiceover_path: Optional[str] = None,
    match_count: int = 10,
    avg_confidence: float = 0.85,
    video_count: int = 5,
    **stage_overrides
) -> Dict[str, Any]:
    """
    Create a checkpoint with fully populated stage data for OUTPUT tests (Rule 25).

    This fixture addresses the requirement that --output-only mode needs checkpoint
    with populated `stages` dict, not just `last_completed_stage`.

    Args:
        project_dir: Path to project directory (used for file paths)
        last_completed_stage: Name of last completed stage
        include_all_stages: Whether to include data for all stages up to last_completed
        voiceover_path: Path to voiceover file
        match_count: Number of matches to generate
        avg_confidence: Average confidence for matches
        video_count: Number of videos in download stage
        **stage_overrides: Override specific stage data

    Returns:
        Valid checkpoint dict with populated stage data

    Example:
        >>> checkpoint = create_checkpoint_with_populated_stages(
        ...     project_dir=tmp_path,
        ...     match_count=20,
        ...     avg_confidence=0.92
        ... )
        >>> assert checkpoint["match"]["match_count"] == 20
        >>> assert len(checkpoint["match"]["matches"]) == 20
    """
    from datetime import datetime

    base_path = project_dir or Path("/tmp/test_project")

    # Generate matches
    matches = []
    for i in range(match_count):
        confidence = avg_confidence + (0.1 * (i % 3 - 1))  # Vary confidence
        matches.append({
            "segment_id": i,
            "video_file": f"vid{(i % video_count) + 1}.mp4",
            "confidence": min(1.0, max(0.0, confidence)),
            "start": float(i * 3),
            "end": float((i + 1) * 3)
        })

    # Generate video paths
    video_paths = [
        str(base_path / "videos" / f"vid{i+1}.mp4")
        for i in range(video_count)
    ]

    checkpoint = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "last_completed_stage": last_completed_stage,
        "config_hash": "test_hash_populated",
        "voiceover_path": voiceover_path or str(base_path / "voiceover.srt"),
        "voiceover_hash": "vo_hash_abc123",
    }

    if include_all_stages:
        checkpoint.update({
            "analyze": {
                "keywords": ["travel", "nature", "adventure", "wildlife", "documentary"],
                "segment_count": match_count,
                "topic": "documentary",
                "entities": ["Location A", "Location B"]
            },
            "entity_images": {
                "images": ["entity1.jpg", "entity2.jpg", "entity3.jpg"],
                "sources": ["google", "bing", "pexels"],
                "count": 3
            },
            "entity_videos": {
                "videos": ["stock1.mp4", "stock2.mp4"],
                "api_calls": 5,
                "sources": ["pexels", "pixabay"]
            },
            "video_metadata": {
                "metadata_count": video_count,
                "caption_languages": ["en"]
            },
            "caption": {
                "caption_count": video_count,
                "languages": ["en"],
                "source": "youtube"
            },
            "download": {
                "video_paths": video_paths,
                "count": video_count,
                "total_size_mb": video_count * 50.0
            },
            "stock": {
                "stock_videos": ["pexels_nature.mp4", "pixabay_travel.mp4"],
                "source": "mixed"
            },
            "broll_download": {
                "broll_paths": [str(base_path / "broll" / f"broll{i+1}.mp4") for i in range(3)],
                "keyword_suffixes": ["aerial", "drone", "cinematic"]
            },
            "remix": {
                "filtered_count": video_count - 1,
                "removed": ["irrelevant.mp4"]
            },
            "transcribe": {
                "transcribed_count": video_count,
                "embedding_count": video_count,
                "model": "base"
            },
            "premise": {
                "premise_detected": True,
                "topic_context": "Documentary about nature"
            },
            "scene_detection": {
                "scenes": video_count * 5,
                "broll_flagged": 3,
                "face_detection_enabled": True
            },
            "match": {
                "match_count": match_count,
                "avg_confidence": avg_confidence,
                "matches": matches,
                "strategy": "embedding_llm"
            },
            "broll_match": {
                "broll_matches": 3,
                "strategies": ["embedding", "keyword", "entity"],
                "v8_entries": 3
            },
            "iterative_match": {
                "iterations": 2,
                "improved_count": 5
            },
            "download_segments": {
                "segments_downloaded": match_count,
                "total_size_mb": match_count * 25.0
            }
        })

    # Apply stage overrides
    for stage_name, data in stage_overrides.items():
        stage_key = stage_name.lower()
        if stage_key in checkpoint:
            if isinstance(data, dict):
                checkpoint[stage_key].update(data)
            else:
                checkpoint[stage_key] = data
        else:
            checkpoint[stage_key] = data

    return checkpoint


def create_concurrent_escalation_fixture(
    num_keywords: int = 3,
    escalation_threshold: int = 2,
    cooldown_seconds: float = 0.0,
    max_tier: int = 3,
    player_clients: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create fixture for testing concurrent download escalation scenarios.

    Provides configuration and initial state for EscalationManager tests
    with multiple keywords operating concurrently.

    Args:
        num_keywords: Number of keywords to generate
        escalation_threshold: Consecutive 403s before escalating
        cooldown_seconds: Cooldown between escalations (0.0 for tests)
        max_tier: Maximum escalation tier
        player_clients: yt-dlp player_client values for extractor args

    Returns:
        Dict containing:
        - keywords: List of test keywords
        - ext_config: ExtractorArgsConfig-compatible dict
        - initial_states: Per-keyword initial state
        - impersonate_args: Default impersonation args

    Example:
        >>> fixture = create_concurrent_escalation_fixture(num_keywords=5)
        >>> assert len(fixture["keywords"]) == 5
        >>> assert fixture["ext_config"]["escalation_threshold"] == 2
    """
    keywords = [f"test_keyword_{i}" for i in range(num_keywords)]

    ext_config = {
        "enabled": True,
        "player_clients": player_clients or ["web_safari", "tv_downgraded", "web"],
        "escalation_threshold": escalation_threshold,
        "cooldown_seconds": cooldown_seconds,
        "max_tier": max_tier
    }

    # Initial state for each keyword (all at tier 1)
    initial_states = {
        kw: {
            "tier": 1,
            "consecutive_403s": 0,
            "last_escalation_time": None,
            "total_failures": 0,
            "total_successes": 0
        }
        for kw in keywords
    }

    impersonate_args = ["--impersonate", "Chrome-136:Macos-15"]

    return {
        "keywords": keywords,
        "ext_config": ext_config,
        "initial_states": initial_states,
        "impersonate_args": impersonate_args,
        # Helper for creating expected escalation results
        "expected_tier2_args": impersonate_args + [
            "--extractor-args",
            f"youtube:player_client={','.join(ext_config['player_clients'])}"
        ],
        "expected_tier3_rotate_cookies": True
    }


def create_otio_timeline_fixture(
    num_segments: int,
    frame_rate: float = 30.0,
    include_audio: bool = True,
    track_config: Optional[Dict[str, int]] = None,
    segment_duration_frames: int = 30
) -> Dict[str, Any]:
    """
    Create fixture for OTIO timeline with N items for auto-split testing (Rule 17).

    This is a parameterized fixture factory that can create timelines of any size
    for testing the 3000-item auto-split threshold.

    Args:
        num_segments: Number of video segments (clips) to create
        frame_rate: Timeline frame rate (default 30.0)
        include_audio: Whether to include matching audio track
        track_config: Dict mapping track names to clip counts (overrides num_segments)
        segment_duration_frames: Duration of each segment in frames

    Returns:
        Dict containing:
        - num_segments: Requested segment count
        - frame_rate: Timeline frame rate
        - expected_parts: Number of parts if split (1 if no split)
        - at_threshold: Whether exactly at 3000 item threshold
        - above_threshold: Whether above 3000 item threshold
        - track_count: Total tracks created
        - total_clips: Total clips across all tracks

    Example:
        >>> fixture = create_otio_timeline_fixture(3000)
        >>> assert fixture["at_threshold"] is True
        >>> assert fixture["expected_parts"] == 1

        >>> fixture = create_otio_timeline_fixture(3001)
        >>> assert fixture["above_threshold"] is True
        >>> assert fixture["expected_parts"] == 2
    """
    import math

    MAX_SEGMENTS_PER_PART = 3000

    # Calculate expected parts
    expected_parts = math.ceil(num_segments / MAX_SEGMENTS_PER_PART) if num_segments > MAX_SEGMENTS_PER_PART else 1

    # Track configuration
    if track_config is None:
        track_config = {"V1": num_segments}
        if include_audio:
            track_config["A1"] = num_segments

    total_clips = sum(track_config.values())
    track_count = len(track_config)

    return {
        "num_segments": num_segments,
        "frame_rate": frame_rate,
        "segment_duration_frames": segment_duration_frames,
        "expected_parts": expected_parts,
        "at_threshold": num_segments == MAX_SEGMENTS_PER_PART,
        "above_threshold": num_segments > MAX_SEGMENTS_PER_PART,
        "below_threshold": num_segments < MAX_SEGMENTS_PER_PART,
        "track_config": track_config,
        "track_count": track_count,
        "total_clips": total_clips,
        "include_audio": include_audio,
        # Constants for reference
        "max_segments_per_part": MAX_SEGMENTS_PER_PART,
        "warning_threshold": 2500,
        "error_threshold": 3000
    }


def create_broll_propagation_chain_state(
    num_scenes: int = 10,
    broll_ratio: float = 0.3,
    face_score_threshold: float = 0.3,
    min_words_threshold: int = 10,
    include_text_metadata: bool = True
) -> Dict[str, Any]:
    """
    Create fixture for B-roll propagation chain validation (Rule 8).

    Tests the propagation of is_broll flag from SceneDetection through
    text_metadata to Match stage, including both face-detection and
    silent-detection methods.

    Args:
        num_scenes: Total number of scenes to generate
        broll_ratio: Ratio of scenes flagged as B-roll (0.0-1.0)
        face_score_threshold: Threshold below which face_score triggers is_broll
        min_words_threshold: Word count below which scene is silent (B-roll)
        include_text_metadata: Whether to include text_metadata entries

    Returns:
        Dict containing:
        - scenes: List of scene dicts with face_score and is_broll
        - text_metadata: List of metadata entries preserving is_broll
        - expected_broll_count: Number of scenes that should be B-roll
        - face_detected_broll: Count from face detection method
        - silent_detected_broll: Count from word count method
        - thresholds: Dict of detection thresholds

    Example:
        >>> state = create_broll_propagation_chain_state(num_scenes=10, broll_ratio=0.4)
        >>> assert state["expected_broll_count"] == 4
        >>> assert len([s for s in state["scenes"] if s["is_broll"]]) == 4
    """
    import random
    random.seed(42)  # Reproducible

    num_broll = int(num_scenes * broll_ratio)
    num_face_broll = num_broll // 2
    num_silent_broll = num_broll - num_face_broll

    scenes = []
    text_metadata = []

    for i in range(num_scenes):
        is_face_broll = i < num_face_broll
        is_silent_broll = num_face_broll <= i < (num_face_broll + num_silent_broll)
        is_broll = is_face_broll or is_silent_broll

        # Face score: low for face-detected B-roll, high for non-B-roll
        if is_face_broll:
            face_score = random.uniform(0.0, face_score_threshold - 0.05)
        else:
            face_score = random.uniform(face_score_threshold + 0.1, 1.0)

        # Word count: low for silent B-roll, high for non-B-roll
        if is_silent_broll:
            word_count = random.randint(0, min_words_threshold - 1)
        else:
            word_count = random.randint(min_words_threshold + 5, 100)

        scene = {
            "scene_id": i,
            "video_path": f"video_{i // 3}.mp4",
            "start_time": float(i * 5),
            "end_time": float((i + 1) * 5),
            "face_score": face_score,
            "word_count": word_count,
            "is_broll": is_broll,
            "detection_method": "face" if is_face_broll else ("silent" if is_silent_broll else None)
        }
        scenes.append(scene)

        if include_text_metadata:
            metadata = {
                "video_path": scene["video_path"],
                "scene_index": i,
                "text": f"Scene {i} transcript " + ("" if is_silent_broll else "with spoken words " * 5),
                "start": scene["start_time"],
                "end": scene["end_time"],
                "is_broll": is_broll,
                "word_count": word_count,
                "face_score": face_score
            }
            text_metadata.append(metadata)

    return {
        "scenes": scenes,
        "text_metadata": text_metadata if include_text_metadata else [],
        "expected_broll_count": num_broll,
        "face_detected_broll": num_face_broll,
        "silent_detected_broll": num_silent_broll,
        "thresholds": {
            "face_score": face_score_threshold,
            "min_words": min_words_threshold
        },
        "num_scenes": num_scenes,
        "broll_ratio": broll_ratio,
        # V8 track expectations
        "expected_v8_entries": num_broll,
        "v8_from_face_detection": num_face_broll,
        "v8_from_silent_detection": num_silent_broll
    }


# =============================================================================
# Convenience aliases for backward compatibility
# =============================================================================
mock_config = create_mock_config
mock_state = create_mock_state
test_checkpoint = create_test_checkpoint

# US-010 fixture aliases
checkpoint_with_stages = create_checkpoint_with_populated_stages
concurrent_escalation = create_concurrent_escalation_fixture
otio_timeline = create_otio_timeline_fixture
broll_chain_state = create_broll_propagation_chain_state


# =============================================================================
# YouTube API Fixtures (US-119-009)
# =============================================================================
from tests.fixtures.youtube_api_fixtures import (
    create_mock_youtube_video_metadata,
    create_mock_video_list_response,
    create_mock_youtube_search_result,
    create_mock_search_list_response,
    create_mock_caption_track,
    create_mock_caption_list_response,
    create_mock_caption_response,
    create_mock_no_captions_response,
    create_mock_youtube_channel,
    create_mock_youtube_api_error,
    create_mock_rate_limit_error,
    create_mock_quota_exceeded_error,
    create_mock_not_found_error,
    create_mock_playlist_item,
    create_mock_playlist_list_response,
    youtube_search_to_video_search_result,
)

__all__ = [
    # Core fixtures
    "create_mock_config",
    "create_mock_state",
    "create_test_checkpoint",
    "create_checkpoint_with_populated_stages",
    "create_concurrent_escalation_fixture",
    "create_otio_timeline_fixture",
    "create_broll_propagation_chain_state",
    # Aliases
    "mock_config",
    "mock_state",
    "test_checkpoint",
    "checkpoint_with_stages",
    "concurrent_escalation",
    "otio_timeline",
    "broll_chain_state",
    # YouTube API fixtures (US-119-009)
    "create_mock_youtube_video_metadata",
    "create_mock_video_list_response",
    "create_mock_youtube_search_result",
    "create_mock_search_list_response",
    "create_mock_caption_track",
    "create_mock_caption_list_response",
    "create_mock_caption_response",
    "create_mock_no_captions_response",
    "create_mock_youtube_channel",
    "create_mock_youtube_api_error",
    "create_mock_rate_limit_error",
    "create_mock_quota_exceeded_error",
    "create_mock_not_found_error",
    "create_mock_playlist_item",
    "create_mock_playlist_list_response",
    "youtube_search_to_video_search_result",
]
