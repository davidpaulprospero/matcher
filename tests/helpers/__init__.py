"""
Test Helper Module - Reusable Assertion Helpers

US-007: Add test helper module for common assertion patterns

This module provides reusable assertion helpers for common validation patterns
across the test suite. These helpers encapsulate complex validation logic and
provide clear, actionable error messages.

Main helpers:
- assert_valid_otio_timeline(): Validate OTIO timeline structure
- assert_valid_match_result(): Validate match result dict
- assert_checkpoint_consistent(): Validate checkpoint structure and consistency
- assert_config_valid(): Validate Config object

Usage:
    from tests.helpers import (
        assert_valid_otio_timeline,
        assert_valid_match_result,
        assert_checkpoint_consistent,
        assert_config_valid,
    )

    def test_timeline_generation():
        timeline = create_timeline(matches, config)
        assert_valid_otio_timeline(timeline)

    def test_matching():
        result = matcher.match(segment)
        assert_valid_match_result(result, min_confidence=0.5)
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# =============================================================================
# OTIO Timeline Validation
# =============================================================================

def assert_valid_otio_timeline(
    timeline: Any,
    *,
    min_tracks: int = 1,
    require_video: bool = True,
    require_audio: bool = False,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.1,
    max_clips: int = 3000,
) -> None:
    """
    Assert that an OTIO timeline has valid structure.

    Validates:
    - Timeline is an opentimelineio.schema.Timeline object
    - Has expected tracks (video and/or audio)
    - All clips have valid source and available ranges
    - Duration matches expected (if provided)
    - Total clip count is within limits (Rule 17)

    Args:
        timeline: OTIO Timeline object to validate
        min_tracks: Minimum number of tracks required (default: 1)
        require_video: Whether video tracks are required (default: True)
        require_audio: Whether audio tracks are required (default: False)
        expected_duration: Expected timeline duration in seconds (optional)
        duration_tolerance: Tolerance for duration comparison (default: 0.1s)
        max_clips: Maximum allowed clips (default: 3000, per Rule 17)

    Raises:
        AssertionError: If timeline is invalid with descriptive message

    Example:
        >>> import opentimelineio as otio
        >>> timeline = create_timeline(matches, config)
        >>> assert_valid_otio_timeline(timeline, min_tracks=2, require_video=True)
    """
    try:
        import opentimelineio as otio
    except ImportError:
        raise AssertionError(
            "opentimelineio not installed. Install with: pip install opentimelineio"
        )

    # Check timeline type
    assert isinstance(timeline, otio.schema.Timeline), (
        f"Expected opentimelineio.schema.Timeline, got {type(timeline).__name__}"
    )

    # Check timeline has tracks
    tracks = list(timeline.tracks)
    assert len(tracks) >= min_tracks, (
        f"Timeline has {len(tracks)} tracks, expected at least {min_tracks}"
    )

    # Count video and audio tracks
    video_tracks = [t for t in tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in tracks if t.kind == otio.schema.TrackKind.Audio]

    if require_video:
        assert len(video_tracks) >= 1, (
            f"Timeline has no video tracks, expected at least 1"
        )

    if require_audio:
        assert len(audio_tracks) >= 1, (
            f"Timeline has no audio tracks, expected at least 1"
        )

    # Count total clips and validate each
    total_clips = 0
    clip_errors = []

    for track in tracks:
        for i, clip in enumerate(track):
            if isinstance(clip, otio.schema.Clip):
                total_clips += 1

                # Validate clip has media reference
                if clip.media_reference is None:
                    clip_errors.append(
                        f"Track '{track.name}' clip {i}: missing media_reference"
                    )
                elif isinstance(clip.media_reference, otio.schema.ExternalReference):
                    # Validate external reference has target_url
                    if not clip.media_reference.target_url:
                        clip_errors.append(
                            f"Track '{track.name}' clip {i}: ExternalReference has empty target_url"
                        )

                # Validate clip has source_range
                if clip.source_range is None:
                    clip_errors.append(
                        f"Track '{track.name}' clip {i}: missing source_range"
                    )
                else:
                    # Validate source_range has positive duration
                    duration = clip.source_range.duration
                    if duration.value <= 0:
                        clip_errors.append(
                            f"Track '{track.name}' clip {i}: source_range duration <= 0"
                        )

    # Report clip errors
    if clip_errors:
        error_sample = clip_errors[:5]
        more = f" (and {len(clip_errors) - 5} more)" if len(clip_errors) > 5 else ""
        raise AssertionError(
            f"Timeline has {len(clip_errors)} invalid clips:\n" +
            "\n".join(f"  - {e}" for e in error_sample) + more
        )

    # Check clip count limit (Rule 17)
    assert total_clips <= max_clips, (
        f"Timeline has {total_clips} clips, exceeds limit of {max_clips}. "
        "Consider using save_timeline_split() for auto-splitting."
    )

    # Check duration if specified
    if expected_duration is not None:
        actual_duration = timeline.duration().to_seconds()
        diff = abs(actual_duration - expected_duration)
        assert diff <= duration_tolerance, (
            f"Timeline duration {actual_duration:.2f}s differs from expected "
            f"{expected_duration:.2f}s by {diff:.2f}s (tolerance: {duration_tolerance}s)"
        )


# =============================================================================
# Match Result Validation
# =============================================================================

def assert_valid_match_result(
    result: Dict[str, Any],
    *,
    min_confidence: float = 0.0,
    require_video_file: bool = True,
    require_timing: bool = True,
    require_strategy: bool = False,
    valid_strategies: Optional[List[str]] = None,
) -> None:
    """
    Assert that a match result dict has valid structure and values.

    Validates:
    - Required fields are present (segment_index, confidence)
    - Confidence is within valid range [0, 1]
    - Confidence meets minimum threshold
    - Video file path is present and non-empty (if required)
    - Timing fields (start, end, video_start, video_end) are valid
    - Strategy is valid (if required)

    Args:
        result: Match result dict to validate
        min_confidence: Minimum acceptable confidence (default: 0.0)
        require_video_file: Whether video_file field is required (default: True)
        require_timing: Whether timing fields are required (default: True)
        require_strategy: Whether strategy field is required (default: False)
        valid_strategies: List of valid strategy names (optional)

    Raises:
        AssertionError: If result is invalid with descriptive message

    Example:
        >>> result = {"segment_index": 0, "confidence": 0.85, "video_file": "vid.mp4"}
        >>> assert_valid_match_result(result, min_confidence=0.5)
    """
    # Check type
    assert isinstance(result, dict), (
        f"Expected dict, got {type(result).__name__}"
    )

    # Required fields
    assert "segment_index" in result, (
        "Match result missing 'segment_index' field"
    )
    assert "confidence" in result, (
        "Match result missing 'confidence' field"
    )

    # Validate segment_index
    seg_idx = result["segment_index"]
    assert isinstance(seg_idx, (int, float)) and seg_idx >= 0, (
        f"segment_index must be non-negative number, got {seg_idx}"
    )

    # Validate confidence
    confidence = result["confidence"]
    assert isinstance(confidence, (int, float)), (
        f"confidence must be a number, got {type(confidence).__name__}"
    )
    assert 0 <= confidence <= 1, (
        f"confidence must be in range [0, 1], got {confidence}"
    )
    assert confidence >= min_confidence, (
        f"confidence {confidence} is below minimum threshold {min_confidence}"
    )

    # Validate video_file if required
    if require_video_file:
        assert "video_file" in result, (
            "Match result missing 'video_file' field"
        )
        video_file = result["video_file"]
        assert video_file and isinstance(video_file, str), (
            f"video_file must be non-empty string, got {video_file!r}"
        )

    # Validate timing fields if required
    if require_timing:
        timing_fields = []
        # Check for either (start, end) or (video_start, video_end)
        if "start" in result or "end" in result:
            timing_fields = ["start", "end"]
        elif "video_start" in result or "video_end" in result:
            timing_fields = ["video_start", "video_end"]

        for field in timing_fields:
            if field in result:
                value = result[field]
                assert isinstance(value, (int, float)), (
                    f"Timing field '{field}' must be a number, got {type(value).__name__}"
                )
                assert value >= 0, (
                    f"Timing field '{field}' must be >= 0, got {value}"
                )

        # If both start and end exist, validate order
        start_field = "start" if "start" in result else "video_start"
        end_field = "end" if "end" in result else "video_end"
        if start_field in result and end_field in result:
            start = result[start_field]
            end = result[end_field]
            assert start <= end, (
                f"Start time {start} must be <= end time {end}"
            )

    # Validate strategy if required
    if require_strategy:
        assert "strategy" in result, (
            "Match result missing 'strategy' field"
        )
        strategy = result["strategy"]
        assert strategy and isinstance(strategy, str), (
            f"strategy must be non-empty string, got {strategy!r}"
        )

        if valid_strategies:
            assert strategy in valid_strategies, (
                f"strategy '{strategy}' not in valid strategies: {valid_strategies}"
            )


# =============================================================================
# Checkpoint Validation
# =============================================================================

# Valid stage names for checkpoint validation
VALID_STAGES = [
    "ANALYZE",
    "ENTITY_IMAGES",
    "ENTITY_VIDEOS",
    "VIDEO_METADATA",
    "CAPTION",
    "DOWNLOAD",
    "STOCK",
    "BROLL_DOWNLOAD",
    "REMIX",
    "TRANSCRIBE",
    "PREMISE",
    "SCENE_DETECTION",
    "MATCH",
    "BROLL_MATCH",
    "ITERATIVE_MATCH",
    "DOWNLOAD_SEGMENTS",
    "OUTPUT",
]


def assert_checkpoint_consistent(
    checkpoint: Dict[str, Any],
    *,
    require_version: bool = True,
    require_stages: bool = False,
    require_voiceover: bool = False,
    expected_stage: Optional[str] = None,
    min_matches: Optional[int] = None,
) -> None:
    """
    Assert that a checkpoint dict has consistent structure and data.

    Validates:
    - Required fields (version, last_completed_stage)
    - Stage name is valid
    - Stages dict is populated (if required, for --output-only mode)
    - Voiceover path and hash are present (if required)
    - Match data consistency (if min_matches specified)

    Args:
        checkpoint: Checkpoint dict to validate
        require_version: Whether version field is required (default: True)
        require_stages: Whether stages dict must be populated (default: False)
        require_voiceover: Whether voiceover fields are required (default: False)
        expected_stage: Expected last_completed_stage value (optional)
        min_matches: Minimum number of matches in match stage (optional)

    Raises:
        AssertionError: If checkpoint is invalid with descriptive message

    Example:
        >>> checkpoint = {"version": "1.0", "last_completed_stage": "MATCH"}
        >>> assert_checkpoint_consistent(checkpoint, expected_stage="MATCH")
    """
    # Check type
    assert isinstance(checkpoint, dict), (
        f"Expected dict, got {type(checkpoint).__name__}"
    )

    # Validate version
    if require_version:
        assert "version" in checkpoint, (
            "Checkpoint missing 'version' field"
        )
        version = checkpoint["version"]
        assert version and isinstance(version, str), (
            f"version must be non-empty string, got {version!r}"
        )

    # Validate last_completed_stage
    assert "last_completed_stage" in checkpoint, (
        "Checkpoint missing 'last_completed_stage' field"
    )
    stage = checkpoint["last_completed_stage"]
    if stage:  # Can be empty string for fresh checkpoint
        assert stage in VALID_STAGES, (
            f"Invalid stage '{stage}', must be one of: {', '.join(VALID_STAGES)}"
        )

    # Check expected stage
    if expected_stage is not None:
        assert stage == expected_stage, (
            f"Expected stage '{expected_stage}', got '{stage}'"
        )

    # Validate stages dict (for --output-only mode, see Rule 25)
    if require_stages:
        # Check for stages dict (new format) or individual stage keys (old format)
        has_stages = "stages" in checkpoint and checkpoint["stages"]
        has_stage_keys = any(
            k.lower() in [s.lower() for s in VALID_STAGES]
            for k in checkpoint.keys()
        )

        assert has_stages or has_stage_keys, (
            "Checkpoint has no stage data. Required for --output-only mode. "
            "Use --match-only instead if checkpoint is corrupted."
        )

    # Validate voiceover fields
    if require_voiceover:
        assert "voiceover_path" in checkpoint, (
            "Checkpoint missing 'voiceover_path' field"
        )
        vo_path = checkpoint["voiceover_path"]
        assert vo_path and isinstance(vo_path, str), (
            f"voiceover_path must be non-empty string, got {vo_path!r}"
        )

    # Validate match data
    if min_matches is not None:
        # Check both new format (stages.match) and old format (match)
        match_data = None
        if "stages" in checkpoint and isinstance(checkpoint.get("stages"), dict):
            match_data = checkpoint["stages"].get("match", {})
            if not match_data:
                match_data = checkpoint["stages"].get("MATCH", {})
        if not match_data:
            match_data = checkpoint.get("match", {})

        matches = match_data.get("matches", [])
        assert len(matches) >= min_matches, (
            f"Checkpoint has {len(matches)} matches, expected at least {min_matches}"
        )

        # Validate match count consistency
        if "match_count" in match_data:
            count = match_data["match_count"]
            assert count == len(matches), (
                f"Inconsistent match data: match_count={count} but "
                f"len(matches)={len(matches)}"
            )


# =============================================================================
# Config Validation
# =============================================================================

def assert_config_valid(
    config: Any,
    *,
    require_api_keys: bool = False,
    require_matching: bool = True,
    require_download: bool = False,
    check_constraints: bool = True,
    allowed_errors: Optional[List[str]] = None,
) -> List[str]:
    """
    Assert that a Config object is valid and internally consistent.

    Validates:
    - Config has expected section attributes
    - Required API keys are set (if require_api_keys=True)
    - Value ranges are valid (confidence thresholds, worker counts, etc.)
    - Enum values are valid (providers, strategies)
    - Constraint relationships hold (min <= max, etc.)

    Args:
        config: Config object to validate
        require_api_keys: Whether API keys must be set (default: False)
        require_matching: Whether matching section is required (default: True)
        require_download: Whether download section is required (default: False)
        check_constraints: Whether to check constraint relationships (default: True)
        allowed_errors: List of error message substrings to ignore (optional)

    Returns:
        List of validation errors (empty if valid)

    Raises:
        AssertionError: If config is fundamentally invalid (missing required sections)

    Example:
        >>> config = load_config("config.yaml")
        >>> errors = assert_config_valid(config, require_api_keys=True)
        >>> assert not errors, f"Config has errors: {errors}"
    """
    errors = []
    allowed_errors = allowed_errors or []

    def is_allowed(error: str) -> bool:
        return any(allowed in error for allowed in allowed_errors)

    # Check config is not None
    assert config is not None, "Config is None"

    # Check required sections exist
    required_sections = []
    if require_matching:
        required_sections.append("matching")
    if require_download:
        required_sections.append("download")

    for section in required_sections:
        assert hasattr(config, section), (
            f"Config missing required section '{section}'"
        )

    # Use Config's built-in validate() if available
    if hasattr(config, 'validate') and callable(config.validate):
        builtin_errors = config.validate()
        for error in builtin_errors:
            if not is_allowed(error):
                errors.append(error)
        return errors

    # Manual validation for mock configs or configs without validate()
    # Check matching section
    if hasattr(config, 'matching'):
        m = config.matching

        # Check min_confidence range
        if hasattr(m, 'min_confidence'):
            if not (0 <= m.min_confidence <= 1):
                error = f"matching.min_confidence must be 0-1, got {m.min_confidence}"
                if not is_allowed(error):
                    errors.append(error)

        # Check provider values
        valid_providers = {'gemini', 'anthropic', 'local', 'embedding_only', 'ollama'}
        if hasattr(m, 'primary_provider'):
            if m.primary_provider not in valid_providers:
                error = (
                    f"matching.primary_provider must be one of {valid_providers}, "
                    f"got '{m.primary_provider}'"
                )
                if not is_allowed(error):
                    errors.append(error)

        # Check embedding_candidates
        if hasattr(m, 'embedding_candidates'):
            if m.embedding_candidates < 1:
                error = f"matching.embedding_candidates must be >= 1, got {m.embedding_candidates}"
                if not is_allowed(error):
                    errors.append(error)

    # Check constraint relationships
    if check_constraints and hasattr(config, 'matching'):
        m = config.matching

        # min_confidence <= high_confidence_threshold
        if hasattr(m, 'min_confidence') and hasattr(m, 'high_confidence_threshold'):
            if m.min_confidence > m.high_confidence_threshold:
                error = (
                    f"matching.min_confidence ({m.min_confidence}) should be <= "
                    f"high_confidence_threshold ({m.high_confidence_threshold})"
                )
                if not is_allowed(error):
                    errors.append(error)

    # Check API keys if required
    if require_api_keys:
        if hasattr(config, 'api_keys'):
            api = config.api_keys
            if hasattr(api, 'gemini_api_key') and not api.gemini_api_key:
                error = "GEMINI_API_KEY not set"
                if not is_allowed(error):
                    errors.append(error)
        elif hasattr(config, 'gemini_api_key') and not config.gemini_api_key:
            error = "GEMINI_API_KEY not set"
            if not is_allowed(error):
                errors.append(error)

    # Check transcription section
    if hasattr(config, 'transcription'):
        t = config.transcription
        if hasattr(t, 'max_workers') and t.max_workers < 1:
            error = f"transcription.max_workers must be >= 1, got {t.max_workers}"
            if not is_allowed(error):
                errors.append(error)

    # Check keyword section
    if hasattr(config, 'keyword'):
        k = config.keyword
        if hasattr(k, 'max_keywords') and k.max_keywords < 1:
            error = f"keyword.max_keywords must be >= 1, got {k.max_keywords}"
            if not is_allowed(error):
                errors.append(error)

    return errors


# =============================================================================
# Additional Utility Assertions
# =============================================================================

def assert_file_exists(path: Union[str, Path], description: str = "File") -> None:
    """
    Assert that a file exists at the given path.

    Args:
        path: Path to check
        description: Description for error message (default: "File")

    Raises:
        AssertionError: If file doesn't exist
    """
    path = Path(path)
    assert path.exists(), f"{description} not found: {path}"
    assert path.is_file(), f"{description} is not a file: {path}"


def assert_dir_exists(path: Union[str, Path], description: str = "Directory") -> None:
    """
    Assert that a directory exists at the given path.

    Args:
        path: Path to check
        description: Description for error message (default: "Directory")

    Raises:
        AssertionError: If directory doesn't exist
    """
    path = Path(path)
    assert path.exists(), f"{description} not found: {path}"
    assert path.is_dir(), f"{description} is not a directory: {path}"


def assert_json_structure(
    data: Any,
    required_keys: Optional[List[str]] = None,
    type_checks: Optional[Dict[str, type]] = None,
) -> None:
    """
    Assert that JSON data has expected structure.

    Args:
        data: JSON data (dict) to check
        required_keys: List of keys that must be present
        type_checks: Dict mapping keys to expected types

    Raises:
        AssertionError: If structure is invalid
    """
    assert isinstance(data, dict), f"Expected dict, got {type(data).__name__}"

    if required_keys:
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    if type_checks:
        for key, expected_type in type_checks.items():
            if key in data:
                assert isinstance(data[key], expected_type), (
                    f"Key '{key}' should be {expected_type.__name__}, "
                    f"got {type(data[key]).__name__}"
                )


# =============================================================================
# Agent Healer Assertions
# =============================================================================

# Valid healer names
VALID_HEALERS = [
    "checkpoint-healer",
    "api-healer",
    "download-healer",
    "disk-healer",
    "path-healer",
    "otio-healer",
    "llm-healer",
]

# Valid healing strategies
VALID_STRATEGIES = [
    "aggressive",
    "conservative",
    "interactive",
    "minimal",
]


def assert_healer_attempt_logged(
    context: Any,
    healer_name: str,
    attempt_num: Optional[int] = None,
    *,
    expected_message: Optional[str] = None,
) -> None:
    """
    Assert that a healer attempt was logged in the healing context.

    Validates that a specific healer logged an attempt during the healing
    process. Works with HealingOrchestrator contexts, mock loggers, or
    log file data.

    Args:
        context: HealingOrchestrator, mock logger, or dict with logged attempts
        healer_name: Name of the healer (e.g., "checkpoint-healer", "api-healer")
        attempt_num: Specific attempt number to check (1-indexed, optional)
        expected_message: Substring expected in the attempt message (optional)

    Raises:
        AssertionError: If healer attempt was not logged

    Example:
        >>> orchestrator.coordinate_heal(error, state, "MATCH", [])
        >>> assert_healer_attempt_logged(orchestrator, "api-healer", 1)
        >>> assert_healer_attempt_logged(orchestrator, "checkpoint-healer",
        ...     expected_message="Attempting to restore")
    """
    # Normalize healer name
    healer_name_normalized = healer_name.lower().replace("_", "-")

    # Extract attempts from different context types
    attempts = _extract_healer_attempts(context, healer_name_normalized)

    # Check that healer has logged attempts
    assert len(attempts) > 0, (
        f"Healer '{healer_name}' did not log any attempts. "
        f"Available healers: {_get_logged_healers(context)}"
    )

    # Check specific attempt number if provided
    if attempt_num is not None:
        assert attempt_num > 0, "attempt_num must be 1-indexed (>= 1)"
        assert len(attempts) >= attempt_num, (
            f"Healer '{healer_name}' only logged {len(attempts)} attempts, "
            f"expected at least {attempt_num}"
        )

    # Check message content if provided
    if expected_message is not None:
        matching_attempts = [
            a for a in attempts
            if expected_message in str(_get_attempt_message(a))
        ]
        assert len(matching_attempts) > 0, (
            f"No attempts from '{healer_name}' contain message: '{expected_message}'. "
            f"Found messages: {[_get_attempt_message(a) for a in attempts[:3]]}"
        )


def assert_healing_strategy_applied(
    context: Any,
    strategy_name: str,
    *,
    check_max_attempts: Optional[int] = None,
    check_heal_delay: Optional[float] = None,
) -> None:
    """
    Assert that a specific healing strategy was applied.

    Validates that the healing context used a specific strategy mode
    (aggressive, conservative, interactive, or minimal) and optionally
    checks strategy parameters.

    Args:
        context: HealingOrchestrator, HealingStrategy, or context dict
        strategy_name: Expected strategy name ("aggressive", "conservative", etc.)
        check_max_attempts: Verify max_attempts_per_stage value (optional)
        check_heal_delay: Verify heal_delay value (optional)

    Raises:
        AssertionError: If strategy was not applied or parameters don't match

    Example:
        >>> runner = ResilientRunner(config, orchestrator)
        >>> assert_healing_strategy_applied(orchestrator, "conservative")
        >>> assert_healing_strategy_applied(orchestrator, "aggressive",
        ...     check_max_attempts=5, check_heal_delay=1.0)
    """
    # Validate strategy name
    strategy_name_lower = strategy_name.lower()
    assert strategy_name_lower in VALID_STRATEGIES, (
        f"Invalid strategy '{strategy_name}', must be one of: {VALID_STRATEGIES}"
    )

    # Extract strategy from context
    strategy = _extract_strategy(context)
    assert strategy is not None, (
        "Could not extract healing strategy from context. "
        "Expected HealingOrchestrator, HealingStrategy, or dict with 'strategy' key."
    )

    # Check strategy mode
    actual_mode = _get_strategy_mode(strategy)
    assert actual_mode.lower() == strategy_name_lower, (
        f"Expected strategy '{strategy_name}', but '{actual_mode}' was applied"
    )

    # Check max_attempts_per_stage if provided
    if check_max_attempts is not None:
        actual_max = _get_strategy_attr(strategy, 'max_attempts_per_stage')
        assert actual_max == check_max_attempts, (
            f"Strategy max_attempts_per_stage is {actual_max}, expected {check_max_attempts}"
        )

    # Check heal_delay if provided
    if check_heal_delay is not None:
        actual_delay = _get_strategy_attr(strategy, 'heal_delay')
        assert abs(actual_delay - check_heal_delay) < 0.01, (
            f"Strategy heal_delay is {actual_delay}, expected {check_heal_delay}"
        )


def assert_recovery_metrics_valid(
    metrics: Any,
    *,
    expected_attempts: Optional[int] = None,
    expected_success: Optional[bool] = None,
    min_successful_heals: Optional[int] = None,
    max_failed_heals: Optional[int] = None,
    expected_healers_used: Optional[List[str]] = None,
    expected_stages_healed: Optional[List[str]] = None,
) -> None:
    """
    Assert that recovery metrics are valid and match expectations.

    Validates HealingMetrics from an orchestrator or runner, checking
    heal counts, success rates, and optional specific expectations.

    Args:
        metrics: HealingMetrics object or dict with metric fields
        expected_attempts: Total expected heal attempts (optional)
        expected_success: Expected overall success (True/False, optional)
        min_successful_heals: Minimum successful heals required (optional)
        max_failed_heals: Maximum failed heals allowed (optional)
        expected_healers_used: List of healer names that should have been used (optional)
        expected_stages_healed: List of stage names that should have been healed (optional)

    Raises:
        AssertionError: If metrics are invalid or don't match expectations

    Example:
        >>> metrics = orchestrator.get_metrics()
        >>> assert_recovery_metrics_valid(metrics, expected_attempts=3, expected_success=True)
        >>> assert_recovery_metrics_valid(metrics, min_successful_heals=2, max_failed_heals=1)
    """
    # Extract metrics values
    total_heals = _get_metric(metrics, 'total_heals', 0)
    successful_heals = _get_metric(metrics, 'successful_heals', 0)
    failed_heals = _get_metric(metrics, 'failed_heals', 0)
    heals_by_healer = _get_metric(metrics, 'heals_by_healer', {})
    heals_by_stage = _get_metric(metrics, 'heals_by_stage', {})

    # Basic consistency check
    assert total_heals >= 0, f"total_heals cannot be negative: {total_heals}"
    assert successful_heals >= 0, f"successful_heals cannot be negative: {successful_heals}"
    assert failed_heals >= 0, f"failed_heals cannot be negative: {failed_heals}"
    assert successful_heals + failed_heals <= total_heals, (
        f"successful_heals ({successful_heals}) + failed_heals ({failed_heals}) "
        f"exceeds total_heals ({total_heals})"
    )

    # Check expected_attempts
    if expected_attempts is not None:
        assert total_heals == expected_attempts, (
            f"Expected {expected_attempts} heal attempts, got {total_heals}"
        )

    # Check expected_success
    if expected_success is not None:
        if expected_success:
            assert successful_heals > 0 or total_heals == 0, (
                "Expected successful recovery but no successful heals recorded"
            )
        else:
            assert failed_heals > 0, (
                "Expected failed recovery but no failed heals recorded"
            )

    # Check min_successful_heals
    if min_successful_heals is not None:
        assert successful_heals >= min_successful_heals, (
            f"Expected at least {min_successful_heals} successful heals, "
            f"got {successful_heals}"
        )

    # Check max_failed_heals
    if max_failed_heals is not None:
        assert failed_heals <= max_failed_heals, (
            f"Expected at most {max_failed_heals} failed heals, "
            f"got {failed_heals}"
        )

    # Check expected_healers_used
    if expected_healers_used is not None:
        healers_used = set(heals_by_healer.keys())
        expected_set = set(h.lower().replace("_", "-") for h in expected_healers_used)
        normalized_used = set(h.lower().replace("_", "-") for h in healers_used)
        missing = expected_set - normalized_used
        assert not missing, (
            f"Expected healers not used: {missing}. "
            f"Healers that were used: {healers_used}"
        )

    # Check expected_stages_healed
    if expected_stages_healed is not None:
        stages_healed = set(heals_by_stage.keys())
        expected_stages = set(s.upper() for s in expected_stages_healed)
        normalized_healed = set(s.upper() for s in stages_healed)
        missing = expected_stages - normalized_healed
        assert not missing, (
            f"Expected stages not healed: {missing}. "
            f"Stages that were healed: {stages_healed}"
        )


def assert_healer_chain_executed(
    context: Any,
    expected_healers: List[str],
    *,
    in_order: bool = False,
    all_required: bool = True,
) -> None:
    """
    Assert that a chain of healers was executed during healing.

    Validates that multiple healers were invoked during a healing attempt,
    optionally checking execution order and completeness.

    Args:
        context: HealingOrchestrator, mock logger, or context dict
        expected_healers: List of healer names expected to be invoked
        in_order: Whether healers must have been invoked in specified order (default: False)
        all_required: Whether all specified healers must have been invoked (default: True)

    Raises:
        AssertionError: If healer chain was not executed as expected

    Example:
        >>> # Test that fallback chain was triggered
        >>> assert_healer_chain_executed(orchestrator,
        ...     ["checkpoint-healer", "api-healer", "download-healer"])
        >>> # Test strict execution order
        >>> assert_healer_chain_executed(orchestrator,
        ...     ["checkpoint-healer", "api-healer"], in_order=True)
    """
    assert len(expected_healers) > 0, "expected_healers list cannot be empty"

    # Get logged healers from context
    logged_healers = _get_logged_healers(context)

    if not logged_healers:
        raise AssertionError(
            f"No healers logged any attempts. Expected: {expected_healers}"
        )

    # Normalize names
    expected_normalized = [h.lower().replace("_", "-") for h in expected_healers]
    logged_normalized = [h.lower().replace("_", "-") for h in logged_healers]

    if all_required:
        missing = [h for h in expected_normalized if h not in logged_normalized]
        if missing:
            raise AssertionError(
                f"Required healers not executed: {missing}. "
                f"Executed healers: {logged_healers}"
            )

    if in_order:
        # Find positions of expected healers in logged order
        positions = []
        for healer in expected_normalized:
            try:
                pos = logged_normalized.index(healer)
                positions.append(pos)
            except ValueError:
                if all_required:
                    raise AssertionError(
                        f"Healer '{healer}' not found in execution log"
                    )

        # Check positions are monotonically increasing
        for i in range(1, len(positions)):
            if positions[i] <= positions[i - 1]:
                raise AssertionError(
                    f"Healers not executed in expected order. "
                    f"Expected: {expected_healers}. "
                    f"Actual order: {logged_healers}"
                )


# =============================================================================
# Internal Helper Functions for Healer Assertions
# =============================================================================

def _extract_healer_attempts(context: Any, healer_name: str) -> List[Any]:
    """Extract attempt records for a specific healer from various context types."""
    attempts = []

    # HealingOrchestrator or similar with get_metrics()
    if hasattr(context, 'get_metrics'):
        metrics = context.get_metrics()
        if hasattr(metrics, 'heals_by_healer') or isinstance(getattr(metrics, 'heals_by_healer', None), dict):
            heals = getattr(metrics, 'heals_by_healer', {})
            for name, count in heals.items():
                if name.lower().replace("_", "-") == healer_name:
                    attempts.extend([{"healer": name}] * count)

    # Object with healing_log attribute (list of entries)
    if hasattr(context, 'healing_log'):
        log = context.healing_log
        if isinstance(log, list):
            for entry in log:
                entry_healer = _get_entry_healer(entry)
                if entry_healer and entry_healer.lower().replace("_", "-") == healer_name:
                    attempts.append(entry)

    # Object with _healer_attempts attribute (test mocks)
    if hasattr(context, '_healer_attempts'):
        healer_attempts = context._healer_attempts
        if isinstance(healer_attempts, dict):
            if healer_name in healer_attempts:
                attempts.extend(healer_attempts[healer_name])

    # Dict context with 'attempts' or 'healers' keys
    if isinstance(context, dict):
        if 'attempts' in context:
            for attempt in context['attempts']:
                entry_healer = _get_entry_healer(attempt)
                if entry_healer and entry_healer.lower().replace("_", "-") == healer_name:
                    attempts.append(attempt)
        if 'healers' in context and healer_name in context['healers']:
            attempts.extend(context['healers'][healer_name])

    return attempts


def _get_entry_healer(entry: Any) -> Optional[str]:
    """Extract healer name from a log entry."""
    if isinstance(entry, dict):
        return entry.get('healer') or entry.get('healer_name') or entry.get('component')
    if hasattr(entry, 'component'):
        return entry.component
    if hasattr(entry, 'healer'):
        return entry.healer
    return None


def _get_attempt_message(attempt: Any) -> str:
    """Extract message from an attempt record."""
    if isinstance(attempt, dict):
        return attempt.get('message', '') or attempt.get('error_message', '')
    if hasattr(attempt, 'message'):
        return attempt.message
    if hasattr(attempt, 'error_message'):
        return attempt.error_message
    return str(attempt)


def _get_logged_healers(context: Any) -> List[str]:
    """Get list of all healers that logged attempts."""
    healers = []

    # HealingOrchestrator or similar with get_metrics()
    if hasattr(context, 'get_metrics'):
        metrics = context.get_metrics()
        if hasattr(metrics, 'heals_by_healer'):
            healers.extend(metrics.heals_by_healer.keys())

    # Object with healing_log attribute
    if hasattr(context, 'healing_log'):
        log = context.healing_log
        if isinstance(log, list):
            for entry in log:
                healer = _get_entry_healer(entry)
                if healer and healer not in healers:
                    healers.append(healer)

    # Object with _healer_attempts attribute
    if hasattr(context, '_healer_attempts'):
        if isinstance(context._healer_attempts, dict):
            for name, attempts in context._healer_attempts.items():
                if attempts and name not in healers:
                    healers.append(name)

    # Dict context
    if isinstance(context, dict):
        if 'attempts' in context:
            for attempt in context['attempts']:
                healer = _get_entry_healer(attempt)
                if healer and healer not in healers:
                    healers.append(healer)
        if 'healers' in context:
            healers.extend(context['healers'].keys())

    return healers


def _extract_strategy(context: Any) -> Any:
    """Extract HealingStrategy from various context types."""
    # Direct HealingStrategy
    if hasattr(context, 'mode') and hasattr(context, 'max_attempts_per_stage'):
        return context

    # HealingOrchestrator with strategy attribute
    if hasattr(context, 'strategy'):
        return context.strategy

    # Dict with 'strategy' key
    if isinstance(context, dict) and 'strategy' in context:
        return context['strategy']

    return None


def _get_strategy_mode(strategy: Any) -> str:
    """Get the mode name from a strategy object."""
    if isinstance(strategy, dict):
        mode = strategy.get('mode', '')
        return mode.value if hasattr(mode, 'value') else str(mode)

    if hasattr(strategy, 'mode'):
        mode = strategy.mode
        return mode.value if hasattr(mode, 'value') else str(mode)

    return str(strategy)


def _get_strategy_attr(strategy: Any, attr: str) -> Any:
    """Get an attribute from a strategy object."""
    if isinstance(strategy, dict):
        return strategy.get(attr)
    return getattr(strategy, attr, None)


def _get_metric(metrics: Any, name: str, default: Any) -> Any:
    """Get a metric value from metrics object or dict."""
    if isinstance(metrics, dict):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main assertion helpers
    'assert_valid_otio_timeline',
    'assert_valid_match_result',
    'assert_checkpoint_consistent',
    'assert_config_valid',
    # Agent healer assertion helpers
    'assert_healer_attempt_logged',
    'assert_healing_strategy_applied',
    'assert_recovery_metrics_valid',
    'assert_healer_chain_executed',
    # Utility assertions
    'assert_file_exists',
    'assert_dir_exists',
    'assert_json_structure',
    # Constants
    'VALID_STAGES',
    'VALID_HEALERS',
    'VALID_STRATEGIES',
]
