"""
Edge case tests for config section __post_init__ behavior.

Covers:
- IterativeMatchingConfig boundary values (0, negative, very large)
- DownloadConfig nested dataclass dict coercion (Rule 2)
- MatchingConfig post_init clamping for out-of-range values
- Missing optional fields with correct defaults
- CaptionRetryBudgetConfig validation errors

US-47-009: Config section post_init edge case tests.
"""

import pytest

from src.config.sections.iterative_matching import IterativeMatchingConfig
from src.config.sections.download import (
    DownloadConfig,
    CaptionFirstConfig,
    CaptionCircuitBreakerConfig,
    CaptionRetryBudgetConfig,
    CookieRotationConfig,
    RateLimitConfig,
    RateLimitBudgetConfig,
    SpeedTrackingConfig,
    CircuitBreakerConfig,
    BatchRetryConfig,
    VPNConfig,
    MullvadConfig,
    ImpersonationConfig,
    ExtractorArgsConfig,
    AudioFirstConfig,
    SpeechScreeningConfig,
    ZeroDownloadRemixConfig,
    LLMTitleFilterConfig,
    RemixConfig,
)
from src.config.sections.matching import (
    MatchingConfig,
    LocationMatchingConfig,
    ChapterDetectionConfig,
)


# ─── IterativeMatchingConfig boundary values ─────────────────────────────


class TestIterativeMatchingConfigBoundaryValues:
    """Test IterativeMatchingConfig post_init with boundary values."""

    @pytest.mark.fast
    def test_target_confidence_clamped_to_zero(self):
        """Negative target_confidence is clamped to 0.0."""
        config = IterativeMatchingConfig(target_confidence=-0.5)
        assert config.target_confidence == 0.0

    @pytest.mark.fast
    def test_target_confidence_clamped_to_one(self):
        """target_confidence above 1.0 is clamped to 1.0."""
        config = IterativeMatchingConfig(target_confidence=1.5)
        assert config.target_confidence == 1.0

    @pytest.mark.fast
    def test_target_confidence_zero(self):
        """target_confidence of 0.0 is valid."""
        config = IterativeMatchingConfig(target_confidence=0.0)
        assert config.target_confidence == 0.0

    @pytest.mark.fast
    def test_target_confidence_one(self):
        """target_confidence of 1.0 is valid."""
        config = IterativeMatchingConfig(target_confidence=1.0)
        assert config.target_confidence == 1.0

    @pytest.mark.fast
    def test_source_spacing_negative_clamped_to_zero(self):
        """Negative source_spacing_seconds is clamped to 0.0."""
        config = IterativeMatchingConfig(source_spacing_seconds=-100.0)
        assert config.source_spacing_seconds == 0.0

    @pytest.mark.fast
    def test_source_spacing_zero(self):
        """source_spacing_seconds of 0.0 is valid."""
        config = IterativeMatchingConfig(source_spacing_seconds=0.0)
        assert config.source_spacing_seconds == 0.0

    @pytest.mark.fast
    def test_max_iterations_zero_clamped_to_one(self):
        """max_iterations of 0 is clamped to 1."""
        config = IterativeMatchingConfig(max_iterations=0)
        assert config.max_iterations == 1

    @pytest.mark.fast
    def test_max_iterations_negative_clamped_to_one(self):
        """Negative max_iterations is clamped to 1."""
        config = IterativeMatchingConfig(max_iterations=-5)
        assert config.max_iterations == 1

    @pytest.mark.fast
    def test_max_iterations_very_large(self):
        """Very large max_iterations is preserved."""
        config = IterativeMatchingConfig(max_iterations=999999)
        assert config.max_iterations == 999999

    @pytest.mark.fast
    def test_min_gap_percentage_negative_clamped_to_zero(self):
        """Negative min_gap_percentage is clamped to 0.0."""
        config = IterativeMatchingConfig(min_gap_percentage=-0.1)
        assert config.min_gap_percentage == 0.0

    @pytest.mark.fast
    def test_min_gap_percentage_above_one_clamped(self):
        """min_gap_percentage above 1.0 is clamped to 1.0."""
        config = IterativeMatchingConfig(min_gap_percentage=2.0)
        assert config.min_gap_percentage == 1.0

    @pytest.mark.fast
    def test_search_results_per_gap_zero_clamped_to_one(self):
        """search_results_per_gap of 0 is clamped to 1."""
        config = IterativeMatchingConfig(search_results_per_gap=0)
        assert config.search_results_per_gap == 1

    @pytest.mark.fast
    def test_search_results_per_gap_negative_clamped_to_one(self):
        """Negative search_results_per_gap is clamped to 1."""
        config = IterativeMatchingConfig(search_results_per_gap=-10)
        assert config.search_results_per_gap == 1

    @pytest.mark.fast
    def test_max_new_videos_per_pass_zero_clamped_to_one(self):
        """max_new_videos_per_pass of 0 is clamped to 1."""
        config = IterativeMatchingConfig(max_new_videos_per_pass=0)
        assert config.max_new_videos_per_pass == 1

    @pytest.mark.fast
    def test_caption_batch_size_zero_clamped_to_one(self):
        """caption_batch_size of 0 is clamped to 1."""
        config = IterativeMatchingConfig(caption_batch_size=0)
        assert config.caption_batch_size == 1

    @pytest.mark.fast
    def test_caption_batch_size_negative_clamped_to_one(self):
        """Negative caption_batch_size is clamped to 1."""
        config = IterativeMatchingConfig(caption_batch_size=-3)
        assert config.caption_batch_size == 1

    @pytest.mark.fast
    def test_caption_fetch_delay_negative_clamped_to_zero(self):
        """Negative caption_fetch_delay is clamped to 0.0."""
        config = IterativeMatchingConfig(caption_fetch_delay=-1.0)
        assert config.caption_fetch_delay == 0.0

    @pytest.mark.fast
    def test_caption_fetch_delay_zero(self):
        """caption_fetch_delay of 0.0 is valid."""
        config = IterativeMatchingConfig(caption_fetch_delay=0.0)
        assert config.caption_fetch_delay == 0.0

    @pytest.mark.fast
    def test_multiple_boundary_values_at_once(self):
        """Multiple boundary values are all clamped correctly."""
        config = IterativeMatchingConfig(
            target_confidence=-1.0,
            source_spacing_seconds=-50.0,
            max_iterations=-10,
            min_gap_percentage=5.0,
            search_results_per_gap=-100,
            max_new_videos_per_pass=0,
            caption_batch_size=0,
            caption_fetch_delay=-2.5,
        )
        assert config.target_confidence == 0.0
        assert config.source_spacing_seconds == 0.0
        assert config.max_iterations == 1
        assert config.min_gap_percentage == 1.0
        assert config.search_results_per_gap == 1
        assert config.max_new_videos_per_pass == 1
        assert config.caption_batch_size == 1
        assert config.caption_fetch_delay == 0.0

    @pytest.mark.fast
    def test_very_large_source_spacing(self):
        """Very large source_spacing_seconds is preserved."""
        config = IterativeMatchingConfig(source_spacing_seconds=1_000_000.0)
        assert config.source_spacing_seconds == 1_000_000.0

    @pytest.mark.fast
    def test_very_large_search_results(self):
        """Very large search_results_per_gap is preserved."""
        config = IterativeMatchingConfig(search_results_per_gap=10_000)
        assert config.search_results_per_gap == 10_000


# ─── DownloadConfig nested dict coercion (Rule 2) ────────────────────────


class TestDownloadConfigNestedDictCoercion:
    """Test DownloadConfig post_init converts nested dicts to dataclasses."""

    @pytest.mark.fast
    def test_cookie_rotation_from_dict(self):
        """CookieRotationConfig is coerced from dict."""
        config = DownloadConfig(
            cookie_rotation={'enabled': True, 'rotation_strategy': 'round_robin', 'rotate_on_errors': ['429']}
        )
        assert isinstance(config.cookie_rotation, CookieRotationConfig)
        assert config.cookie_rotation.enabled is True
        assert config.cookie_rotation.rotation_strategy == 'round_robin'

    @pytest.mark.fast
    def test_rate_limit_from_dict(self):
        """RateLimitConfig is coerced from dict."""
        config = DownloadConfig(
            rate_limit={'initial_backoff_seconds': 10.0, 'backoff_multiplier': 3.0}
        )
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert config.rate_limit.initial_backoff_seconds == 10.0
        assert config.rate_limit.backoff_multiplier == 3.0

    @pytest.mark.fast
    def test_vpn_from_dict(self):
        """VPNConfig is coerced from dict."""
        config = DownloadConfig(
            vpn={'enabled': True, 'switch_command': 'nordvpn connect'}
        )
        assert isinstance(config.vpn, VPNConfig)
        assert config.vpn.enabled is True
        assert config.vpn.switch_command == 'nordvpn connect'

    @pytest.mark.fast
    def test_mullvad_from_dict(self):
        """MullvadConfig is coerced from dict."""
        config = DownloadConfig(
            mullvad={'enabled': True, 'preferred_countries': ['us', 'gb']}
        )
        assert isinstance(config.mullvad, MullvadConfig)
        assert config.mullvad.enabled is True
        assert config.mullvad.preferred_countries == ['us', 'gb']

    @pytest.mark.fast
    def test_speed_tracking_from_dict(self):
        """SpeedTrackingConfig is coerced from dict."""
        config = DownloadConfig(
            speed_tracking={'enabled': False, 'window_size': 10}
        )
        assert isinstance(config.speed_tracking, SpeedTrackingConfig)
        assert config.speed_tracking.enabled is False
        assert config.speed_tracking.window_size == 10

    @pytest.mark.fast
    def test_circuit_breaker_from_dict(self):
        """CircuitBreakerConfig is coerced from dict."""
        config = DownloadConfig(
            circuit_breaker={'enabled': True, 'consecutive_failures_threshold': 10}
        )
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert config.circuit_breaker.consecutive_failures_threshold == 10

    @pytest.mark.fast
    def test_batch_retry_from_dict(self):
        """BatchRetryConfig is coerced from dict."""
        config = DownloadConfig(
            batch_retry={'enabled': False, 'max_passes': 5}
        )
        assert isinstance(config.batch_retry, BatchRetryConfig)
        assert config.batch_retry.enabled is False
        assert config.batch_retry.max_passes == 5

    @pytest.mark.fast
    def test_impersonation_from_dict(self):
        """ImpersonationConfig is coerced from dict."""
        config = DownloadConfig(
            impersonation={'enabled': False, 'detection_timeout': 20}
        )
        assert isinstance(config.impersonation, ImpersonationConfig)
        assert config.impersonation.enabled is False
        assert config.impersonation.detection_timeout == 20

    @pytest.mark.fast
    def test_extractor_args_from_dict(self):
        """ExtractorArgsConfig is coerced from dict."""
        config = DownloadConfig(
            extractor_args={'enabled': True, 'escalation_threshold': 5}
        )
        assert isinstance(config.extractor_args, ExtractorArgsConfig)
        assert config.extractor_args.escalation_threshold == 5

    @pytest.mark.fast
    def test_rate_limit_budget_from_dict(self):
        """RateLimitBudgetConfig is coerced from dict."""
        config = DownloadConfig(
            rate_limit_budget={'enabled': False, 'max_rotations': 20}
        )
        assert isinstance(config.rate_limit_budget, RateLimitBudgetConfig)
        assert config.rate_limit_budget.max_rotations == 20

    @pytest.mark.fast
    def test_all_nested_configs_from_dicts(self):
        """All 15 nested configs are coerced from dicts in a single construction."""
        config = DownloadConfig(
            llm_title_filter={'enabled': False},
            audio_first={'enabled': True},
            caption_first={'enabled': True},
            zero_download_remix={'enabled': False},
            speech_screening={'enabled': True},
            cookie_rotation={'enabled': True, 'rotate_on_errors': ['403']},
            rate_limit={'initial_backoff_seconds': 2.0},
            vpn={'enabled': False},
            mullvad={'enabled': False},
            speed_tracking={'enabled': True},
            circuit_breaker={'enabled': True},
            batch_retry={'enabled': True},
            impersonation={'enabled': True},
            extractor_args={'enabled': True},
            rate_limit_budget={'enabled': True},
        )
        assert isinstance(config.llm_title_filter, LLMTitleFilterConfig)
        assert isinstance(config.audio_first, AudioFirstConfig)
        assert isinstance(config.caption_first, CaptionFirstConfig)
        assert isinstance(config.zero_download_remix, ZeroDownloadRemixConfig)
        assert isinstance(config.speech_screening, SpeechScreeningConfig)
        assert isinstance(config.cookie_rotation, CookieRotationConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.vpn, VPNConfig)
        assert isinstance(config.mullvad, MullvadConfig)
        assert isinstance(config.speed_tracking, SpeedTrackingConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.batch_retry, BatchRetryConfig)
        assert isinstance(config.impersonation, ImpersonationConfig)
        assert isinstance(config.extractor_args, ExtractorArgsConfig)
        assert isinstance(config.rate_limit_budget, RateLimitBudgetConfig)

    @pytest.mark.fast
    def test_deeply_nested_caption_first_with_circuit_breaker_dict(self):
        """CaptionFirstConfig coerces its own nested circuit_breaker from dict."""
        config = CaptionFirstConfig(
            circuit_breaker={'enabled': False, 'threshold': 20, 'pause_seconds': 60.0}
        )
        assert isinstance(config.circuit_breaker, CaptionCircuitBreakerConfig)
        assert config.circuit_breaker.enabled is False
        assert config.circuit_breaker.threshold == 20

    @pytest.mark.fast
    def test_deeply_nested_caption_first_with_retry_budget_dict(self):
        """CaptionFirstConfig coerces its own nested retry_budget from dict."""
        config = CaptionFirstConfig(
            retry_budget={'enabled': True, 'max_attempts': 200, 'attempts_per_video': 2.0}
        )
        assert isinstance(config.retry_budget, CaptionRetryBudgetConfig)
        assert config.retry_budget.max_attempts == 200

    @pytest.mark.fast
    def test_download_config_with_deeply_nested_dicts(self):
        """DownloadConfig with caption_first as dict containing nested dicts."""
        config = DownloadConfig(
            caption_first={
                'enabled': True,
                'circuit_breaker': {'enabled': True, 'threshold': 15},
                'retry_budget': {'enabled': True, 'max_attempts': 50, 'attempts_per_video': 1.5},
            }
        )
        assert isinstance(config.caption_first, CaptionFirstConfig)
        assert isinstance(config.caption_first.circuit_breaker, CaptionCircuitBreakerConfig)
        assert config.caption_first.circuit_breaker.threshold == 15
        assert isinstance(config.caption_first.retry_budget, CaptionRetryBudgetConfig)
        assert config.caption_first.retry_budget.max_attempts == 50

    @pytest.mark.fast
    def test_empty_dict_creates_defaults(self):
        """Empty dict creates config with all defaults."""
        config = DownloadConfig(mullvad={})
        assert isinstance(config.mullvad, MullvadConfig)
        assert config.mullvad.enabled is False
        assert config.mullvad.rotation_strategy == 'random'


# ─── MatchingConfig post_init clamping ────────────────────────────────────


class TestMatchingConfigPostInitClamping:
    """Test MatchingConfig post_init sets nested defaults and coerces dicts."""

    @pytest.mark.fast
    def test_location_matching_none_gets_default(self):
        """location_matching=None creates a default LocationMatchingConfig."""
        config = MatchingConfig(location_matching=None)
        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert config.location_matching.enabled is True

    @pytest.mark.fast
    def test_chapter_detection_none_gets_default(self):
        """chapter_detection=None creates a default ChapterDetectionConfig."""
        config = MatchingConfig(chapter_detection=None)
        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.chapter_detection.enabled is True

    @pytest.mark.fast
    def test_location_matching_from_dict(self):
        """location_matching is coerced from dict."""
        config = MatchingConfig(
            location_matching={'enabled': False, 'geographic_penalty': 0.6}
        )
        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert config.location_matching.enabled is False
        assert config.location_matching.geographic_penalty == 0.6

    @pytest.mark.fast
    def test_chapter_detection_from_dict(self):
        """chapter_detection is coerced from dict."""
        config = MatchingConfig(
            chapter_detection={'enabled': False, 'max_chapters': 10}
        )
        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.chapter_detection.enabled is False
        assert config.chapter_detection.max_chapters == 10

    @pytest.mark.fast
    def test_location_matching_already_dataclass(self):
        """location_matching as dataclass instance is preserved."""
        loc = LocationMatchingConfig(enabled=False)
        config = MatchingConfig(location_matching=loc)
        assert config.location_matching is loc

    @pytest.mark.fast
    def test_chapter_detection_already_dataclass(self):
        """chapter_detection as dataclass instance is preserved."""
        ch = ChapterDetectionConfig(enabled=False)
        config = MatchingConfig(chapter_detection=ch)
        assert config.chapter_detection is ch

    @pytest.mark.fast
    def test_empty_dict_location_matching(self):
        """Empty dict for location_matching creates defaults."""
        config = MatchingConfig(location_matching={})
        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert config.location_matching.enabled is True
        assert config.location_matching.hard_filter_level == "city"

    @pytest.mark.fast
    def test_empty_dict_chapter_detection(self):
        """Empty dict for chapter_detection creates defaults."""
        config = MatchingConfig(chapter_detection={})
        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.chapter_detection.enabled is True
        assert config.chapter_detection.max_chapters == 20

    @pytest.mark.fast
    def test_matching_config_float_fields_accept_boundary_values(self):
        """MatchingConfig float fields accept extreme values without clamping."""
        config = MatchingConfig(
            min_confidence=0.0,
            high_confidence_threshold=1.0,
            low_confidence_threshold=0.0,
            reuse_penalty=0.0,
            keyword_boost=0.0,
            entity_boost=0.0,
        )
        assert config.min_confidence == 0.0
        assert config.high_confidence_threshold == 1.0
        assert config.low_confidence_threshold == 0.0


# ─── CaptionRetryBudgetConfig validation ─────────────────────────────────


class TestCaptionRetryBudgetConfigValidation:
    """Test CaptionRetryBudgetConfig post_init validation raises on bad values."""

    @pytest.mark.fast
    def test_max_attempts_below_minimum_raises(self):
        """max_attempts < 10 (and != 0) raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 10"):
            CaptionRetryBudgetConfig(max_attempts=5)

    @pytest.mark.fast
    def test_max_attempts_one_raises(self):
        """max_attempts of 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 10"):
            CaptionRetryBudgetConfig(max_attempts=1)

    @pytest.mark.fast
    def test_max_attempts_nine_raises(self):
        """max_attempts of 9 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 10"):
            CaptionRetryBudgetConfig(max_attempts=9)

    @pytest.mark.fast
    def test_max_attempts_zero_unlimited_ok(self):
        """max_attempts of 0 (unlimited) is valid."""
        config = CaptionRetryBudgetConfig(max_attempts=0)
        assert config.max_attempts == 0

    @pytest.mark.fast
    def test_max_attempts_ten_ok(self):
        """max_attempts of 10 (minimum valid) is accepted."""
        config = CaptionRetryBudgetConfig(max_attempts=10)
        assert config.max_attempts == 10

    @pytest.mark.fast
    def test_max_attempts_negative_raises(self):
        """Negative max_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 10"):
            CaptionRetryBudgetConfig(max_attempts=-1)

    @pytest.mark.fast
    def test_attempts_per_video_below_one_raises(self):
        """attempts_per_video < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="attempts_per_video must be >= 1.0"):
            CaptionRetryBudgetConfig(attempts_per_video=0.5)

    @pytest.mark.fast
    def test_attempts_per_video_zero_raises(self):
        """attempts_per_video of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="attempts_per_video must be >= 1.0"):
            CaptionRetryBudgetConfig(attempts_per_video=0.0)

    @pytest.mark.fast
    def test_attempts_per_video_one_ok(self):
        """attempts_per_video of 1.0 (minimum valid) is accepted."""
        config = CaptionRetryBudgetConfig(attempts_per_video=1.0)
        assert config.attempts_per_video == 1.0

    @pytest.mark.fast
    def test_max_backoff_below_minimum_raises(self):
        """max_backoff_time_seconds < 30.0 (and != 0) raises ValueError."""
        with pytest.raises(ValueError, match="max_backoff_time_seconds must be >= 30.0"):
            CaptionRetryBudgetConfig(max_backoff_time_seconds=10.0)

    @pytest.mark.fast
    def test_max_backoff_zero_unlimited_ok(self):
        """max_backoff_time_seconds of 0.0 (unlimited) is valid."""
        config = CaptionRetryBudgetConfig(max_backoff_time_seconds=0.0)
        assert config.max_backoff_time_seconds == 0.0

    @pytest.mark.fast
    def test_max_backoff_thirty_ok(self):
        """max_backoff_time_seconds of 30.0 (minimum valid) is accepted."""
        config = CaptionRetryBudgetConfig(max_backoff_time_seconds=30.0)
        assert config.max_backoff_time_seconds == 30.0

    @pytest.mark.fast
    def test_valid_config_all_defaults(self):
        """Default CaptionRetryBudgetConfig passes validation."""
        config = CaptionRetryBudgetConfig()
        assert config.max_attempts == 100
        assert config.attempts_per_video == 2.0
        assert config.max_backoff_time_seconds == 300.0


# ─── CookieRotationConfig validation ─────────────────────────────────────


class TestCookieRotationConfigValidation:
    """Test CookieRotationConfig post_init validation."""

    @pytest.mark.fast
    def test_enabled_with_empty_errors_raises(self):
        """Enabled cookie rotation with empty rotate_on_errors raises ValueError."""
        with pytest.raises(ValueError, match="rotate_on_errors must be a non-empty list"):
            CookieRotationConfig(enabled=True, rotate_on_errors=[])

    @pytest.mark.fast
    def test_disabled_with_empty_errors_ok(self):
        """Disabled cookie rotation with empty rotate_on_errors is valid."""
        config = CookieRotationConfig(enabled=False, rotate_on_errors=[])
        assert config.enabled is False


# ─── Missing optional fields default correctly ────────────────────────────


class TestMissingOptionalFieldDefaults:
    """Test config sections handle missing optional fields with correct defaults."""

    @pytest.mark.fast
    def test_iterative_matching_defaults(self):
        """IterativeMatchingConfig with no args has correct defaults."""
        config = IterativeMatchingConfig()
        assert config.enabled is True
        assert config.target_confidence == 0.90
        assert config.source_spacing_seconds == 300.0
        assert config.max_iterations == 5
        assert config.min_gap_percentage == 0.05
        assert config.search_results_per_gap == 10
        assert config.max_new_videos_per_pass == 50
        assert config.caption_batch_size == 10
        assert config.caption_fetch_delay == 0.5
        assert config.caption_timeout == 30
        assert len(config.gap_pattern_categories) == 5

    @pytest.mark.fast
    def test_download_config_defaults(self):
        """DownloadConfig with no args has correct nested defaults."""
        config = DownloadConfig()
        assert isinstance(config.llm_title_filter, LLMTitleFilterConfig)
        assert isinstance(config.audio_first, AudioFirstConfig)
        assert isinstance(config.caption_first, CaptionFirstConfig)
        assert isinstance(config.zero_download_remix, ZeroDownloadRemixConfig)
        assert isinstance(config.speech_screening, SpeechScreeningConfig)
        assert isinstance(config.cookie_rotation, CookieRotationConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.vpn, VPNConfig)
        assert isinstance(config.mullvad, MullvadConfig)
        assert isinstance(config.speed_tracking, SpeedTrackingConfig)
        assert isinstance(config.circuit_breaker, CircuitBreakerConfig)
        assert isinstance(config.batch_retry, BatchRetryConfig)
        assert isinstance(config.impersonation, ImpersonationConfig)
        assert isinstance(config.extractor_args, ExtractorArgsConfig)
        assert isinstance(config.rate_limit_budget, RateLimitBudgetConfig)
        assert config.max_retries == 3
        assert config.quality == "1080p"

    @pytest.mark.fast
    def test_matching_config_defaults(self):
        """MatchingConfig with no args has correct nested defaults."""
        config = MatchingConfig()
        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.min_confidence == 0.7
        assert config.high_confidence_threshold == 0.85
        assert config.embedding_candidates == 50
        assert config.llm_rerank_candidates == 5
        assert config.caption_quality_weights is None
        assert config.multimodal_weights is None

    @pytest.mark.fast
    def test_caption_first_config_defaults(self):
        """CaptionFirstConfig with no args has correct nested defaults."""
        config = CaptionFirstConfig()
        assert isinstance(config.circuit_breaker, CaptionCircuitBreakerConfig)
        assert isinstance(config.retry_budget, CaptionRetryBudgetConfig)
        assert config.preferred_language == "en"
        assert config.fallback_languages == []
        assert config.preferred_formats == ["json3", "vtt", "srt"]
        assert config.retry_budgets == {
            "network": 3, "timeout": 2, "parse": 1,
            "unavailable": 0, "rate_limit": 2,
        }

    @pytest.mark.fast
    def test_circuit_breaker_config_defaults(self):
        """CircuitBreakerConfig with no args has correct defaults."""
        config = CircuitBreakerConfig()
        assert config.enabled is True
        assert config.consecutive_failures_threshold == 5
        assert config.pause_seconds == 60.0
        assert config.max_pause_seconds == 300.0
        assert config.block_download_retries is True

    @pytest.mark.fast
    def test_mullvad_config_defaults(self):
        """MullvadConfig with no args has correct defaults."""
        config = MullvadConfig()
        assert config.enabled is False
        assert config.rotation_strategy == 'random'
        assert config.max_rotations_per_session == 5
        assert config.verification_timeout == 10
        assert 'us' in config.preferred_countries

    @pytest.mark.fast
    def test_impersonation_config_defaults(self):
        """ImpersonationConfig with no args has correct defaults."""
        config = ImpersonationConfig()
        assert config.enabled is True
        assert config.preferred_targets == []
        assert config.detect_at_startup is True
        assert config.detection_timeout == 10

    @pytest.mark.fast
    def test_batch_retry_config_defaults(self):
        """BatchRetryConfig with no args has correct defaults."""
        config = BatchRetryConfig()
        assert config.enabled is True
        assert config.delay_seconds == 120.0
        assert config.max_passes == 2
        assert config.respect_circuit_breaker is True
        assert config.wait_for_cookie_cooldown is True
        assert config.max_combined_wait_seconds == 300.0

    @pytest.mark.fast
    def test_location_matching_config_defaults(self):
        """LocationMatchingConfig with no args has correct defaults."""
        config = LocationMatchingConfig()
        assert config.enabled is True
        assert config.hard_filter_level == "city"
        assert config.geographic_penalty == 0.4
        assert config.hierarchy_bonus == 0.15
        assert config.landmark_bonus == 0.2

    @pytest.mark.fast
    def test_chapter_detection_config_defaults(self):
        """ChapterDetectionConfig with no args has correct defaults."""
        config = ChapterDetectionConfig()
        assert config.enabled is True
        assert config.default_strategy == 'topic'
        assert config.max_chunk_chars == 6000
        assert config.min_chapter_confidence == 0.5
        assert config.min_chapter_segments == 3
        assert config.max_chapters == 20

    @pytest.mark.fast
    def test_partial_dict_fills_remaining_defaults(self):
        """Partial dict coercion fills unspecified fields with defaults."""
        config = DownloadConfig(mullvad={'enabled': True})
        assert config.mullvad.enabled is True
        # Unspecified fields should have defaults
        assert config.mullvad.rotation_strategy == 'random'
        assert config.mullvad.max_rotations_per_session == 5
        assert config.mullvad.verification_timeout == 10
        assert config.mullvad.rotation_delay_seconds == 5
        assert len(config.mullvad.preferred_countries) > 0
