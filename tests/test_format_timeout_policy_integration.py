"""Tests for FormatTimeoutPolicy integration with CaptionFetcher (US-59-005).

Verifies that:
- json3 gets a longer timeout (45s) than vtt/srt (25s)
- Progressive fallback reduces timeout by 0.8x per fallback_level
- CaptionFetcher._fetch_subtitle_with_format uses FormatTimeoutPolicy
- CaptionFetcher.__init__ instantiates FormatTimeoutPolicy with config overrides
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.caption_timeout_manager import FormatTimeoutPolicy


# ============================================================================
# FormatTimeoutPolicy Unit Tests
# ============================================================================


@pytest.mark.fast
class TestFormatTimeoutPolicyDefaults:
    """Test FormatTimeoutPolicy default timeout values."""

    def test_json3_gets_longer_timeout_than_vtt_srt(self):
        """json3 should get 45s timeout, vtt/srt should get 25s."""
        policy = FormatTimeoutPolicy()

        json3_timeout = policy.get_timeout('json3', fallback_level=0)
        vtt_timeout = policy.get_timeout('vtt', fallback_level=0)
        srt_timeout = policy.get_timeout('srt', fallback_level=0)

        assert json3_timeout == 45.0, f"json3 should be 45s, got {json3_timeout}"
        assert vtt_timeout == 25.0, f"vtt should be 25s, got {vtt_timeout}"
        assert srt_timeout == 25.0, f"srt should be 25s, got {srt_timeout}"
        assert json3_timeout > vtt_timeout, "json3 should have longer timeout than vtt"
        assert json3_timeout > srt_timeout, "json3 should have longer timeout than srt"

    def test_srv3_timeout(self):
        """srv3 should get 30s timeout."""
        policy = FormatTimeoutPolicy()
        assert policy.get_timeout('srv3', fallback_level=0) == 30.0

    def test_unknown_format_gets_default_30s(self):
        """Unknown formats fall back to 30s."""
        policy = FormatTimeoutPolicy()
        assert policy.get_timeout('unknown_fmt', fallback_level=0) == 30.0


@pytest.mark.fast
class TestFormatTimeoutPolicyFallback:
    """Test progressive fallback reduction (0.8x per level)."""

    def test_third_format_gets_reduced_timeout(self):
        """Third format attempt (fallback_level=2) should get 0.8^2 = 0.64x timeout."""
        policy = FormatTimeoutPolicy()

        # json3 at level 0 = 45.0
        level0 = policy.get_timeout('json3', fallback_level=0)
        assert level0 == 45.0

        # json3 at level 1 = 45.0 * 0.8 = 36.0
        level1 = policy.get_timeout('json3', fallback_level=1)
        assert level1 == pytest.approx(36.0)

        # json3 at level 2 = 45.0 * 0.8^2 = 28.8
        level2 = policy.get_timeout('json3', fallback_level=2)
        assert level2 == pytest.approx(28.8)

    def test_vtt_fallback_reduction(self):
        """vtt at fallback_level=2 should be 25.0 * 0.8^2 = 16.0."""
        policy = FormatTimeoutPolicy()

        level0 = policy.get_timeout('vtt', fallback_level=0)
        level2 = policy.get_timeout('vtt', fallback_level=2)

        assert level0 == 25.0
        assert level2 == pytest.approx(25.0 * 0.64)  # 0.8^2 = 0.64

    def test_no_reduction_at_level_zero(self):
        """fallback_level=0 should return base timeout unmodified."""
        policy = FormatTimeoutPolicy()

        for fmt in ['json3', 'srv3', 'vtt', 'srt']:
            base = policy.timeouts[fmt]
            actual = policy.get_timeout(fmt, fallback_level=0)
            assert actual == base, f"{fmt}: expected {base}, got {actual}"

    def test_progressive_fallback_disabled(self):
        """When progressive_fallback=False, no reduction applied."""
        policy = FormatTimeoutPolicy(progressive_fallback=False)

        level0 = policy.get_timeout('json3', fallback_level=0)
        level3 = policy.get_timeout('json3', fallback_level=3)

        assert level0 == level3 == 45.0

    def test_custom_fallback_reduction(self):
        """Custom fallback_reduction factor is applied correctly."""
        policy = FormatTimeoutPolicy(fallback_reduction=0.5)

        # json3 at level 1 = 45.0 * 0.5 = 22.5
        assert policy.get_timeout('json3', fallback_level=1) == pytest.approx(22.5)
        # json3 at level 2 = 45.0 * 0.25 = 11.25
        assert policy.get_timeout('json3', fallback_level=2) == pytest.approx(11.25)


# ============================================================================
# CaptionFetcher Integration Tests
# ============================================================================


@pytest.mark.fast
class TestCaptionFetcherFormatTimeout:
    """Test FormatTimeoutPolicy integration in CaptionFetcher."""

    def test_init_creates_default_format_timeout_policy(self):
        """CaptionFetcher.__init__ creates FormatTimeoutPolicy with defaults."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher(config=None)

        assert hasattr(fetcher, '_format_timeout_policy')
        assert isinstance(fetcher._format_timeout_policy, FormatTimeoutPolicy)
        # Verify default timeouts are present
        assert fetcher._format_timeout_policy.get_timeout('json3') == 45.0
        assert fetcher._format_timeout_policy.get_timeout('vtt') == 25.0

    def test_init_with_config_format_timeouts(self):
        """CaptionFetcher uses format_timeouts from config when provided."""
        from src.caption_fetcher import CaptionFetcher

        # Mock config with format_timeouts
        mock_caption_first = MagicMock()
        mock_caption_first.timeout = 30
        mock_caption_first.max_retries = 3
        mock_caption_first.retry_delay = 2.0
        mock_caption_first.preferred_formats = ['json3', 'vtt', 'srt']
        mock_caption_first.adaptive_format_order = True
        mock_caption_first.format_timeouts = {
            'json3': 60.0,
            'vtt': 15.0,
            'srt': 15.0,
        }

        mock_config = MagicMock()
        mock_config.download.caption_first = mock_caption_first

        fetcher = CaptionFetcher(config=mock_config)

        assert fetcher._format_timeout_policy.get_timeout('json3') == 60.0
        assert fetcher._format_timeout_policy.get_timeout('vtt') == 15.0

    @patch('src.caption_fetcher.subprocess.run')
    def test_fetch_subtitle_with_format_uses_policy_timeout(self, mock_run):
        """_fetch_subtitle_with_format passes format-specific timeout to subprocess."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher(config=None)

        # Mock subprocess to return non-zero (simulating failure) to keep test simple
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr='no subtitles',
            stdout=''
        )

        from src.caption.exceptions import CaptionUnavailableError
        with pytest.raises(CaptionUnavailableError):
            fetcher._fetch_subtitle_with_format(
                video_url='https://youtube.com/watch?v=test',
                video_id='test',
                temp_dir=Path('/tmp/test'),
                language='en',
                auto_generated=True,
                subtitle_format='json3',
                fallback_level=0
            )

        # Verify subprocess.run was called with json3's timeout (45s)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs['timeout'] == 45.0, (
            f"json3 at level 0 should use 45s timeout, got {call_kwargs.kwargs['timeout']}"
        )

    @patch('src.caption_fetcher.subprocess.run')
    def test_fetch_subtitle_with_format_applies_fallback_reduction(self, mock_run):
        """_fetch_subtitle_with_format reduces timeout at higher fallback levels."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher(config=None)

        mock_run.return_value = MagicMock(
            returncode=1,
            stderr='no subtitles',
            stdout=''
        )

        from src.caption.exceptions import CaptionUnavailableError

        # Call with fallback_level=2 for vtt → 25.0 * 0.8^2 = 16.0
        with pytest.raises(CaptionUnavailableError):
            fetcher._fetch_subtitle_with_format(
                video_url='https://youtube.com/watch?v=test',
                video_id='test',
                temp_dir=Path('/tmp/test'),
                language='en',
                auto_generated=True,
                subtitle_format='vtt',
                fallback_level=2
            )

        call_kwargs = mock_run.call_args
        expected_timeout = 25.0 * (0.8 ** 2)  # 16.0
        assert call_kwargs.kwargs['timeout'] == pytest.approx(expected_timeout), (
            f"vtt at fallback_level=2 should use {expected_timeout}s, "
            f"got {call_kwargs.kwargs['timeout']}"
        )
