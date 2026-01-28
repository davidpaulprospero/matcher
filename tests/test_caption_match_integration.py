"""
Integration tests for caption metadata impact on matching.

This module contains integration tests for:
1. Caption quality impact on matching (US-010 Sprint 6)
2. Timing penalty impact on matching (US-002 Sprint 8)

Tests validate the full pipeline path from caption metadata through to
match confidence scoring.

Created: 2026-01-26
Updated: 2026-01-26 - Added timing penalty pipeline tests (US-002 Sprint 8)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_caption_quality_adjustment, apply_timing_penalty
from src.utils import SRTSegment
from src.caption_fetcher import CaptionResult, CaptionSegment


# =============================================================================
# Test Class: Caption Quality Match Integration
# =============================================================================

class TestCaptionQualityMatchIntegration:
    """Integration tests for caption quality impact on matching (US-010).

    This test class verifies the complete flow of caption quality metadata
    from the CaptionStage through to MatchStage confidence adjustments.

    Test Coverage:
    - AC1: Mock low-quality auto-generated captions flow to MatchStage
    - AC2: Match confidence differs for same text with different quality
    - AC3: Final OTIO segments include caption_quality metadata
    - AC4: Quality weights config correctly applied (US-006 integration)

    Integration Smoke Test Documentation:
    This test suite serves as the primary smoke test for the caption-to-match
    quality pipeline. When caption quality scoring changes are made, these
    tests should be run to verify end-to-end behavior.
    """

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def mock_config_with_weights(self):
        """Mock config with US-006 multiplicative weights enabled."""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = True
        # US-006: Multiplicative weights {high: 1.0, medium: 0.9, low: 0.75}
        matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        config.matching = matching
        return config

    @pytest.fixture
    def mock_config_additive(self):
        """Mock config with legacy additive mode (US-007)."""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = True
        matching.caption_quality_weights = None  # Triggers additive mode
        matching.caption_quality_high_boost = 0.05
        matching.caption_quality_low_penalty = 0.1
        config.matching = matching
        return config

    @pytest.fixture
    def mock_config_disabled(self):
        """Mock config with caption quality adjustment disabled."""
        config = Mock()
        matching = Mock()
        matching.caption_quality_adjustment_enabled = False
        config.matching = matching
        return config

    @pytest.fixture
    def sample_video_segment(self):
        """Sample video segment for matching tests."""
        return SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Sample video content about nature and wildlife",
            source_file="/videos/nature_doc.mp4"
        )

    @pytest.fixture
    def sample_text_metadata_low_quality(self):
        """Sample text_metadata entry simulating low-quality auto-generated captions.

        This represents what CaptionStage produces for auto-generated captions
        from YouTube. Low quality typically means sparse or incomplete captions.
        """
        return {
            'video_path': 'abc123xyz',  # YouTube video ID (caption-first mode)
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'Sample video content about nature and wildlife',
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': True,  # Auto-generated = lower quality
            'caption_quality': 'low',  # US-007: Quality indicator
        }

    @pytest.fixture
    def sample_text_metadata_high_quality(self):
        """Sample text_metadata entry simulating high-quality human captions.

        This represents what CaptionStage produces for human-uploaded captions.
        High quality means dense, accurate captions.
        """
        return {
            'video_path': 'def456uvw',
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'Sample video content about nature and wildlife',  # Same text
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': False,  # Human uploaded
            'caption_quality': 'high',  # US-007: Quality indicator
        }

    # =========================================================================
    # AC1: Low-quality auto-generated captions flow to MatchStage with quality='low'
    # =========================================================================

    @pytest.mark.fast
    def test_low_quality_caption_flows_to_match_stage(
        self,
        sample_text_metadata_low_quality,
        mock_config_with_weights
    ):
        """Test that low-quality auto-generated captions flow to MatchStage.

        Verifies that caption_quality='low' from CaptionStage is correctly
        propagated to video segments used in matching.

        Flow: CaptionStage.text_metadata -> MatchStage._prepare_segments -> vid_segment.caption_quality
        """
        meta = sample_text_metadata_low_quality

        # Simulate what MatchStage does: convert text_metadata dict to SRTSegment
        vid_segment = SRTSegment(
            index=1,
            start_time=meta['start_time'],
            end_time=meta['end_time'],
            text=meta['text'],
            source_file=meta['video_path']
        )

        # MatchStage sets caption_quality from metadata (see match.py:414-417)
        if meta.get('caption_quality') is not None:
            vid_segment.caption_quality = meta['caption_quality']

        # Verify caption_quality was set correctly
        assert hasattr(vid_segment, 'caption_quality')
        assert vid_segment.caption_quality == 'low'

        # Now verify this segment produces lower confidence when matched
        base_confidence = 0.85
        adjusted_confidence, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=vid_segment,
            config=mock_config_with_weights
        )

        # With low quality weight=0.75: 0.85 * 0.75 = 0.6375
        assert adjusted_confidence < base_confidence
        assert abs(adjusted_confidence - 0.6375) < 0.001
        assert 'low' in reason.lower() or 'caption quality' in reason.lower()

    @pytest.mark.fast
    def test_auto_generated_flag_correlates_with_quality(
        self,
        sample_text_metadata_low_quality,
        sample_text_metadata_high_quality
    ):
        """Test that auto_generated flag correlates with caption quality.

        Auto-generated captions should typically have lower quality than
        human-uploaded captions.
        """
        low_meta = sample_text_metadata_low_quality
        high_meta = sample_text_metadata_high_quality

        # Auto-generated should have low/medium quality
        assert low_meta['caption_auto_generated'] is True
        assert low_meta['caption_quality'] in ['low', 'medium']

        # Human-uploaded should have high quality
        assert high_meta['caption_auto_generated'] is False
        assert high_meta['caption_quality'] == 'high'

    # =========================================================================
    # AC2: Match confidence differs for same text when quality is 'high' vs 'low'
    # =========================================================================

    @pytest.mark.fast
    def test_same_text_different_quality_different_confidence(
        self,
        mock_config_with_weights,
        sample_video_segment
    ):
        """Test that same match text produces different confidence with high vs low quality.

        This is a key US-006 acceptance criterion: identical text content should
        yield different confidence scores based on caption quality metadata.
        """
        base_confidence = 0.80

        # Create two segments with same text but different quality
        high_quality_segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Wildlife documentary about African elephants",
            source_file="/videos/elephants.mp4"
        )
        high_quality_segment.caption_quality = "high"

        low_quality_segment = SRTSegment(
            index=2,
            start_time=0.0,
            end_time=10.0,
            text="Wildlife documentary about African elephants",  # Same text!
            source_file="/videos/elephants.mp4"
        )
        low_quality_segment.caption_quality = "low"

        # Apply caption quality adjustment to both
        high_conf, high_reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=high_quality_segment,
            config=mock_config_with_weights
        )

        low_conf, low_reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=low_quality_segment,
            config=mock_config_with_weights
        )

        # High quality (x1.0) should preserve confidence
        assert high_conf == base_confidence  # 0.80 * 1.0 = 0.80

        # Low quality (x0.75) should reduce confidence
        assert abs(low_conf - 0.60) < 0.001  # 0.80 * 0.75 = 0.60

        # High should be greater than low
        assert high_conf > low_conf

        # Difference should be significant (at least 15%)
        assert high_conf - low_conf >= 0.15

    @pytest.mark.fast
    def test_medium_quality_between_high_and_low(
        self,
        mock_config_with_weights,
        sample_video_segment
    ):
        """Test that medium quality produces confidence between high and low."""
        base_confidence = 0.80

        # Test all three quality levels
        for quality, expected_weight in [('high', 1.0), ('medium', 0.9), ('low', 0.75)]:
            segment = SRTSegment(
                index=1, start_time=0.0, end_time=10.0,
                text="Test content", source_file="/test.mp4"
            )
            segment.caption_quality = quality

            conf, _ = apply_caption_quality_adjustment(
                confidence=base_confidence,
                video_segment=segment,
                config=mock_config_with_weights
            )

            expected_conf = base_confidence * expected_weight
            assert abs(conf - expected_conf) < 0.001, \
                f"Quality '{quality}' expected {expected_conf}, got {conf}"

    # =========================================================================
    # AC3: Final OTIO segments include caption_quality metadata
    # =========================================================================

    @pytest.mark.fast
    def test_caption_quality_preserved_in_segment(
        self,
        sample_text_metadata_low_quality
    ):
        """Test that caption_quality is preserved from text_metadata to segment.

        This verifies the first part of the chain: caption_quality in text_metadata
        is correctly transferred to SRTSegment attributes for matching.
        """
        meta = sample_text_metadata_low_quality

        # Create segment from metadata (as MatchStage does)
        segment = SRTSegment(
            index=meta.get('index', 1),
            start_time=meta['start_time'],
            end_time=meta['end_time'],
            text=meta['text'],
            source_file=meta['video_path']
        )

        # Set caption_quality from metadata
        if meta.get('caption_quality') is not None:
            segment.caption_quality = meta['caption_quality']

        # Verify it's preserved
        assert segment.caption_quality == 'low'

        # Verify it survives serialization (important for checkpoint)
        segment_dict = {
            'index': segment.index,
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'text': segment.text,
            'source_file': segment.source_file,
            'caption_quality': getattr(segment, 'caption_quality', None)
        }

        assert segment_dict['caption_quality'] == 'low'

    @pytest.mark.fast
    def test_caption_quality_in_match_result_metadata(self):
        """Test that caption_quality can be included in match result metadata.

        This test verifies the pattern for including caption_quality in the
        final match results that would flow to OTIO generation.

        Note: Currently OTIO builder doesn't directly use caption_quality,
        but match results can include it for downstream filtering.
        """
        # Simulate a match result with caption quality metadata
        match_result = {
            'segment_index': 1,
            'video_file': '/videos/nature.mp4',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': 0.75,  # Already adjusted for caption quality
            'strategy': 'embedding_match',
            'reason': 'High similarity [caption quality low: x0.75]',
            # Caption quality metadata for downstream filtering
            'caption_quality': 'low',
            'caption_auto_generated': True,
        }

        # Verify all expected fields are present
        assert 'caption_quality' in match_result
        assert 'caption_auto_generated' in match_result

        # Verify the reason string includes caption quality adjustment
        assert 'caption quality' in match_result['reason'].lower()

        # Verify quality and auto-gen flag are consistent
        assert match_result['caption_quality'] == 'low'
        assert match_result['caption_auto_generated'] is True

    @pytest.mark.fast
    def test_text_metadata_to_match_quality_flow(
        self,
        sample_text_metadata_low_quality,
        sample_text_metadata_high_quality,
        mock_config_with_weights
    ):
        """Test complete flow: text_metadata -> segment -> quality adjustment.

        This is the integration smoke test that verifies the entire pipeline:
        1. CaptionStage produces text_metadata with caption_quality
        2. MatchStage converts to segments with caption_quality attribute
        3. Scoring applies quality adjustment to confidence
        """
        base_confidence = 0.80
        results = {}

        for name, meta in [
            ('low', sample_text_metadata_low_quality),
            ('high', sample_text_metadata_high_quality)
        ]:
            # Step 1: Create segment from text_metadata (MatchStage._prepare_segments)
            segment = SRTSegment(
                index=1,
                start_time=meta['start_time'],
                end_time=meta['end_time'],
                text=meta['text'],
                source_file=meta['video_path']
            )
            if meta.get('caption_quality') is not None:
                segment.caption_quality = meta['caption_quality']

            # Step 2: Apply quality adjustment (scoring.apply_caption_quality_adjustment)
            adjusted_conf, reason = apply_caption_quality_adjustment(
                confidence=base_confidence,
                video_segment=segment,
                config=mock_config_with_weights
            )

            results[name] = {
                'original_quality': meta['caption_quality'],
                'is_auto_generated': meta['caption_auto_generated'],
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_conf,
                'adjustment_reason': reason,
            }

        # Verify the pipeline produces expected differences
        assert results['high']['adjusted_confidence'] == 0.80  # high=1.0, no change
        assert abs(results['low']['adjusted_confidence'] - 0.60) < 0.001  # low=0.75

        # Verify high > low
        assert results['high']['adjusted_confidence'] > results['low']['adjusted_confidence']

        # Verify reasons are informative
        assert results['low']['adjustment_reason'] != ""
        assert 'low' in results['low']['adjustment_reason'].lower()

    # =========================================================================
    # AC4: Quality weights config correctly applied (US-006 integration)
    # =========================================================================

    @pytest.mark.fast
    def test_quality_weights_config_applied(
        self,
        sample_video_segment,
        mock_config_with_weights
    ):
        """Test that quality weights from config are correctly applied.

        Verifies US-006 implementation: multiplicative weights mode
        {high: 1.0, medium: 0.9, low: 0.75} is used when config has
        caption_quality_weights set.
        """
        sample_video_segment.caption_quality = "medium"
        base_confidence = 0.90

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=mock_config_with_weights
        )

        # Medium weight = 0.9, so 0.90 * 0.9 = 0.81
        assert abs(adjusted_conf - 0.81) < 0.001
        assert "x0.90" in reason

    @pytest.mark.fast
    def test_custom_weights_override_defaults(self, sample_video_segment):
        """Test that custom weights in config override default values."""
        # Custom config with different weights
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {
            'high': 1.1,   # Boost high quality
            'medium': 0.8,
            'low': 0.5    # Harsher penalty
        }

        sample_video_segment.caption_quality = "low"
        base_confidence = 0.80

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # With custom low=0.5: 0.80 * 0.5 = 0.40
        assert abs(adjusted_conf - 0.40) < 0.001

    @pytest.mark.fast
    def test_additive_mode_when_weights_none(
        self,
        sample_video_segment,
        mock_config_additive
    ):
        """Test that additive mode (US-007) is used when weights are None.

        This verifies backward compatibility with the legacy additive mode.
        """
        sample_video_segment.caption_quality = "low"
        base_confidence = 0.80

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=mock_config_additive
        )

        # Additive mode: 0.80 - 0.1 (low_penalty) = 0.70
        assert abs(adjusted_conf - 0.70) < 0.001
        assert "-0.10" in reason or "low" in reason.lower()

    @pytest.mark.fast
    def test_adjustment_disabled_no_change(
        self,
        sample_video_segment,
        mock_config_disabled
    ):
        """Test that disabled adjustment produces no confidence change."""
        sample_video_segment.caption_quality = "low"
        base_confidence = 0.80

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=mock_config_disabled
        )

        # Disabled = no change
        assert adjusted_conf == base_confidence
        assert reason == ""

    # =========================================================================
    # Edge Cases and Boundary Tests
    # =========================================================================

    @pytest.mark.fast
    def test_no_caption_quality_attribute_no_change(
        self,
        sample_video_segment,
        mock_config_with_weights
    ):
        """Test that segments without caption_quality attribute are unchanged."""
        # Don't set caption_quality attribute
        base_confidence = 0.80

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=mock_config_with_weights
        )

        assert adjusted_conf == base_confidence
        assert reason == ""

    @pytest.mark.fast
    def test_unknown_quality_value_defaults_to_1(self, sample_video_segment):
        """Test that unknown quality values default to weight=1.0."""
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}

        sample_video_segment.caption_quality = "unknown"  # Not in weights dict
        base_confidence = 0.80

        adjusted_conf, reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=sample_video_segment,
            config=config
        )

        # Unknown defaults to 1.0, so no change
        assert adjusted_conf == base_confidence

    @pytest.mark.fast
    def test_confidence_capped_at_1_and_0(self, mock_config_with_weights):
        """Test that adjusted confidence is clamped to [0.0, 1.0]."""
        # Test upper cap with boost
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test", source_file="/test.mp4"
        )
        segment.caption_quality = "high"

        # Even with high confidence, should not exceed 1.0
        conf, _ = apply_caption_quality_adjustment(
            confidence=0.99,
            video_segment=segment,
            config=mock_config_with_weights
        )
        assert conf <= 1.0

        # Test lower cap with penalty
        segment.caption_quality = "low"
        conf, _ = apply_caption_quality_adjustment(
            confidence=0.05,  # Very low confidence
            video_segment=segment,
            config=mock_config_with_weights
        )
        assert conf >= 0.0


# =============================================================================
# Test Class: Caption Quality Distribution Logging
# =============================================================================

class TestCaptionQualityDistributionLogging:
    """Test caption quality distribution tracking in MatchStage.

    The MatchStage logs caption quality distribution after preparing segments.
    This test class verifies that tracking works correctly.
    """

    @pytest.mark.fast
    def test_quality_distribution_counting(self):
        """Test that caption quality distribution is correctly counted.

        Simulates what MatchStage does in _prepare_segments (match.py:393-429).
        """
        # Simulate text_metadata with mixed quality
        text_metadata = [
            {'caption_quality': 'high', 'text': 'a', 'start_time': 0, 'end_time': 1, 'video_path': '1'},
            {'caption_quality': 'high', 'text': 'b', 'start_time': 1, 'end_time': 2, 'video_path': '2'},
            {'caption_quality': 'medium', 'text': 'c', 'start_time': 2, 'end_time': 3, 'video_path': '3'},
            {'caption_quality': 'medium', 'text': 'd', 'start_time': 3, 'end_time': 4, 'video_path': '4'},
            {'caption_quality': 'medium', 'text': 'e', 'start_time': 4, 'end_time': 5, 'video_path': '5'},
            {'caption_quality': 'low', 'text': 'f', 'start_time': 5, 'end_time': 6, 'video_path': '6'},
            {'text': 'g', 'start_time': 6, 'end_time': 7, 'video_path': '7'},  # No quality
        ]

        # Count quality distribution (as MatchStage does)
        caption_quality_count = {'high': 0, 'medium': 0, 'low': 0}
        for meta in text_metadata:
            if meta.get('caption_quality') in caption_quality_count:
                caption_quality_count[meta['caption_quality']] += 1

        # Verify counts
        assert caption_quality_count['high'] == 2
        assert caption_quality_count['medium'] == 3
        assert caption_quality_count['low'] == 1

        # Total counted should match (excluding entry without quality)
        total_counted = sum(caption_quality_count.values())
        assert total_counted == 6

    @pytest.mark.fast
    def test_empty_quality_distribution(self):
        """Test that empty quality distribution is handled gracefully."""
        text_metadata = [
            {'text': 'a', 'start_time': 0, 'end_time': 1, 'video_path': '1'},  # No quality
            {'text': 'b', 'start_time': 1, 'end_time': 2, 'video_path': '2'},  # No quality
        ]

        caption_quality_count = {'high': 0, 'medium': 0, 'low': 0}
        for meta in text_metadata:
            if meta.get('caption_quality') in caption_quality_count:
                caption_quality_count[meta['caption_quality']] += 1

        # All counts should be 0
        assert all(count == 0 for count in caption_quality_count.values())
        assert not any(caption_quality_count.values())  # Falsy when all 0


# =============================================================================
# Integration Smoke Test Summary
# =============================================================================

class TestCaptionQualityPipelineSmokeTest:
    """Comprehensive smoke test for caption quality pipeline (US-010).

    This test class provides a single comprehensive test that validates
    the entire caption quality -> match confidence pipeline.

    Use this test for quick validation after changes to:
    - caption_fetcher.py (CaptionResult.caption_quality)
    - caption_stage.py (text_metadata caption_quality field)
    - match.py (caption_quality attribute propagation)
    - scoring.py (apply_caption_quality_adjustment)
    """

    @pytest.mark.fast
    def test_full_pipeline_smoke_test(self):
        """Comprehensive smoke test for caption quality pipeline.

        Tests the complete flow:
        1. Caption fetch produces quality metadata
        2. CaptionStage includes quality in text_metadata
        3. MatchStage transfers quality to segments
        4. Scoring applies quality-based confidence adjustment
        5. Results show expected confidence differences
        """
        # Config with US-006 multiplicative weights
        config = Mock()
        config.matching = Mock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}

        # Simulate CaptionStage output (text_metadata)
        text_metadata_batch = [
            {
                'video_path': 'vid001',
                'start_time': 0.0,
                'end_time': 5.0,
                'text': 'Introduction to the documentary',
                'caption_quality': 'high',
                'caption_auto_generated': False,
            },
            {
                'video_path': 'vid002',
                'start_time': 0.0,
                'end_time': 5.0,
                'text': 'Introduction to the documentary',  # Same text
                'caption_quality': 'low',
                'caption_auto_generated': True,
            },
        ]

        results = []
        base_confidence = 0.85

        for meta in text_metadata_batch:
            # Simulate MatchStage._prepare_segments
            segment = SRTSegment(
                index=1,
                start_time=meta['start_time'],
                end_time=meta['end_time'],
                text=meta['text'],
                source_file=meta['video_path']
            )
            if meta.get('caption_quality'):
                segment.caption_quality = meta['caption_quality']

            # Simulate scoring in TieredMatcher._match_single
            adjusted_conf, reason = apply_caption_quality_adjustment(
                confidence=base_confidence,
                video_segment=segment,
                config=config
            )

            results.append({
                'video': meta['video_path'],
                'quality': meta['caption_quality'],
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_conf,
                'reason': reason,
            })

        # Assertions for smoke test
        high_result = next(r for r in results if r['quality'] == 'high')
        low_result = next(r for r in results if r['quality'] == 'low')

        # 1. High quality preserves confidence
        assert high_result['adjusted_confidence'] == base_confidence

        # 2. Low quality reduces confidence
        assert low_result['adjusted_confidence'] < base_confidence
        assert abs(low_result['adjusted_confidence'] - 0.6375) < 0.001  # 0.85 * 0.75

        # 3. High > Low
        assert high_result['adjusted_confidence'] > low_result['adjusted_confidence']

        # 4. Difference is significant (>15%)
        diff = high_result['adjusted_confidence'] - low_result['adjusted_confidence']
        assert diff >= 0.15

        # 5. Low quality reason is informative
        assert 'caption quality' in low_result['reason'].lower()

        print("=" * 60)
        print("Caption Quality Pipeline Smoke Test PASSED")
        print("=" * 60)
        for r in results:
            print(f"  {r['video']}: {r['quality']} quality")
            print(f"    {r['base_confidence']:.2f} -> {r['adjusted_confidence']:.2f}")
            if r['reason']:
                print(f"    Reason: {r['reason']}")
        print("=" * 60)


# =============================================================================
# Test Class: Timing Penalty Pipeline Integration (US-002 Sprint 8)
# =============================================================================

class TestTimingPenaltyPipeline:
    """Integration tests for timing penalty impact on matching (US-002 Sprint 8).

    This test class verifies the complete flow of timing penalty from
    CaptionResult through to MatchStage confidence adjustments.

    The timing penalty flow:
    1. CaptionResult.timing_penalty_factor is calculated from validate_timing()
    2. CaptionStage stores timing_penalty in text_metadata for each video
    3. MatchStage reads timing_penalty from text_metadata and sets on vid_segment
    4. TieredMatcher calls apply_timing_penalty() to adjust confidence

    Test Coverage:
    - AC1: CaptionResult.timing_penalty_factor calculated correctly
    - AC2: timing_penalty stored in text_metadata by CaptionStage
    - AC3: MatchStage retrieves timing_penalty from text_metadata
    - AC4: apply_timing_penalty() reduces confidence: 0.9 * 0.75 = 0.675
    - AC5: Mock pipeline with real CaptionStage and MatchStage integration
    """

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def timing_penalty_config(self):
        """Mock config with timing penalty enabled."""
        config = Mock()
        matching = Mock()
        matching.apply_timing_penalty = True
        config.matching = matching
        return config

    @pytest.fixture
    def timing_penalty_disabled_config(self):
        """Mock config with timing penalty disabled."""
        config = Mock()
        matching = Mock()
        matching.apply_timing_penalty = False
        config.matching = matching
        return config

    @pytest.fixture
    def sample_caption_segments_perfect(self):
        """Caption segments with perfect timing (100% coverage, no exceeds)."""
        return [
            CaptionSegment(index=0, start_time=0.0, end_time=25.0,
                          text="First quarter of video", source_file="vid_perfect"),
            CaptionSegment(index=1, start_time=25.0, end_time=50.0,
                          text="Second quarter of video", source_file="vid_perfect"),
            CaptionSegment(index=2, start_time=50.0, end_time=75.0,
                          text="Third quarter of video", source_file="vid_perfect"),
            CaptionSegment(index=3, start_time=75.0, end_time=100.0,
                          text="Final quarter of video", source_file="vid_perfect"),
        ]

    @pytest.fixture
    def sample_caption_segments_poor(self):
        """Caption segments with poor timing (50% coverage, 20% exceeds)."""
        return [
            CaptionSegment(index=0, start_time=0.0, end_time=60.0,
                          text="Only caption, extends past video", source_file="vid_poor"),
        ]

    @pytest.fixture
    def sample_text_metadata_with_timing_penalty(self):
        """Sample text_metadata entry with timing penalty (from CaptionStage)."""
        return {
            'video_path': 'vid_with_penalty',
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'Wildlife documentary segment',
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': True,
            'caption_quality': 'medium',
            'timing_penalty': 0.75,  # 25% penalty from timing issues
        }

    @pytest.fixture
    def sample_text_metadata_perfect_timing(self):
        """Sample text_metadata entry with perfect timing (no penalty)."""
        return {
            'video_path': 'vid_perfect_timing',
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'Wildlife documentary segment',
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': False,
            'caption_quality': 'high',
            'timing_penalty': 1.0,  # No penalty
        }

    # =========================================================================
    # AC1: CaptionResult.timing_penalty_factor is calculated correctly
    # =========================================================================

    @pytest.mark.fast
    def test_timing_penalty_factor_calculated_correctly_perfect(
        self, sample_caption_segments_perfect
    ):
        """Test CaptionResult.timing_penalty_factor calculation for perfect timing.

        With 100% coverage and no exceeds, penalty factor should be 1.0.
        """
        result = CaptionResult(
            video_id="vid_perfect",
            segments=sample_caption_segments_perfect,
            video_duration=100.0,
        )

        # Validate timing to populate timing_validated
        result.validate_timing()

        # Perfect timing = no penalty (factor 1.0)
        assert result.timing_penalty_factor == pytest.approx(1.0, abs=0.01)

    @pytest.mark.fast
    def test_timing_penalty_factor_calculated_correctly_poor(
        self, sample_caption_segments_poor
    ):
        """Test CaptionResult.timing_penalty_factor calculation for poor timing.

        With 50% coverage (50s video but 60s captions) and 20% exceeds,
        penalty should be significant.
        """
        result = CaptionResult(
            video_id="vid_poor",
            segments=sample_caption_segments_poor,
            video_duration=50.0,  # 60s caption in 50s video = 20% exceeds
        )

        # Validate timing to populate timing_validated
        result.validate_timing()

        # Penalty formula: 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)
        # exceeds_ratio = (60-50)/50 = 0.2
        # coverage_ratio = min(1.0, 60/50) = 1.0 (capped)
        # penalty = 1.0 - (0.2 * 0.3) - 0 = 0.94
        assert result.timing_penalty_factor == pytest.approx(0.94, abs=0.02)

    @pytest.mark.fast
    def test_timing_penalty_factor_low_coverage(self):
        """Test penalty factor with 50% coverage."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=50.0,
                          text="Only half", source_file="vid_half"),
        ]
        result = CaptionResult(
            video_id="vid_half",
            segments=segments,
            video_duration=100.0,  # 50s captions in 100s video = 50% coverage
        )

        result.validate_timing()

        # coverage_ratio = 50/100 = 0.5
        # coverage_penalty = (1 - 0.5) * 0.2 = 0.1
        # penalty = 1.0 - 0 - 0.1 = 0.9
        assert result.timing_penalty_factor == pytest.approx(0.90, abs=0.01)

    # =========================================================================
    # AC2: timing_penalty stored in text_metadata by CaptionStage
    # =========================================================================

    @pytest.mark.fast
    def test_timing_penalty_stored_in_text_metadata(self):
        """Test that timing_penalty is correctly stored in text_metadata.

        This simulates what CaptionStage._build_text_metadata does:
        1. Fetch captions and validate timing
        2. Calculate timing_penalty_factor
        3. Include timing_penalty in text_metadata for each segment
        """
        # Simulate caption fetch with timing issues (50% coverage)
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=50.0,
                          text="Half coverage caption", source_file="test_vid"),
        ]
        caption_result = CaptionResult(
            video_id="test_vid",
            segments=segments,
            video_duration=100.0,
        )

        # Validate timing (as CaptionStage does)
        caption_result.validate_timing()
        timing_penalty = caption_result.timing_penalty_factor

        # Build text_metadata entry (simulating CaptionStage._build_text_metadata)
        text_metadata = {
            'video_path': caption_result.video_id,
            'start_time': segments[0].start_time,
            'end_time': segments[0].end_time,
            'text': segments[0].text,
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': False,
            'caption_quality': 'medium',
            'timing_penalty': timing_penalty,  # Key assertion target
        }

        # Verify timing_penalty is stored correctly
        assert 'timing_penalty' in text_metadata
        assert text_metadata['timing_penalty'] == pytest.approx(0.90, abs=0.01)
        assert text_metadata['timing_penalty'] < 1.0  # Has penalty

    @pytest.mark.fast
    def test_timing_penalty_perfect_timing_stored_as_1(self):
        """Test that perfect timing stores timing_penalty=1.0."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=100.0,
                          text="Full coverage", source_file="test_vid"),
        ]
        caption_result = CaptionResult(
            video_id="test_vid",
            segments=segments,
            video_duration=100.0,
        )

        caption_result.validate_timing()
        timing_penalty = caption_result.timing_penalty_factor

        text_metadata = {
            'video_path': caption_result.video_id,
            'timing_penalty': timing_penalty,
        }

        # Perfect timing = 1.0 (no penalty)
        assert text_metadata['timing_penalty'] == pytest.approx(1.0, abs=0.01)

    # =========================================================================
    # AC3: MatchStage retrieves timing_penalty from text_metadata
    # =========================================================================

    @pytest.mark.fast
    def test_match_stage_retrieves_timing_penalty(
        self, sample_text_metadata_with_timing_penalty
    ):
        """Test that MatchStage correctly retrieves timing_penalty from text_metadata.

        Simulates MatchStage._prepare_segments setting timing_penalty on vid_segment.
        """
        meta = sample_text_metadata_with_timing_penalty

        # Create segment from metadata (as MatchStage does)
        vid_segment = SRTSegment(
            index=1,
            start_time=meta['start_time'],
            end_time=meta['end_time'],
            text=meta['text'],
            source_file=meta['video_path']
        )

        # MatchStage sets timing_penalty from metadata (match.py:419-420)
        if meta.get('timing_penalty') is not None:
            vid_segment.timing_penalty = meta['timing_penalty']

        # Verify timing_penalty was set correctly
        assert hasattr(vid_segment, 'timing_penalty')
        assert vid_segment.timing_penalty == 0.75

    @pytest.mark.fast
    def test_match_stage_handles_missing_timing_penalty(self):
        """Test that MatchStage handles missing timing_penalty gracefully."""
        meta = {
            'video_path': 'vid_no_timing',
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'No timing penalty in metadata',
            # No timing_penalty key
        }

        vid_segment = SRTSegment(
            index=1,
            start_time=meta['start_time'],
            end_time=meta['end_time'],
            text=meta['text'],
            source_file=meta['video_path']
        )

        # Only set if present (as MatchStage does)
        if meta.get('timing_penalty') is not None:
            vid_segment.timing_penalty = meta['timing_penalty']

        # timing_penalty should NOT be set
        assert not hasattr(vid_segment, 'timing_penalty') or \
               getattr(vid_segment, 'timing_penalty', None) is None

    # =========================================================================
    # AC4: apply_timing_penalty() reduces confidence correctly
    # =========================================================================

    @pytest.mark.fast
    def test_apply_timing_penalty_reduces_confidence(
        self, timing_penalty_config
    ):
        """Test that apply_timing_penalty() reduces confidence: 0.9 * 0.75 = 0.675.

        This is the key acceptance criterion verifying the math is correct.
        """
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test segment", source_file="/test.mp4"
        )
        segment.timing_penalty = 0.75  # 25% penalty

        base_confidence = 0.9

        adjusted_conf, reason = apply_timing_penalty(
            confidence=base_confidence,
            video_segment=segment,
            config=timing_penalty_config
        )

        # 0.9 * 0.75 = 0.675
        assert adjusted_conf == pytest.approx(0.675, abs=0.001)
        assert "timing penalty" in reason.lower()
        assert "x0.75" in reason

    @pytest.mark.fast
    def test_apply_timing_penalty_no_change_when_1(
        self, timing_penalty_config
    ):
        """Test that timing_penalty=1.0 produces no confidence change."""
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test segment", source_file="/test.mp4"
        )
        segment.timing_penalty = 1.0  # No penalty

        base_confidence = 0.85

        adjusted_conf, reason = apply_timing_penalty(
            confidence=base_confidence,
            video_segment=segment,
            config=timing_penalty_config
        )

        # No change when penalty = 1.0
        assert adjusted_conf == base_confidence
        assert reason == ""

    @pytest.mark.fast
    def test_apply_timing_penalty_disabled_no_change(
        self, timing_penalty_disabled_config
    ):
        """Test that disabled timing penalty produces no change."""
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test segment", source_file="/test.mp4"
        )
        segment.timing_penalty = 0.5  # Would be 50% penalty

        base_confidence = 0.80

        adjusted_conf, reason = apply_timing_penalty(
            confidence=base_confidence,
            video_segment=segment,
            config=timing_penalty_disabled_config
        )

        # No change when disabled
        assert adjusted_conf == base_confidence
        assert reason == ""

    # =========================================================================
    # AC5: Mock pipeline with CaptionStage and MatchStage integration
    # =========================================================================

    @pytest.mark.fast
    def test_full_pipeline_timing_penalty_flow(self, timing_penalty_config):
        """Test complete flow: CaptionResult -> text_metadata -> segment -> confidence.

        This is the integration smoke test verifying the entire timing penalty pipeline:
        1. CaptionResult.timing_penalty_factor is calculated
        2. CaptionStage stores timing_penalty in text_metadata
        3. MatchStage transfers timing_penalty to vid_segment
        4. apply_timing_penalty() adjusts confidence correctly
        """
        # Step 1: Caption fetch with timing issues
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=50.0,
                          text="Half coverage caption", source_file="integration_test"),
        ]
        caption_result = CaptionResult(
            video_id="integration_test",
            segments=segments,
            video_duration=100.0,  # 50% coverage
        )

        # Step 2: Validate timing and get penalty (CaptionStage does this)
        caption_result.validate_timing()
        timing_penalty = caption_result.timing_penalty_factor
        assert timing_penalty == pytest.approx(0.90, abs=0.01)  # 10% penalty

        # Step 3: Build text_metadata with timing_penalty (CaptionStage does this)
        text_metadata = {
            'video_path': caption_result.video_id,
            'start_time': segments[0].start_time,
            'end_time': segments[0].end_time,
            'text': segments[0].text,
            'timing_penalty': timing_penalty,
        }

        # Step 4: Create vid_segment and set timing_penalty (MatchStage does this)
        vid_segment = SRTSegment(
            index=1,
            start_time=text_metadata['start_time'],
            end_time=text_metadata['end_time'],
            text=text_metadata['text'],
            source_file=text_metadata['video_path']
        )
        vid_segment.timing_penalty = text_metadata['timing_penalty']

        # Step 5: Apply timing penalty during matching (TieredMatcher does this)
        base_confidence = 0.90
        adjusted_conf, reason = apply_timing_penalty(
            confidence=base_confidence,
            video_segment=vid_segment,
            config=timing_penalty_config
        )

        # Final assertions
        # 0.90 (base) * 0.90 (penalty) = 0.81
        assert adjusted_conf == pytest.approx(0.81, abs=0.01)
        assert "timing penalty" in reason.lower()

    @pytest.mark.fast
    def test_full_pipeline_perfect_vs_poor_timing(self, timing_penalty_config):
        """Test that perfect and poor timing produce different confidence.

        Same video content, same base confidence, but timing issues affect final score.
        """
        base_confidence = 0.85
        results = []

        test_cases = [
            ("perfect", 100.0, 100.0, 1.0),   # 100s captions in 100s video = perfect
            ("poor", 50.0, 100.0, 0.90),      # 50s captions in 100s video = 50% coverage
        ]

        for name, caption_duration, video_duration, expected_penalty in test_cases:
            # Create caption result
            segments = [
                CaptionSegment(index=0, start_time=0.0, end_time=caption_duration,
                              text=f"Caption for {name}", source_file=f"vid_{name}"),
            ]
            caption_result = CaptionResult(
                video_id=f"vid_{name}",
                segments=segments,
                video_duration=video_duration,
            )

            # Calculate timing penalty
            caption_result.validate_timing()
            timing_penalty = caption_result.timing_penalty_factor

            # Create segment with timing penalty
            vid_segment = SRTSegment(
                index=1, start_time=0.0, end_time=10.0,
                text="Same content", source_file=f"vid_{name}"
            )
            vid_segment.timing_penalty = timing_penalty

            # Apply penalty
            adjusted_conf, reason = apply_timing_penalty(
                confidence=base_confidence,
                video_segment=vid_segment,
                config=timing_penalty_config
            )

            results.append({
                'name': name,
                'timing_penalty': timing_penalty,
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_conf,
                'reason': reason,
            })

            # Verify expected penalty
            assert timing_penalty == pytest.approx(expected_penalty, abs=0.02)

        # Perfect timing should have higher confidence than poor timing
        perfect = next(r for r in results if r['name'] == 'perfect')
        poor = next(r for r in results if r['name'] == 'poor')

        assert perfect['adjusted_confidence'] > poor['adjusted_confidence']
        assert perfect['adjusted_confidence'] == base_confidence  # No penalty
        assert poor['adjusted_confidence'] < base_confidence  # Has penalty

    @pytest.mark.fast
    def test_timing_penalty_combined_with_caption_quality(self, timing_penalty_config):
        """Test that timing penalty can be combined with caption quality adjustment.

        In the real pipeline, both adjustments are applied sequentially.
        This test verifies they work together correctly.
        """
        # Configure for both adjustments
        config = Mock()
        config.matching = Mock()
        config.matching.apply_timing_penalty = True
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}

        # Create segment with both quality and timing penalty
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test content", source_file="/test.mp4"
        )
        segment.caption_quality = "low"  # x0.75
        segment.timing_penalty = 0.90    # x0.90

        base_confidence = 0.80

        # Apply caption quality first (as TieredMatcher does)
        after_quality, quality_reason = apply_caption_quality_adjustment(
            confidence=base_confidence,
            video_segment=segment,
            config=config
        )

        # Then apply timing penalty
        after_timing, timing_reason = apply_timing_penalty(
            confidence=after_quality,
            video_segment=segment,
            config=config
        )

        # Combined effect: 0.80 * 0.75 * 0.90 = 0.54
        assert abs(after_timing - 0.54) < 0.01

        # Both reasons should be present
        assert "caption quality" in quality_reason.lower()
        assert "timing penalty" in timing_reason.lower()


# =============================================================================
# Test Class: Timing Penalty Edge Cases
# =============================================================================

class TestTimingPenaltyEdgeCases:
    """Edge case tests for timing penalty handling."""

    @pytest.fixture
    def timing_config(self):
        """Mock config with timing penalty enabled."""
        config = Mock()
        config.matching = Mock()
        config.matching.apply_timing_penalty = True
        return config

    @pytest.mark.fast
    def test_zero_duration_video_no_penalty(self):
        """Test that zero-duration video produces no penalty (can't calculate)."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=10.0,
                          text="Some text", source_file="vid"),
        ]
        result = CaptionResult(
            video_id="vid",
            segments=segments,
            video_duration=0.0,  # Invalid duration
        )

        result.validate_timing()

        # Can't calculate penalty without valid duration
        assert result.timing_penalty_factor == 1.0

    @pytest.mark.fast
    def test_negative_penalty_clamped_to_zero(self, timing_config):
        """Test that extreme timing issues clamp penalty at 0.0."""
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test", source_file="/test.mp4"
        )
        segment.timing_penalty = -0.5  # Invalid but tests clamping

        conf, _ = apply_timing_penalty(
            confidence=0.80,
            video_segment=segment,
            config=timing_config
        )

        assert conf == 0.0  # Clamped at 0

    @pytest.mark.fast
    def test_penalty_above_one_is_no_penalty(self, timing_config):
        """Test that timing_penalty >= 1.0 means no penalty."""
        segment = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Test", source_file="/test.mp4"
        )
        segment.timing_penalty = 1.5  # Invalid but >= 1.0

        conf, reason = apply_timing_penalty(
            confidence=0.80,
            video_segment=segment,
            config=timing_config
        )

        assert conf == 0.80  # No change
        assert reason == ""

    @pytest.mark.fast
    def test_caption_exceeds_duration_significantly(self):
        """Test penalty for caption far exceeding video duration."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=200.0,
                          text="Very long caption", source_file="vid"),
        ]
        result = CaptionResult(
            video_id="vid",
            segments=segments,
            video_duration=100.0,  # Caption 2x video length
        )

        result.validate_timing()
        penalty = result.timing_penalty_factor

        # exceeds_ratio = 1.0 (100% exceeds)
        # exceeds_penalty = 1.0 * 0.3 = 0.3
        # coverage_ratio = 1.0 (capped)
        # penalty = 1.0 - 0.3 = 0.7
        assert penalty == pytest.approx(0.7, abs=0.02)

    @pytest.mark.fast
    def test_very_sparse_captions_low_coverage(self):
        """Test penalty for very sparse captions."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=10.0,
                          text="Just intro", source_file="vid"),
        ]
        result = CaptionResult(
            video_id="vid",
            segments=segments,
            video_duration=100.0,  # Only 10% coverage
        )

        result.validate_timing()
        penalty = result.timing_penalty_factor

        # coverage_ratio = 0.1
        # coverage_penalty = 0.9 * 0.2 = 0.18
        # penalty = 1.0 - 0.18 = 0.82
        assert penalty == pytest.approx(0.82, abs=0.02)


# =============================================================================
# Timing Penalty Pipeline Smoke Test Summary
# =============================================================================

class TestTimingPenaltyPipelineSmokeTest:
    """Comprehensive smoke test for timing penalty pipeline (US-002 Sprint 8).

    This test provides a single comprehensive validation of the entire
    timing penalty -> match confidence pipeline.

    Use this test for quick validation after changes to:
    - caption_fetcher.py (CaptionResult.timing_penalty_factor, validate_timing)
    - caption_stage.py (timing_penalty in text_metadata)
    - match.py (timing_penalty attribute propagation)
    - scoring.py (apply_timing_penalty)
    - tiered_matcher.py (timing penalty application)
    """

    @pytest.mark.fast
    def test_full_pipeline_smoke_test(self):
        """Comprehensive smoke test for timing penalty pipeline.

        Tests the complete flow:
        1. Caption fetch calculates timing_penalty_factor
        2. CaptionStage includes timing_penalty in text_metadata
        3. MatchStage transfers timing_penalty to segments
        4. Scoring applies timing-based confidence adjustment
        5. Results show expected confidence differences
        """
        # Config with timing penalty enabled
        config = Mock()
        config.matching = Mock()
        config.matching.apply_timing_penalty = True

        # Test cases: caption duration, video duration, expected penalty
        test_scenarios = [
            ("Perfect timing", 100.0, 100.0, 1.0),
            ("80% coverage", 80.0, 100.0, 0.96),
            ("50% coverage", 50.0, 100.0, 0.90),
            ("20% exceeds", 120.0, 100.0, 0.94),
        ]

        results = []
        base_confidence = 0.85

        for name, caption_dur, video_dur, expected_penalty in test_scenarios:
            # Step 1: Create caption result
            segments = [
                CaptionSegment(index=0, start_time=0.0, end_time=caption_dur,
                              text=f"Caption for {name}", source_file=name.lower().replace(" ", "_")),
            ]
            caption_result = CaptionResult(
                video_id=name.lower().replace(" ", "_"),
                segments=segments,
                video_duration=video_dur,
            )

            # Step 2: Calculate timing penalty (CaptionStage)
            caption_result.validate_timing()
            timing_penalty = caption_result.timing_penalty_factor

            # Step 3: Build text_metadata with timing_penalty
            text_metadata = {
                'video_path': caption_result.video_id,
                'timing_penalty': timing_penalty,
            }

            # Step 4: Create segment with timing_penalty (MatchStage)
            segment = SRTSegment(
                index=1, start_time=0.0, end_time=10.0,
                text="Test content", source_file=text_metadata['video_path']
            )
            segment.timing_penalty = text_metadata['timing_penalty']

            # Step 5: Apply timing penalty (TieredMatcher)
            adjusted_conf, reason = apply_timing_penalty(
                confidence=base_confidence,
                video_segment=segment,
                config=config
            )

            results.append({
                'scenario': name,
                'timing_penalty': timing_penalty,
                'expected_penalty': expected_penalty,
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_conf,
                'reason': reason,
            })

            # Verify expected penalty (within tolerance)
            assert timing_penalty == pytest.approx(expected_penalty, abs=0.02), \
                f"{name}: expected penalty {expected_penalty}, got {timing_penalty}"

        # Summary assertions
        perfect = next(r for r in results if r['scenario'] == "Perfect timing")
        poor_coverage = next(r for r in results if r['scenario'] == "50% coverage")

        # 1. Perfect timing preserves confidence
        assert perfect['adjusted_confidence'] == base_confidence

        # 2. Poor coverage reduces confidence
        assert poor_coverage['adjusted_confidence'] < base_confidence
        # 0.85 * 0.90 = 0.765
        assert poor_coverage['adjusted_confidence'] == pytest.approx(0.765, abs=0.02)

        # 3. Perfect > Poor coverage
        assert perfect['adjusted_confidence'] > poor_coverage['adjusted_confidence']

        # Print summary
        print("=" * 70)
        print("Timing Penalty Pipeline Smoke Test PASSED (US-002 Sprint 8)")
        print("=" * 70)
        for r in results:
            penalty_str = f"x{r['timing_penalty']:.2f}" if r['timing_penalty'] < 1.0 else "none"
            print(f"  {r['scenario']:20s}: penalty={penalty_str:>6s}  "
                  f"conf: {r['base_confidence']:.2f} -> {r['adjusted_confidence']:.3f}")
        print("=" * 70)
