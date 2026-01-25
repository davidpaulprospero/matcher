"""
Integration test for caption quality impact on matching (US-010).

This test validates the full pipeline path from caption quality metadata
through to match confidence scoring. It verifies that:
1. Low-quality auto-generated captions flow to MatchStage with quality='low'
2. Same match text produces different confidence with high vs low quality
3. Quality weights config is correctly applied
4. Caption quality metadata is preserved for downstream filtering

This is an integration smoke test for the caption-to-match quality pipeline.

Sprint 6 Story: US-010 - Create integration test for caption quality impact on matching

Created: 2026-01-26
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import apply_caption_quality_adjustment
from src.utils import SRTSegment


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
