"""
Additional tests for src/vision.py to cover remaining lines.

Focuses on:
- get_scene_text helper function (lines 520-544)
- process_video_vision simplified function (lines 373-410)
- Edge cases in TranscriptAnalyzer
- VisionCache serialization
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import (
    TranscriptAnalyzer,
    VisionProcessor,
    VisionCache,
    VideoVisionDecision,
    SceneAnalysis,
    process_video_vision,
    process_video_vision_full,
    get_scene_text,
)
from src.cache import CacheEntry


# ============================================================================
# Test get_scene_text helper function (lines 520-544)
# ============================================================================

class TestGetSceneText:
    """Test get_scene_text helper function"""

    def test_get_scene_text_with_dict_segments(self):
        """Test with dict transcript segments"""
        scene = {"start_time": 5.0, "end_time": 15.0}
        transcript_segments = [
            {"start_time": 0.0, "end_time": 4.0, "text": "Before scene"},
            {"start_time": 6.0, "end_time": 12.0, "text": "Inside scene"},
            {"start_time": 14.0, "end_time": 16.0, "text": "Overlapping end"},
            {"start_time": 20.0, "end_time": 25.0, "text": "After scene"}
        ]

        result = get_scene_text(scene, transcript_segments)

        assert "Inside scene" in result
        assert "Overlapping end" in result
        assert "Before scene" not in result
        assert "After scene" not in result

    def test_get_scene_text_with_object_segments(self):
        """Test with SRTSegment-like objects"""
        scene = {"start_time": 0.0, "end_time": 10.0}

        seg1 = Mock()
        seg1.start_time = 0.0
        seg1.end_time = 5.0
        seg1.text = "First segment"

        seg2 = Mock()
        seg2.start_time = 5.0
        seg2.end_time = 10.0
        seg2.text = "Second segment"

        transcript_segments = [seg1, seg2]

        result = get_scene_text(scene, transcript_segments)

        assert "First segment" in result
        assert "Second segment" in result

    def test_get_scene_text_empty_segments(self):
        """Test with empty transcript segments"""
        scene = {"start_time": 0.0, "end_time": 10.0}
        transcript_segments = []

        result = get_scene_text(scene, transcript_segments)

        assert result == ""

    def test_get_scene_text_no_overlap(self):
        """Test when no segments overlap with scene"""
        scene = {"start_time": 100.0, "end_time": 110.0}
        transcript_segments = [
            {"start_time": 0.0, "end_time": 10.0, "text": "Early segment"}
        ]

        result = get_scene_text(scene, transcript_segments)

        assert result == ""

    def test_get_scene_text_default_end_time(self):
        """Test scene with missing end_time uses default"""
        scene = {"start_time": 0.0}  # No end_time
        transcript_segments = [
            {"start_time": 2.0, "end_time": 4.0, "text": "Within default range"}
        ]

        result = get_scene_text(scene, transcript_segments)

        assert "Within default range" in result


# ============================================================================
# Test process_video_vision simplified function (lines 353-410)
# ============================================================================

class TestProcessVideoVision:
    """Test process_video_vision simplified function"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = True
        config.vision.max_scenes_per_video = 50
        return config

    @pytest.fixture
    def mock_config_disabled(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = False
        return config

    def test_process_vision_disabled(self, mock_config_disabled):
        """Test returns empty when vision disabled"""
        result = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=[],
            cache=None,
            config=mock_config_disabled
        )

        assert result == []

    def test_process_vision_with_dict_segments(self, mock_config):
        """Test processes dict transcript segments"""
        transcript_segments = [
            {"start_time": 0.0, "end_time": 5.0, "text": "First segment text"},
            {"start_time": 5.0, "end_time": 10.0, "text": "Second segment text"}
        ]

        result = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript_segments,
            cache=".cache",
            config=mock_config
        )

        assert len(result) == 2
        assert result[0]['description'] == "First segment text"
        assert result[1]['description'] == "Second segment text"

    def test_process_vision_with_object_segments(self, mock_config):
        """Test processes SRTSegment-like objects"""
        seg1 = Mock()
        seg1.start_time = 0.0
        seg1.end_time = 5.0
        seg1.text = "Object segment one"

        seg2 = Mock()
        seg2.start_time = 5.0
        seg2.end_time = 10.0
        seg2.text = "Object segment two"

        transcript_segments = [seg1, seg2]

        result = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript_segments,
            cache=Mock(cache_dir=".cache"),
            config=mock_config
        )

        assert len(result) == 2
        assert result[0]['description'] == "Object segment one"

    def test_process_vision_skips_empty_text(self, mock_config):
        """Test skips segments with empty text"""
        transcript_segments = [
            {"start_time": 0.0, "end_time": 5.0, "text": "Has text"},
            {"start_time": 5.0, "end_time": 10.0, "text": ""},  # Empty
            {"start_time": 10.0, "end_time": 15.0, "text": "   "}  # Whitespace only
        ]

        result = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript_segments,
            cache=None,
            config=mock_config
        )

        assert len(result) == 1
        assert result[0]['description'] == "Has text"

    def test_process_vision_respects_max_scenes(self, mock_config):
        """Test respects max_scenes_per_video limit"""
        mock_config.vision.max_scenes_per_video = 2

        transcript_segments = [
            {"start_time": i * 5, "end_time": (i + 1) * 5, "text": f"Segment {i}"}
            for i in range(10)
        ]

        result = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript_segments,
            cache=None,
            config=mock_config,
            max_scenes_per_video=2
        )

        assert len(result) == 2


# ============================================================================
# Test TranscriptAnalyzer edge cases (lines 121-130)
# ============================================================================

class TestTranscriptAnalyzerEdgeCases:
    """Test TranscriptAnalyzer edge cases"""

    @pytest.fixture
    def analyzer(self):
        config = Mock()
        config.vision = Mock()
        config.vision.min_words_per_scene = 5
        config.vision.coverage_threshold = 0.3
        config.vision.max_scenes_per_video = 50
        return TranscriptAnalyzer(config)

    def test_analyze_good_coverage_reason(self, analyzer):
        """Test reason message for good coverage"""
        scenes = [{"start_time": 0.0, "end_time": 5.0}]
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "Lots of words here to test coverage calculation"}]

        decision = analyzer.analyze_video_transcript("/test.mp4", scenes, transcript)

        assert "Good transcript coverage" in decision.reason

    def test_analyze_low_coverage_reason(self, analyzer):
        """Test reason message for low coverage"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0},
            {"start_time": 15.0, "end_time": 20.0}
        ]
        transcript = []  # No transcript

        decision = analyzer.analyze_video_transcript("/test.mp4", scenes, transcript)

        assert "Low transcript coverage" in decision.reason

    def test_analyze_sparse_scenes_reason(self, analyzer):
        """Test reason message for sparse scenes"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0}
        ]
        # Only one scene has text
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "Words to fill this scene with content"}]

        decision = analyzer.analyze_video_transcript("/test.mp4", scenes, transcript)

        # Should mention sparse scenes
        assert "sparse" in decision.reason.lower() or "Low" in decision.reason

    def test_get_priority_scenes_with_objects(self, analyzer):
        """Test get_priority_scenes with SRTSegment objects"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0}
        ]

        seg = Mock()
        seg.start_time = 0.0
        seg.end_time = 5.0
        seg.text = "Only this scene has text with enough words to count"

        transcript = [seg]

        result = analyzer.get_priority_scenes(scenes, transcript, max_scenes=2)

        # Scenes 1 and 2 (indices) should be prioritized (no text)
        assert len(result) == 2
        assert 1 in result or 2 in result


# ============================================================================
# Test VisionCache (lines 190-212)
# ============================================================================

class TestVisionCacheExtended:
    """Test VisionCache serialization"""

    def test_serialize_entry(self):
        """Test entry serialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VisionCache(cache_dir=tmpdir, index_name="vision_test.json")

            entry = CacheEntry(
                key="test_key",
                data={"description": "A beautiful scene"},
                cached_at="2026-01-01T00:00:00",
                metadata={"api_calls": 1}
            )

            result = cache._serialize_entry(entry)

            assert result['data'] == {"description": "A beautiful scene"}
            assert result['cached_at'] == "2026-01-01T00:00:00"
            assert result['metadata'] == {"api_calls": 1}

    def test_deserialize_entry(self):
        """Test entry deserialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VisionCache(cache_dir=tmpdir, index_name="vision_test.json")

            data = {
                'data': {"description": "Test description"},
                'cached_at': "2026-01-01T00:00:00",
                'metadata': {"version": 1}
            }

            entry = cache._deserialize_entry(data)

            assert entry.data == {"description": "Test description"}
            assert entry.cached_at == "2026-01-01T00:00:00"
            assert entry.metadata == {"version": 1}

    def test_deserialize_entry_missing_metadata(self):
        """Test deserialization with missing metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = VisionCache(cache_dir=tmpdir, index_name="vision_test.json")

            data = {
                'data': {"description": "Test"},
                'cached_at': "2026-01-01T00:00:00"
                # No metadata
            }

            entry = cache._deserialize_entry(data)

            assert entry.metadata == {}


# ============================================================================
# Test VisionProcessor OpenAI path (lines 232-234)
# ============================================================================

class TestVisionProcessorProviders:
    """Test VisionProcessor provider handling"""

    def test_get_api_key_openai(self):
        """Test getting OpenAI API key"""
        config = Mock()
        config.vision = Mock()
        config.vision.provider = "openai"
        config.vision.model = "gpt-4-vision"
        config.vision.estimated_cost_per_call = 0.01

        processor = VisionProcessor(config)

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_openai_key'}):
            key = processor._get_api_key()
            assert key == 'test_openai_key'

    def test_get_api_key_unknown_provider(self):
        """Test getting API key for unknown provider"""
        config = Mock()
        config.vision = Mock()
        config.vision.provider = "unknown"
        config.vision.model = "test"
        config.vision.estimated_cost_per_call = 0.001

        processor = VisionProcessor(config)

        key = processor._get_api_key()
        assert key is None


# ============================================================================
# Test process_video_vision_full (lines 413-517)
# ============================================================================

class TestProcessVideoVisionFull:
    """Test process_video_vision_full function"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = True
        config.vision.min_words_per_scene = 5
        config.vision.coverage_threshold = 0.3
        config.vision.max_scenes_per_video = 5
        config.vision.estimated_cost_per_call = 0.001
        config.vision.provider = "gemini"
        config.vision.model = "gemini-2.0-flash"
        return config

    @pytest.fixture
    def mock_config_disabled(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = False
        return config

    def test_process_full_disabled(self, mock_config_disabled):
        """Test returns empty when disabled"""
        results, stats = process_video_vision_full(
            video_path="/test.mp4",
            scenes=[],
            transcript_segments=[],
            config=mock_config_disabled
        )

        assert results == []
        assert stats.get('skipped') is True

    def test_process_full_good_coverage(self, mock_config):
        """Test skips when transcript coverage is good"""
        scenes = [{"start_time": 0.0, "end_time": 5.0}]
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "This has plenty of words for coverage"}]

        results, stats = process_video_vision_full(
            video_path="/test.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            config=mock_config
        )

        assert results == []
        assert stats.get('skipped') is True

    @patch.object(VisionProcessor, 'describe_scene')
    @patch.object(VisionProcessor, '_get_api_key')
    def test_process_full_with_vision(self, mock_api_key, mock_describe, mock_config):
        """Test full vision processing"""
        mock_api_key.return_value = "test_key"
        mock_describe.return_value = "A beautiful mountain scene"

        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0}
        ]
        # Only one scene has transcript
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "Words here but sparse overall"}]

        results, stats = process_video_vision_full(
            video_path="/test.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            config=mock_config,
            cache_dir=".cache"
        )

        assert len(results) > 0
        assert 'scenes_processed' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
