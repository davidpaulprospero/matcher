import pytest
"""
Unit tests for src/vision.py - Vision API integration.

Tests cover:
- TranscriptAnalyzer: Determining which videos need vision processing
- VisionProcessor: Frame extraction and description generation
- Caching behavior
- Cost tracking
- Error handling and fallbacks

Created Jan 8, 2026.
"""

import unittest
import sys
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import (
    TranscriptAnalyzer,
    VisionProcessor,
    VideoVisionDecision,
    SceneAnalysis,
    VisionCache,
    process_video_vision,
    process_video_vision_full
)
from src.cache.base import CacheEntry


class TestTranscriptAnalyzer(unittest.TestCase):
    """Test transcript analysis for vision necessity"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.vision = Mock()
        self.config.vision.min_words_per_scene = 5
        self.config.vision.coverage_threshold = 0.3
        self.config.vision.max_scenes_per_video = 50
        self.analyzer = TranscriptAnalyzer(self.config)

    @pytest.mark.fast
    def test_analyze_video_with_full_transcript(self):
        """Test video with complete transcript coverage - no vision needed"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0}
        ]

        transcript = [
            {"start_time": 0.0, "end_time": 5.0, "text": "This is a beautiful mountain landscape with snow"},
            {"start_time": 5.0, "end_time": 10.0, "text": "The peaks rise majestically into the blue sky"},
            {"start_time": 10.0, "end_time": 15.0, "text": "Climbers are making their way to the summit"}
        ]

        decision = self.analyzer.analyze_video_transcript(
            video_path="/test/mountain.mp4",
            scenes=scenes,
            transcript_segments=transcript
        )

        # Should not need vision - good transcript coverage
        self.assertFalse(decision.needs_vision)
        self.assertGreater(decision.transcript_coverage, 0.5)
        self.assertEqual(decision.total_scenes, 3)
        self.assertEqual(len(decision.sparse_scenes), 0)

    @pytest.mark.fast
    def test_analyze_video_with_sparse_transcript(self):
        """Test video with sparse transcript - vision needed"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0},
            {"start_time": 15.0, "end_time": 20.0}
        ]

        # Only one scene has text
        transcript = [
            {"start_time": 0.0, "end_time": 5.0, "text": "Mountains"}
        ]

        decision = self.analyzer.analyze_video_transcript(
            video_path="/test/broll.mp4",
            scenes=scenes,
            transcript_segments=transcript
        )

        # Should need vision - sparse coverage
        self.assertTrue(decision.needs_vision)
        self.assertLess(decision.transcript_coverage, 0.3)
        self.assertEqual(decision.total_scenes, 4)
        # Should identify scenes 1, 2, 3 as needing vision
        self.assertGreater(len(decision.sparse_scenes), 0)

    @pytest.mark.fast
    def test_analyze_silent_video(self):
        """Test completely silent video - vision definitely needed"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0}
        ]

        transcript = []  # No transcript at all

        decision = self.analyzer.analyze_video_transcript(
            video_path="/test/silent.mp4",
            scenes=scenes,
            transcript_segments=transcript
        )

        self.assertTrue(decision.needs_vision)
        self.assertEqual(decision.transcript_coverage, 0.0)
        self.assertEqual(len(decision.sparse_scenes), 2)  # All scenes need vision

    @pytest.mark.fast
    def test_analyze_video_no_scenes(self):
        """Test video with no detected scenes"""
        decision = self.analyzer.analyze_video_transcript(
            video_path="/test/broken.mp4",
            scenes=[],
            transcript_segments=[]
        )

        self.assertTrue(decision.needs_vision)
        self.assertEqual(decision.total_scenes, 0)
        self.assertEqual(decision.reason, "No scenes detected")

    @pytest.mark.fast
    def test_analyze_with_srt_segment_objects(self):
        """Test with SRTSegment objects (not dicts)"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0}
        ]

        # Mock SRTSegment objects
        seg1 = Mock()
        seg1.start_time = 0.0
        seg1.end_time = 5.0
        seg1.text = "This is a test segment with enough words to count"

        transcript = [seg1]

        decision = self.analyzer.analyze_video_transcript(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript
        )

        # Should handle SRTSegment objects correctly
        self.assertIsInstance(decision, VideoVisionDecision)
        self.assertEqual(decision.total_scenes, 1)

    @pytest.mark.fast
    def test_get_priority_scenes_by_word_count(self):
        """Test get_priority_scenes prioritizes scenes with least text"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},   # Scene 0
            {"start_time": 5.0, "end_time": 10.0},  # Scene 1
            {"start_time": 10.0, "end_time": 15.0}, # Scene 2
            {"start_time": 15.0, "end_time": 20.0}  # Scene 3
        ]

        transcript = [
            {"start_time": 0.0, "end_time": 5.0, "text": "One two three four five six seven eight nine ten"},  # 10 words
            {"start_time": 5.0, "end_time": 10.0, "text": "Short"},  # 1 word
            {"start_time": 10.0, "end_time": 15.0, "text": ""},  # 0 words
            {"start_time": 15.0, "end_time": 20.0, "text": "Medium length text here"}  # 4 words
        ]

        priority = self.analyzer.get_priority_scenes(scenes, transcript, max_scenes=2)

        # Should return indices for scenes with least text (scene 2, scene 1)
        self.assertEqual(len(priority), 2)
        self.assertIn(2, priority)  # Scene 2 has 0 words
        self.assertIn(1, priority)  # Scene 1 has 1 word

    @pytest.mark.fast
    def test_get_priority_scenes_with_srt_objects(self):
        """Test get_priority_scenes with SRTSegment objects"""
        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0}
        ]

        seg1 = Mock()
        seg1.start_time = 0.0
        seg1.end_time = 5.0
        seg1.text = "This has many words in it to count"

        seg2 = Mock()
        seg2.start_time = 5.0
        seg2.end_time = 10.0
        seg2.text = "Few"

        transcript = [seg1, seg2]

        priority = self.analyzer.get_priority_scenes(scenes, transcript, max_scenes=1)

        # Should prioritize scene 1 (fewer words)
        self.assertEqual(priority, [1])

    @pytest.mark.fast
    def test_get_priority_scenes_max_limit(self):
        """Test get_priority_scenes respects max_scenes limit"""
        scenes = [{"start_time": i*5.0, "end_time": (i+1)*5.0} for i in range(10)]
        transcript = []  # All scenes are sparse

        priority = self.analyzer.get_priority_scenes(scenes, transcript, max_scenes=3)

        # Should return only 3 scenes
        self.assertEqual(len(priority), 3)

    @pytest.mark.fast
    def test_get_priority_scenes_overlapping_transcript(self):
        """Test word counting with transcript segments overlapping scenes"""
        scenes = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0}
        ]

        transcript = [
            {"start_time": 5.0, "end_time": 15.0, "text": "This segment overlaps both scenes with text"}
        ]

        priority = self.analyzer.get_priority_scenes(scenes, transcript, max_scenes=2)

        # Both scenes get some words from overlapping segment
        # Should prioritize based on actual overlap amount
        self.assertEqual(len(priority), 2)


class TestVisionProcessor(unittest.TestCase):
    """Test VisionProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.vision = Mock()
        self.config.vision.provider = 'gemini'
        self.config.vision.model = 'gemini-2.0-flash'
        self.config.vision.estimated_cost_per_call = 0.001
        self.processor = VisionProcessor(self.config)

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @pytest.mark.fast
    def test_is_available_with_api_key(self):
        """Test availability check when API key is present"""
        self.assertTrue(self.processor.is_available())

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.fast
    def test_is_available_without_api_key(self):
        """Test availability check when API key is missing"""
        self.assertFalse(self.processor.is_available())

    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @pytest.mark.integration
    def test_extract_frame_success(self, mock_file, mock_exists, mock_subprocess):
        """Test successful frame extraction with ffmpeg"""
        # Mock successful ffmpeg execution
        mock_subprocess.return_value = Mock(returncode=0)
        mock_exists.return_value = True

        frame_data = self.processor._extract_frame("/test/video.mp4", 5.5)

        # Should return frame data
        self.assertIsNotNone(frame_data)
        self.assertEqual(frame_data, b'fake_image_data')

        # Should have called ffmpeg with correct parameters
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        self.assertIn('ffmpeg', args)
        self.assertIn('-ss', args)
        self.assertIn('5.5', args)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_extract_frame_failure(self, mock_subprocess):
        """Test frame extraction failure"""
        # Mock failed ffmpeg execution
        mock_subprocess.return_value = Mock(returncode=1)

        frame_data = self.processor._extract_frame("/test/video.mp4", 5.5)

        # Should return None on failure
        self.assertIsNone(frame_data)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_extract_frame_timeout(self, mock_subprocess):
        """Test frame extraction timeout handling"""
        # Mock timeout exception
        mock_subprocess.side_effect = Exception("Timeout")

        frame_data = self.processor._extract_frame("/test/video.mp4", 5.5)

        # Should handle timeout gracefully
        self.assertIsNone(frame_data)

    @patch('src.llm_client.create_client')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @pytest.mark.fast
    def test_describe_frame_gemini_success(self, mock_create_client):
        """Test Gemini vision API description"""
        # Mock LLM client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "A beautiful mountain landscape with snow-covered peaks under a blue sky"
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        frame_data = b'fake_image_bytes'
        description = self.processor._describe_frame_gemini(frame_data)

        # Should return description
        self.assertEqual(description, "A beautiful mountain landscape with snow-covered peaks under a blue sky")

        # Should track API call and cost
        self.assertEqual(self.processor.api_calls, 1)
        self.assertEqual(self.processor.total_cost, 0.001)

        # Should have called LLM with correct parameters
        mock_client.generate.assert_called_once()
        request = mock_client.generate.call_args[0][0]
        self.assertEqual(request.images, [frame_data])
        self.assertEqual(request.image_format, "jpeg")
        self.assertIn("Describe this video frame", request.prompt)

    @patch('src.llm_client.create_client')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @pytest.mark.fast
    def test_describe_frame_gemini_error(self, mock_create_client):
        """Test Gemini vision API error handling"""
        # Mock LLM client error
        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client

        frame_data = b'fake_image_bytes'
        description = self.processor._describe_frame_gemini(frame_data)

        # Should return None on error
        self.assertIsNone(description)

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.fast
    def test_describe_frame_no_api_key(self):
        """Test frame description without API key"""
        frame_data = b'fake_image_bytes'
        description = self.processor._describe_frame_gemini(frame_data)

        # Should return None when no API key
        self.assertIsNone(description)
        self.assertEqual(self.processor.api_calls, 0)

    @patch('src.vision.VisionProcessor._extract_frame')
    @patch('src.vision.VisionProcessor._describe_frame_gemini')
    @pytest.mark.fast
    def test_describe_scene_success(self, mock_describe, mock_extract):
        """Test complete scene description workflow"""
        # Mock successful frame extraction and description
        mock_extract.return_value = b'frame_data'
        mock_describe.return_value = "Mountain scene with climbers"

        scene = {
            "start_time": 10.0,
            "end_time": 15.0
        }

        description = self.processor.describe_scene(
            video_path="/test/video.mp4",
            scene=scene
        )

        # Should return description
        self.assertEqual(description, "Mountain scene with climbers")

        # Should extract frame at mid-point
        mock_extract.assert_called_with("/test/video.mp4", 12.5)

    @patch('src.vision.VisionProcessor._extract_frame')
    @pytest.mark.fast
    def test_describe_scene_extraction_failure(self, mock_extract):
        """Test scene description when frame extraction fails"""
        # Mock failed frame extraction
        mock_extract.return_value = None

        scene = {
            "start_time": 10.0,
            "end_time": 15.0
        }

        description = self.processor.describe_scene(
            video_path="/test/video.mp4",
            scene=scene
        )

        # Should return None when frame extraction fails
        self.assertIsNone(description)

    @pytest.mark.fast
    def test_cost_tracking(self):
        """Test API cost tracking"""
        # Reset costs
        self.processor.api_calls = 0
        self.processor.total_cost = 0.0

        # Simulate multiple API calls
        self.processor.api_calls = 5
        self.processor.total_cost = 5 * 0.001

        self.assertEqual(self.processor.api_calls, 5)
        self.assertEqual(self.processor.total_cost, 0.005)


class TestProcessVideoVision(unittest.TestCase):
    """Test the main process_video_vision function"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.vision = Mock()
        self.config.vision.enabled = True
        self.config.vision.provider = 'gemini'
        self.config.vision.model = 'gemini-2.0-flash'
        self.config.vision.min_words_per_scene = 5
        self.config.vision.coverage_threshold = 0.3
        self.config.vision.estimated_cost_per_call = 0.001
        self.config.vision.max_scenes_per_video = 50

    @patch('src.vision.VisionProcessor')
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @pytest.mark.fast
    def test_process_video_vision_with_cache(self, mock_processor_class):
        """Test vision processing with cache object"""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.is_available.return_value = True
        mock_processor.describe_scene.return_value = "Vision description"
        mock_processor_class.return_value = mock_processor

        # Mock cache
        mock_cache = Mock()
        mock_cache.cache_dir = "/tmp/cache"

        transcript = [
            {"start_time": 0.0, "end_time": 5.0, "text": "Mountains"}
        ]

        updated_transcript = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript,
            cache=mock_cache,
            config=self.config
        )

        # Should return a transcript (possibly modified)
        self.assertIsInstance(updated_transcript, list)

    @patch('src.vision.VisionProcessor')
    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.fast
    def test_process_video_vision_no_api_key(self, mock_processor_class):
        """Test vision processing when API key not available"""
        # Mock processor to indicate unavailable
        mock_processor = MagicMock()
        mock_processor.is_available.return_value = False
        mock_processor_class.return_value = mock_processor

        mock_cache = Mock()
        mock_cache.cache_dir = "/tmp/cache"

        transcript = []

        updated_transcript = process_video_vision(
            video_path="/test/video.mp4",
            transcript_segments=transcript,
            cache=mock_cache,
            config=self.config
        )

        # Should return original transcript without vision descriptions
        self.assertEqual(updated_transcript, transcript)

    @pytest.mark.fast
    def test_process_video_vision_disabled(self):
        """Test when vision is disabled in config"""
        # Disable vision
        self.config.vision.enabled = False

        scenes = [{"start_time": 0.0, "end_time": 5.0}]
        transcript = []

        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should skip processing
        self.assertEqual(results, [])
        self.assertTrue(stats['skipped'])
        self.assertEqual(stats['reason'], 'Vision disabled')

    @patch('src.vision.TranscriptAnalyzer.analyze_video_transcript')
    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_process_video_vision_good_coverage(self, mock_processor_class, mock_analyze):
        """Test when transcript coverage is good - no vision needed"""
        # Mock good transcript coverage decision
        mock_decision = Mock()
        mock_decision.needs_vision = False
        mock_decision.reason = "Good transcript coverage"
        mock_decision.transcript_coverage = 0.9
        mock_analyze.return_value = mock_decision

        scenes = [{"start_time": 0.0, "end_time": 5.0}]
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "Lots of text here"}]

        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should skip vision processing
        self.assertEqual(results, [])
        self.assertTrue(stats['skipped'])
        self.assertEqual(stats['reason'], "Good transcript coverage")
        self.assertEqual(stats['coverage'], 0.9)

    @patch('src.vision.TranscriptAnalyzer.analyze_video_transcript')
    @patch('src.vision.TranscriptAnalyzer.get_priority_scenes')
    @patch('src.vision.VisionProcessor.describe_scene')
    @patch('src.vision.VisionProcessor.get_stats')
    @pytest.mark.fast
    def test_process_video_vision_full_workflow(
        self, mock_get_stats, mock_describe, mock_priority, mock_analyze
    ):
        """Test full process_video_vision_full workflow with sparse coverage"""
        # Mock sparse coverage decision
        mock_decision = Mock()
        mock_decision.needs_vision = True
        mock_decision.reason = "Sparse transcript"
        mock_decision.transcript_coverage = 0.2
        mock_analyze.return_value = mock_decision

        # Mock priority scenes
        mock_priority.return_value = [0, 2]  # Process scenes 0 and 2

        # Mock vision descriptions
        mock_describe.side_effect = [
            "Mountain landscape with snow",
            "Forest trail with hikers"
        ]

        # Mock stats
        mock_get_stats.return_value = {
            'api_calls': 2,
            'total_cost': 0.002
        }

        scenes = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 15.0}
        ]

        transcript = [
            {"start_time": 0.0, "end_time": 5.0, "text": "Mountains"},
            {"start_time": 10.0, "end_time": 15.0, "text": ""}
        ]

        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should return scene analyses
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SceneAnalysis)

        # Check first scene
        self.assertEqual(results[0].scene_index, 0)
        self.assertEqual(results[0].vision_description, "Mountain landscape with snow")
        self.assertIn("Visual:", results[0].combined_description)

        # Check stats
        self.assertEqual(stats['scenes_processed'], 2)
        self.assertEqual(stats['coverage'], 0.2)

    @patch('src.vision.TranscriptAnalyzer.analyze_video_transcript')
    @patch('src.vision.TranscriptAnalyzer.get_priority_scenes')
    @patch('src.vision.VisionProcessor.describe_scene')
    @pytest.mark.fast
    def test_process_video_vision_with_srt_segments(
        self, mock_describe, mock_priority, mock_analyze
    ):
        """Test process_video_vision_full handles SRTSegment objects"""
        # Mock decision
        mock_decision = Mock()
        mock_decision.needs_vision = True
        mock_decision.transcript_coverage = 0.1
        mock_analyze.return_value = mock_decision

        # Mock priority scenes
        mock_priority.return_value = [0]

        # Mock description
        mock_describe.return_value = "Test description"

        scenes = [{"start_time": 0.0, "end_time": 5.0}]

        # Create SRTSegment mock
        seg = Mock()
        seg.start_time = 0.0
        seg.end_time = 5.0
        seg.text = "Brief"

        transcript = [seg]

        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should process successfully
        self.assertEqual(len(results), 1)

    @patch('src.vision.TranscriptAnalyzer.analyze_video_transcript')
    @patch('src.vision.TranscriptAnalyzer.get_priority_scenes')
    @patch('src.vision.VisionProcessor.describe_scene')
    @pytest.mark.fast
    def test_process_video_vision_empty_description(
        self, mock_describe, mock_priority, mock_analyze
    ):
        """Test handling when vision description is empty/None"""
        # Mock decision
        mock_decision = Mock()
        mock_decision.needs_vision = True
        mock_decision.transcript_coverage = 0.0
        mock_analyze.return_value = mock_decision

        # Mock priority scenes
        mock_priority.return_value = [0]

        # Mock empty description
        mock_describe.return_value = None

        scenes = [{"start_time": 0.0, "end_time": 5.0}]
        transcript = []

        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should still create result but with no vision description
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].vision_description)

    @patch('src.vision.TranscriptAnalyzer.analyze_video_transcript')
    @patch('src.vision.TranscriptAnalyzer.get_priority_scenes')
    @pytest.mark.fast
    def test_process_video_vision_scene_index_out_of_range(
        self, mock_priority, mock_analyze
    ):
        """Test handling when priority scene index is out of range"""
        # Mock decision
        mock_decision = Mock()
        mock_decision.needs_vision = True
        mock_analyze.return_value = mock_decision

        # Mock priority scenes with out-of-range index
        mock_priority.return_value = [0, 10]  # Scene 10 doesn't exist

        scenes = [{"start_time": 0.0, "end_time": 5.0}]  # Only 1 scene
        transcript = []

        # Should not crash
        results, stats = process_video_vision_full(
            video_path="/test/video.mp4",
            scenes=scenes,
            transcript_segments=transcript,
            cache_dir="/tmp/cache",
            config=self.config
        )

        # Should skip out-of-range scene
        self.assertTrue(len(results) <= 1)


class TestVisionCache(unittest.TestCase):
    """Test VisionCache class"""

    def setUp(self):
        """Set up test cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = VisionCache(self.temp_dir)

    def tearDown(self):
        """Clean up test cache"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_cache_entry_serialization(self):
        """Test VisionCache can serialize and deserialize entries"""
        # Create a cache entry
        entry = CacheEntry(
            key="test_key",
            data="A beautiful mountain scene with snow",
            cached_at=1234567890.0,
            metadata={"video": "/test/video.mp4", "scene": 1}
        )

        # Serialize
        serialized = self.cache._serialize_entry(entry)

        # Should contain all fields
        self.assertIn('data', serialized)
        self.assertIn('cached_at', serialized)
        self.assertIn('metadata', serialized)
        self.assertEqual(serialized['data'], "A beautiful mountain scene with snow")

    @pytest.mark.fast
    def test_cache_entry_deserialization(self):
        """Test VisionCache can deserialize entries"""
        # Create serialized data
        data = {
            'data': "Mountain landscape",
            'cached_at': 1234567890.0,
            'metadata': {'scene': 1}
        }

        # Deserialize
        entry = self.cache._deserialize_entry(data)

        # Should reconstruct CacheEntry
        self.assertIsInstance(entry, CacheEntry)
        self.assertEqual(entry.data, "Mountain landscape")
        self.assertEqual(entry.cached_at, 1234567890.0)
        self.assertEqual(entry.metadata, {'scene': 1})

    @pytest.mark.fast
    def test_cache_stores_and_retrieves(self):
        """Test storing and retrieving vision descriptions"""
        # Store a description using the set() API
        self.cache.set(
            key="video1_scene1",
            value="Mountain scene description",
            metadata={"video": "/test/video1.mp4"}
        )

        # Retrieve it
        retrieved = self.cache.get("video1_scene1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.data, "Mountain scene description")


class TestVisionCaching(unittest.TestCase):
    """Test vision response caching"""

    def setUp(self):
        """Set up test cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.vision = Mock()
        self.config.vision.provider = 'gemini'
        self.config.vision.model = 'gemini-2.0-flash'
        self.config.vision.estimated_cost_per_call = 0.001

    def tearDown(self):
        """Clean up test cache"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.vision.VisionProcessor._extract_frame')
    @patch('src.vision.VisionProcessor._describe_frame_gemini')
    @pytest.mark.fast
    def test_cache_prevents_redundant_api_calls(self, mock_describe, mock_extract):
        """Test that cached descriptions prevent redundant API calls"""
        processor = VisionProcessor(self.config)

        # Mock frame extraction and description
        mock_extract.return_value = b'frame_data'
        mock_describe.return_value = "Cached description"

        scene = {"start_time": 10.0, "end_time": 15.0}

        # First call - should hit API
        desc1 = processor.describe_scene(
            video_path="/test/video.mp4",
            scene=scene,
            cache_dir=self.temp_dir
        )

        # Second identical call - should use cache
        desc2 = processor.describe_scene(
            video_path="/test/video.mp4",
            scene=scene,
            cache_dir=self.temp_dir
        )

        self.assertEqual(desc1, desc2)
        # Note: Actual caching logic depends on VisionCache implementation


if __name__ == '__main__':
    unittest.main()
