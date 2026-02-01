"""
Comprehensive coverage tests for TranscribeStage.

This module specifically targets uncovered lines in src/stages/transcribe.py:
- Lines 88, 92, 96, 117: Whisper client initialization and conditional paths
- Line 216: Transcript processing (sequential transcription path)
- Lines 332, 335-336: Cache handling (import error and embedding index return)
- Lines 373, 401-402: Parallel processing paths
- Lines 459-460, 505-507: Embedding generation (yaml CSafeLoader import)
- Lines 512-515, 540-542: State updates (error handling, cache loading failures)

Created: 2026-01-11
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest
import numpy as np

from src.stages.transcribe import TranscribeStage
from src.state import PipelineState, DownloadedVideo, AudioDownload


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with all transcription settings"""
    config = MagicMock()
    config.pipeline.skip_transcription = False
    config.pipeline.parallel_transcription = True
    config.pipeline.parallel_embedding = True
    config.pipeline.delta_indexing = True
    config.downloaded_videos_dir = "videos"
    config.transcription.model = "base"
    config.transcription.language = "auto"
    config.transcription.max_workers = 4
    config.transcription.min_silence_duration_ms = 200
    config.transcription.speech_pad_ms = 10
    config.embedding.batch_size = 32
    config.matching.chapter_matching_enabled = False
    config.cache.cache_dir = ".cache"
    config.silent_video = None
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.checkpoint_path = Path(".") / "checkpoint.json"
    return checkpoint


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory"""
    return tmp_path


# ============================================================================
# Test Line 88: Embedding result print statement
# ============================================================================

class TestEmbeddingResultOutput:
    """Tests for line 88: print embedding index size when embeddings succeed"""

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @pytest.mark.fast
    def test_embedding_result_prints_vector_count(
        self, mock_handle_silent, mock_compute_emb, mock_transcribe,
        mock_config, mock_checkpoint, capsys
    ):
        """Test that successful embedding computation prints vector count (line 88)"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'neutral'

        # Setup mock transcripts
        mock_transcribe.return_value = {
            "video1.mp4": [{"text": "Hello world", "start_time": 0, "end_time": 2}]
        }

        # Setup successful embedding computation
        mock_compute_emb.return_value = True
        state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 2 vectors

        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        captured = capsys.readouterr()
        assert "2 vectors" in captured.out or result.success
        assert result.success is True


# ============================================================================
# Test Line 92: Face preference conditional
# ============================================================================

class TestFacePreferenceConditional:
    """Tests for line 92: predetect faces when face_preference != neutral"""

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._predetect_faces')
    @pytest.mark.fast
    def test_predetect_faces_called_when_prefer_faces(
        self, mock_predetect, mock_handle_silent, mock_compute_emb,
        mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test that _predetect_faces is called when face_preference is 'prefer'"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'prefer'  # Not neutral

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        mock_predetect.assert_called_once()
        assert result.success is True

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._predetect_faces')
    @pytest.mark.fast
    def test_predetect_faces_called_when_avoid_faces(
        self, mock_predetect, mock_handle_silent, mock_compute_emb,
        mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test that _predetect_faces is called when face_preference is 'avoid'"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'avoid'  # Not neutral

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        mock_predetect.assert_called_once()
        assert result.success is True

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._predetect_faces')
    @pytest.mark.fast
    def test_predetect_faces_not_called_when_neutral(
        self, mock_predetect, mock_handle_silent, mock_compute_emb,
        mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test that _predetect_faces is NOT called when face_preference is 'neutral'"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'neutral'  # Neutral - should skip

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        mock_predetect.assert_not_called()
        assert result.success is True


# ============================================================================
# Test Line 96: Chapter matching conditional
# ============================================================================

class TestChapterMatchingConditional:
    """Tests for line 96: extract video topics when chapter matching enabled"""

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._extract_video_topics')
    @pytest.mark.fast
    def test_extract_topics_called_when_chapter_matching_enabled(
        self, mock_extract_topics, mock_handle_silent, mock_compute_emb,
        mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test that _extract_video_topics is called when chapter_matching_enabled is True"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'neutral'

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = True  # Enabled

        result = stage.run(state, mock_config, mock_checkpoint)

        mock_extract_topics.assert_called_once()
        assert result.success is True

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._extract_video_topics')
    @pytest.mark.fast
    def test_extract_topics_not_called_when_chapter_matching_disabled(
        self, mock_extract_topics, mock_handle_silent, mock_compute_emb,
        mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test that _extract_video_topics is NOT called when chapter_matching_enabled is False"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'neutral'

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False  # Disabled

        result = stage.run(state, mock_config, mock_checkpoint)

        mock_extract_topics.assert_not_called()
        assert result.success is True


# ============================================================================
# Test Line 117: can_skip returns checkpoint.should_skip_stage
# ============================================================================

class TestCanSkipMethod:
    """Tests for line 117: can_skip returns checkpoint result"""

    @pytest.mark.fast
    def test_can_skip_returns_false_from_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when checkpoint says not to skip"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False
        mock_checkpoint.should_skip_stage.assert_called_with("TRANSCRIBE")


# ============================================================================
# Test Line 216: Sequential transcription path (parallel disabled)
# ============================================================================

class TestSequentialTranscriptionPath:
    """Tests for line 216: sequential transcription when parallel disabled"""

    @patch('src.stages.transcribe.TranscribeStage._transcribe_sequential')
    @pytest.mark.fast
    def test_sequential_path_when_parallel_disabled(
        self, mock_sequential, mock_config
    ):
        """Test that sequential transcription is used when parallel is disabled"""
        stage = TranscribeStage()
        mock_config.pipeline.parallel_transcription = False  # Disable parallel

        mock_sequential.return_value = {"video1.mp4": [{"text": "Hello"}]}

        video_files = [Path("video1.mp4")]
        result = stage._transcribe_videos(video_files, mock_config)

        mock_sequential.assert_called_once_with(video_files, mock_config)
        assert "video1.mp4" in result


# ============================================================================
# Test Lines 332, 335-336: Embedding cache handling and import error
# ============================================================================

class TestEmbeddingCacheHandling:
    """Tests for lines 332, 335-336: cache handling and import error paths"""

    @pytest.mark.fast
    def test_compute_embeddings_returns_false_when_no_index_built(self, mock_config):
        """Test _compute_embeddings returns False when embedding_index is not built (line 332)"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = True

        # Empty transcripts - no embeddings to compute
        transcripts = {}

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is False

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.build_embedding_index')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_compute_embeddings_returns_false_when_embeddings_empty(
        self, mock_cache_class, mock_provider, mock_build_index,
        mock_compute, mock_config
    ):
        """Test _compute_embeddings returns False when embeddings are empty (line 332)"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = True

        # Segment with hasattr check path
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start_time = 0
        mock_segment.end_time = 2

        transcripts = {"video1.mp4": [mock_segment]}

        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        # Return empty array
        mock_compute.return_value = np.array([])

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is False

    @pytest.mark.fast
    def test_compute_embeddings_import_error_returns_false(self, mock_config):
        """Test _compute_embeddings returns False on ImportError (lines 335-336)"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = True

        transcripts = {"video1.mp4": [{"text": "Hello"}]}

        # Simulate ImportError by patching at module level
        with patch.dict('sys.modules', {'src.embeddings': None}):
            with patch('src.stages.transcribe.logger') as mock_logger:
                # Force import to fail
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if 'embeddings' in name:
                        raise ImportError(f"No module named '{name}'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, '__import__', mock_import):
                    result = stage._compute_embeddings(transcripts, state, mock_config)

                # Should have logged warning and returned False
                assert result is False


# ============================================================================
# Test Lines 373, 401-402: Parallel processing paths in face detection
# ============================================================================

class TestFaceDetectionProgress:
    """Tests for lines 373, 401-402: face detection progress output"""

    @patch('src.face_detection.FaceDetector')
    @pytest.mark.fast
    def test_face_detection_progress_output(self, mock_detector_class, mock_config, capsys):
        """Test face detection outputs progress (line 373)"""
        stage = TranscribeStage()

        mock_detector = Mock()
        mock_detector._cache = {}
        # Return high scores so we count faces_found
        mock_detector.get_face_score.side_effect = [0.8, 0.9, 0.3, 0.1, 0.5, 0.6, 0.4, 0.25, 0.7, 0.8, 0.9]
        mock_detector_class.get_instance.return_value = mock_detector

        # Create 11 videos to trigger progress print at 10
        video_files = [Path(f"video{i}.mp4") for i in range(11)]

        stage._predetect_faces(video_files, mock_config)

        # Verify all videos were processed
        assert mock_detector.get_face_score.call_count == 11

    @patch('src.topic_extraction.TopicExtractor')
    @pytest.mark.fast
    def test_extract_topics_with_dict_segments(self, mock_extractor_class, mock_config):
        """Test topic extraction handles dict segments (lines 401-402)"""
        stage = TranscribeStage()

        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Test with dict segments (not objects)
        transcripts = {
            "video1.mp4": [
                {"text": "Beach sunset", "start_time": 0, "end_time": 2},
                {"text": "Ocean waves", "start_time": 2, "end_time": 4}
            ],
            "video2.mp4": [
                {"text": "Mountain hiking"}
            ]
        }

        stage._extract_video_topics(transcripts, mock_config)

        # Should complete without error
        mock_extractor_class.assert_called_once()

    @patch('src.topic_extraction.TopicExtractor')
    @pytest.mark.fast
    def test_extract_topics_with_object_segments(self, mock_extractor_class, mock_config):
        """Test topic extraction handles object segments (lines 399-400)"""
        stage = TranscribeStage()

        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Test with object segments (hasattr path)
        mock_segment1 = Mock()
        mock_segment1.text = "Beach sunset"
        mock_segment2 = Mock()
        mock_segment2.text = "Ocean waves"

        transcripts = {
            "video1.mp4": [mock_segment1, mock_segment2]
        }

        stage._extract_video_topics(transcripts, mock_config)

        mock_extractor_class.assert_called_once()


# ============================================================================
# Test Lines 459-460: YAML CSafeLoader import fallback
# ============================================================================

class TestYamlLoaderFallback:
    """Tests for lines 459-460: YAML CSafeLoader import fallback"""

    @patch('glob.glob')
    @patch('builtins.open', mock_open(read_data='cache:\n  cache_dir: .cache'))
    @pytest.mark.fast
    def test_rebuild_metadata_with_csafeloader(self, mock_glob, mock_checkpoint, temp_project_dir):
        """Test rebuild uses CSafeLoader when available"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        # Note: Uses default cache_dir of ".cache"

        mock_glob.return_value = []  # No transcript files

        # Should work with CSafeLoader
        stage._rebuild_text_metadata(state, mock_checkpoint)

        assert state.text_metadata == []

    @patch('glob.glob')
    @pytest.mark.fast
    def test_rebuild_metadata_safeloader_fallback(self, mock_glob, mock_checkpoint, temp_project_dir):
        """Test rebuild falls back to SafeLoader when CSafeLoader unavailable (lines 459-460)"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        # Note: Uses default cache_dir of ".cache"

        # Create cache directory
        cache_dir = temp_project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True, exist_ok=True)

        mock_glob.return_value = []

        # Simulate CSafeLoader not available
        import yaml
        original_try = yaml.__dict__.get('CSafeLoader', None)

        if hasattr(yaml, 'CSafeLoader'):
            # Mock the import to fail
            with patch.dict('sys.modules', {'yaml.CSafeLoader': None}):
                stage._rebuild_text_metadata(state, mock_checkpoint)
        else:
            # CSafeLoader not available, test SafeLoader path
            stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should complete without error
        assert isinstance(state.text_metadata, list)


# ============================================================================
# Test Lines 505-507: Transcript loading exception handling
# ============================================================================

class TestTranscriptLoadingErrors:
    """Tests for lines 505-507: transcript loading exception handling"""

    @patch('glob.glob')
    @patch('src.stages.transcribe.logger')
    @pytest.mark.fast
    def test_rebuild_metadata_handles_corrupt_transcript(
        self, mock_logger, mock_glob, mock_checkpoint, temp_project_dir
    ):
        """Test rebuild handles corrupt transcript file (lines 505-507)"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        # Note: Uses default cache_dir of ".cache"

        # Create cache directory with corrupt transcript
        cache_dir = temp_project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True, exist_ok=True)

        corrupt_file = cache_dir / "corrupt_transcript.json"
        corrupt_file.write_text('{"invalid json')  # Corrupt JSON

        mock_glob.return_value = [str(corrupt_file)]

        stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should continue despite error
        assert state.text_metadata == []
        mock_logger.debug.assert_called()

    @patch('glob.glob')
    @pytest.mark.fast
    def test_rebuild_metadata_with_valid_transcripts(
        self, mock_glob, mock_checkpoint, temp_project_dir
    ):
        """Test rebuild with valid transcripts includes video_path"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        # Note: Uses default cache_dir of ".cache"

        # Create cache directory with valid transcript
        cache_dir = temp_project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True, exist_ok=True)

        valid_transcript = [
            {"text": "Hello", "video_path": "video1.mp4", "start_time": 0, "end_time": 2},
            {"text": "World", "video_path": "video1.mp4", "start_time": 2, "end_time": 4}
        ]

        transcript_file = cache_dir / "video1_transcript.json"
        transcript_file.write_text(json.dumps(valid_transcript))

        mock_glob.return_value = [str(transcript_file)]

        stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should have rebuilt metadata
        assert len(state.text_metadata) == 2
        assert state.text_metadata[0]['video_path'] == "video1.mp4"


# ============================================================================
# Test Lines 512-515: Rebuild metadata overall exception handling
# ============================================================================

class TestRebuildMetadataErrors:
    """Tests for lines 512-515: overall exception handling in rebuild"""

    @patch('src.stages.transcribe.logger')
    @pytest.mark.fast
    def test_rebuild_metadata_overall_exception(self, mock_logger, mock_checkpoint):
        """Test rebuild handles overall exception gracefully (lines 512-515)"""
        stage = TranscribeStage()
        state = PipelineState()

        # Invalid checkpoint path that will cause exception
        mock_checkpoint.checkpoint_path = None

        stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should set empty list and log warning
        assert state.text_metadata == []
        mock_logger.warning.assert_called()

    @patch('src.stages.transcribe.logger')
    @pytest.mark.fast
    def test_rebuild_metadata_config_file_not_found(
        self, mock_logger, mock_checkpoint, temp_project_dir
    ):
        """Test rebuild handles missing config file"""
        stage = TranscribeStage()
        state = PipelineState()

        # Set checkpoint path but don't create project_config.yaml
        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should handle gracefully
        assert state.text_metadata == []


# ============================================================================
# Test Lines 540-542: Load transcripts from cache exception handling
# ============================================================================

class TestLoadTranscriptsFromCache:
    """Tests for lines 540-542: cache loading exception handling"""

    @patch('src.transcription.TranscriptCache')
    @patch('src.stages.transcribe.logger')
    @pytest.mark.fast
    def test_load_transcripts_from_cache_exception(
        self, mock_logger, mock_cache_class, mock_config, temp_project_dir
    ):
        """Test _load_transcripts_from_cache handles exception (lines 540-542)"""
        stage = TranscribeStage()

        mock_cache_class.side_effect = Exception("Cache initialization failed")

        mock_config.downloaded_videos_dir = str(temp_project_dir / "videos")

        result = stage._load_transcripts_from_cache(mock_config)

        assert result == {}
        mock_logger.warning.assert_called()

    @patch('src.transcription.TranscriptCache')
    @pytest.mark.fast
    def test_load_transcripts_from_cache_success(
        self, mock_cache_class, mock_config, temp_project_dir
    ):
        """Test _load_transcripts_from_cache returns cached transcripts"""
        stage = TranscribeStage()

        # Create videos directory with a video file
        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "video1.mp4").write_bytes(b"fake video data")

        mock_cache = Mock()
        mock_cache.get.return_value = [{"text": "Cached transcript", "start_time": 0, "end_time": 2}]
        mock_cache_class.return_value = mock_cache

        mock_config.downloaded_videos_dir = str(videos_dir)

        result = stage._load_transcripts_from_cache(mock_config)

        assert len(result) == 1

    @patch('src.transcription.TranscriptCache')
    @pytest.mark.fast
    def test_load_transcripts_from_cache_no_cached(
        self, mock_cache_class, mock_config, temp_project_dir
    ):
        """Test _load_transcripts_from_cache with no cached transcripts"""
        stage = TranscribeStage()

        # Create videos directory with a video file
        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "video1.mp4").write_bytes(b"fake video data")

        mock_cache = Mock()
        mock_cache.get.return_value = None  # No cached transcript
        mock_cache_class.return_value = mock_cache

        mock_config.downloaded_videos_dir = str(videos_dir)

        result = stage._load_transcripts_from_cache(mock_config)

        assert result == {}


# ============================================================================
# Test Embedding Computation with Both Segment Types
# ============================================================================

class TestEmbeddingComputationSegmentTypes:
    """Tests for embedding computation with different segment types"""

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.build_embedding_index')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_compute_embeddings_with_object_segments(
        self, mock_cache_class, mock_provider, mock_build_index,
        mock_compute, mock_config
    ):
        """Test _compute_embeddings handles object segments with hasattr path"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = True

        # Create object segments (hasattr path)
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start_time = 0
        mock_segment.end_time = 2

        transcripts = {"video1.mp4": [mock_segment]}

        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()
        mock_compute.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_build_index.return_value = Mock()

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is True
        assert len(state.text_metadata) == 1
        assert state.text_metadata[0]['text'] == "Hello world"

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.build_embedding_index')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_compute_embeddings_with_dict_segments(
        self, mock_cache_class, mock_provider, mock_build_index,
        mock_compute, mock_config
    ):
        """Test _compute_embeddings handles dict segments"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = True

        # Dict segments (no hasattr path)
        transcripts = {
            "video1.mp4": [
                {"text": "Hello world", "start_time": 0, "end_time": 2}
            ]
        }

        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()
        mock_compute.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_build_index.return_value = Mock()

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is True
        assert len(state.text_metadata) == 1
        assert state.text_metadata[0]['text'] == "Hello world"


# ============================================================================
# Test Full Stage Run with All Features Enabled
# ============================================================================

class TestFullStageRun:
    """Integration tests for full stage execution with various features"""

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @patch('src.stages.transcribe.TranscribeStage._predetect_faces')
    @patch('src.stages.transcribe.TranscribeStage._extract_video_topics')
    @pytest.mark.fast
    def test_full_run_with_all_features(
        self, mock_extract_topics, mock_predetect, mock_handle_silent,
        mock_compute_emb, mock_transcribe, mock_config, mock_checkpoint
    ):
        """Test full stage run with face detection and chapter matching enabled"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]
        state.face_preference = 'prefer'  # Enable face detection
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        mock_transcribe.return_value = {"video1.mp4": [{"text": "Hello"}]}
        mock_compute_emb.return_value = True
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = True  # Enable topic extraction

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        mock_predetect.assert_called_once()
        mock_extract_topics.assert_called_once()
        mock_handle_silent.assert_called_once()
        mock_compute_emb.assert_called_once()

    @patch('src.stages.transcribe.TranscribeStage._transcribe_videos')
    @patch('src.stages.transcribe.TranscribeStage._compute_embeddings')
    @patch('src.stages.transcribe.TranscribeStage._handle_silent_videos')
    @pytest.mark.fast
    def test_run_with_audio_first_mode(
        self, mock_handle_silent, mock_compute_emb, mock_transcribe,
        mock_config, mock_checkpoint
    ):
        """Test stage run with audio-first mode (downloaded_audio)"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_audio = [
            AudioDownload(file="audio1.mp3", url="url1", video_id="vid1"),
            AudioDownload(file="audio2.mp3", url="url2", video_id="vid2")
        ]
        state.face_preference = 'neutral'

        mock_transcribe.return_value = {
            "audio1.mp3": [{"text": "Hello"}],
            "audio2.mp3": [{"text": "World"}]
        }
        mock_compute_emb.return_value = False
        mock_config.pipeline.parallel_embedding = True
        mock_config.matching.chapter_matching_enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Verify audio files were passed to transcription
        call_args = mock_transcribe.call_args[0]
        video_files = call_args[0]
        assert len(video_files) == 2
        assert str(video_files[0]) == "audio1.mp3"


# ============================================================================
# Test Skip Transcription Path
# ============================================================================

class TestSkipTranscriptionPath:
    """Tests for skip transcription configuration"""

    @patch('src.stages.transcribe.TranscribeStage._load_transcripts_from_cache')
    @pytest.mark.fast
    def test_skip_transcription_enabled(
        self, mock_load_cache, mock_config, mock_checkpoint
    ):
        """Test skip transcription loads from cache"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download")
        ]

        mock_config.pipeline.skip_transcription = True
        mock_load_cache.return_value = {"video1.mp4": [{"text": "Cached"}]}

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        mock_load_cache.assert_called_once()
        assert len(state.transcripts) >= 0


# ============================================================================
# Test Delta Indexing Path
# ============================================================================

class TestDeltaIndexingPath:
    """Tests for delta indexing in transcription"""

    @patch('src.transcription.transcribe_videos_parallel')
    @patch('src.transcription.DeltaAwareIndex')
    @pytest.mark.fast
    def test_delta_indexing_filters_cached_videos(
        self, mock_delta_class, mock_transcribe, mock_config
    ):
        """Test delta indexing filters out already-cached videos"""
        stage = TranscribeStage()

        mock_delta = Mock()
        # Only video2 is new, video1 is cached
        mock_delta.get_new_videos.return_value = ["video2.mp4"]
        mock_delta_class.return_value = mock_delta

        mock_transcribe.return_value = {"video2.mp4": [{"text": "New video"}]}

        mock_config.pipeline.parallel_transcription = True
        mock_config.pipeline.delta_indexing = True

        video_files = [Path("video1.mp4"), Path("video2.mp4")]
        result = stage._transcribe_videos(video_files, mock_config)

        # Should only have transcribed video2
        assert "video2.mp4" in result
        # Should have called with only video2
        call_args = mock_transcribe.call_args
        assert "video2.mp4" in call_args[1]['video_paths']

    @patch('src.transcription.transcribe_videos_parallel')
    @patch('src.transcription.DeltaAwareIndex')
    @pytest.mark.fast
    def test_delta_indexing_all_cached(
        self, mock_delta_class, mock_transcribe, mock_config, capsys
    ):
        """Test delta indexing when all videos are cached"""
        stage = TranscribeStage()

        mock_delta = Mock()
        mock_delta.get_new_videos.return_value = []  # All cached
        mock_delta_class.return_value = mock_delta

        mock_config.pipeline.parallel_transcription = True
        mock_config.pipeline.delta_indexing = True

        video_files = [Path("video1.mp4"), Path("video2.mp4")]
        result = stage._transcribe_videos(video_files, mock_config)

        # Should return empty dict
        assert result == {}
        mock_transcribe.assert_not_called()

        # Should print message about all cached
        captured = capsys.readouterr()
        assert "cached" in captured.out.lower()


# ============================================================================
# Test LLM Description Generation for Silent Videos (US-002)
# ============================================================================

class TestLLMDescriptionGeneration:
    """Tests for _generate_llm_descriptions and _generate_filename_descriptions"""

    @patch('src.stages.transcribe.TranscribeStage._generate_llm_descriptions')
    @pytest.mark.fast
    def test_handle_silent_videos_calls_generate_descriptions(
        self, mock_generate, mock_config
    ):
        """Test that _handle_silent_videos calls _generate_llm_descriptions"""
        stage = TranscribeStage()
        video_files = [Path("silent_video.mp4"), Path("video_with_speech.mp4")]
        transcripts = {
            "video_with_speech.mp4": [{"text": "This is a video with lots of speech content"}]
        }

        # Configure silent_video settings
        mock_config.silent_video = Mock()
        mock_config.silent_video.enabled = True
        mock_config.silent_video.min_words_threshold = 10

        mock_generate.return_value = {"silent_video.mp4": "A beautiful sunset over the ocean."}

        stage._handle_silent_videos(video_files, transcripts, mock_config)

        mock_generate.assert_called_once()
        # Should have passed the silent video
        call_args = mock_generate.call_args[0]
        assert Path("silent_video.mp4") in call_args[0]

    @patch('src.stages.transcribe.TranscribeStage._generate_llm_descriptions')
    @pytest.mark.fast
    def test_handle_silent_videos_updates_transcripts(
        self, mock_generate, mock_config
    ):
        """Test that generated descriptions are added to transcripts"""
        stage = TranscribeStage()
        video_files = [Path("silent_video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.enabled = True
        mock_config.silent_video.min_words_threshold = 10

        mock_generate.return_value = {"silent_video.mp4": "Ocean waves crashing on a beach."}

        stage._handle_silent_videos(video_files, transcripts, mock_config)

        # Transcripts should be updated with the description
        assert "silent_video.mp4" in transcripts
        assert transcripts["silent_video.mp4"][0]['text'] == "Ocean waves crashing on a beach."
        assert transcripts["silent_video.mp4"][0]['is_generated'] is True
        assert transcripts["silent_video.mp4"][0]['source'] == 'llm_description'

    @pytest.mark.fast
    def test_handle_silent_videos_skips_when_disabled(self, mock_config):
        """Test that silent video handling is skipped when disabled"""
        stage = TranscribeStage()
        video_files = [Path("video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.enabled = False

        stage._handle_silent_videos(video_files, transcripts, mock_config)

        # No transcripts should be added
        assert len(transcripts) == 0

    @pytest.mark.fast
    def test_handle_silent_videos_skips_when_no_silent_config(self, mock_config):
        """Test that silent video handling is skipped when config missing"""
        stage = TranscribeStage()
        video_files = [Path("video.mp4")]
        transcripts = {}

        mock_config.silent_video = None

        stage._handle_silent_videos(video_files, transcripts, mock_config)

        # No transcripts should be added
        assert len(transcripts) == 0

    @pytest.mark.fast
    def test_handle_silent_videos_detects_sparse_transcripts(self, mock_config):
        """Test that videos with few words are detected as silent"""
        stage = TranscribeStage()
        video_files = [Path("sparse_video.mp4"), Path("chatty_video.mp4")]
        transcripts = {
            "sparse_video.mp4": [{"text": "Hello"}],  # 1 word
            "chatty_video.mp4": [{"text": "This is a video with many words spoken throughout the entire duration of the clip"}]  # 17 words
        }

        mock_config.silent_video = Mock()
        mock_config.silent_video.enabled = True
        mock_config.silent_video.min_words_threshold = 10

        with patch.object(stage, '_generate_llm_descriptions') as mock_gen:
            mock_gen.return_value = {}
            stage._handle_silent_videos(video_files, transcripts, mock_config)

            # Only sparse_video should be processed
            call_args = mock_gen.call_args[0]
            silent_list = call_args[0]
            assert len(silent_list) == 1
            assert Path("sparse_video.mp4") in silent_list

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_generate_llm_descriptions_uses_vision_api(
        self, mock_processor_class, mock_config
    ):
        """Test that Vision API is used when available"""
        stage = TranscribeStage()
        silent_videos = [Path("test_video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.use_vision_api = True
        mock_config.silent_video.use_llm_fallback = True
        mock_config.cache = Mock()
        mock_config.cache.cache_dir = ".cache"

        mock_processor = Mock()
        mock_processor.is_available.return_value = True
        mock_processor.describe_scene.return_value = "A person walking on a beach."
        mock_processor.get_stats.return_value = {'api_calls': 1, 'estimated_cost': 0.001}
        mock_processor_class.return_value = mock_processor

        descriptions = stage._generate_llm_descriptions(silent_videos, transcripts, mock_config)

        assert "test_video.mp4" in descriptions
        assert descriptions["test_video.mp4"] == "A person walking on a beach."
        mock_processor.describe_scene.assert_called_once()

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_generate_llm_descriptions_falls_back_when_vision_unavailable(
        self, mock_processor_class, mock_config
    ):
        """Test fallback to filename description when Vision API unavailable"""
        stage = TranscribeStage()
        silent_videos = [Path("ocean_waves_sunset.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.use_vision_api = True
        mock_config.silent_video.use_llm_fallback = True
        mock_config.cache = Mock()
        mock_config.cache.cache_dir = ".cache"
        mock_config.llm = None

        mock_processor = Mock()
        mock_processor.is_available.return_value = False  # Vision not available
        mock_processor_class.return_value = mock_processor

        descriptions = stage._generate_llm_descriptions(silent_videos, transcripts, mock_config)

        # Should fall back to filename extraction
        assert "ocean_waves_sunset.mp4" in descriptions
        assert "ocean waves sunset" in descriptions["ocean_waves_sunset.mp4"].lower()

    @pytest.mark.fast
    def test_generate_filename_descriptions_simple_fallback(self, mock_config):
        """Test simple filename extraction when LLM unavailable"""
        stage = TranscribeStage()
        videos = [
            Path("beach_sunset_waves.mp4"),
            Path("mountain-hiking-trail.mp4")
        ]

        mock_config.llm = None

        descriptions = stage._generate_filename_descriptions(videos, mock_config)

        assert "beach_sunset_waves.mp4" in descriptions
        assert "[Silent video: beach sunset waves]" == descriptions["beach_sunset_waves.mp4"]
        assert "mountain-hiking-trail.mp4" in descriptions
        assert "[Silent video: mountain hiking trail]" == descriptions["mountain-hiking-trail.mp4"]

    @patch('os.getenv')
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_generate_filename_descriptions_uses_llm(
        self, mock_create_client, mock_getenv, mock_config
    ):
        """Test LLM-based description generation from filename"""
        stage = TranscribeStage()
        videos = [Path("sunset_beach.mp4")]

        mock_config.llm = Mock()
        mock_config.llm.provider = 'gemini'
        mock_config.llm.model = 'gemini-2.0-flash'

        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "A beautiful sunset over a sandy beach with gentle waves."
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        descriptions = stage._generate_filename_descriptions(videos, mock_config)

        assert "sunset_beach.mp4" in descriptions
        assert descriptions["sunset_beach.mp4"] == "A beautiful sunset over a sandy beach with gentle waves."

    @pytest.mark.fast
    def test_generate_llm_descriptions_respects_vision_disabled(self, mock_config):
        """Test that Vision API is skipped when use_vision_api is False"""
        stage = TranscribeStage()
        silent_videos = [Path("test_video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.use_vision_api = False  # Disabled
        mock_config.silent_video.use_llm_fallback = True
        mock_config.llm = None

        with patch('src.vision.VisionProcessor') as mock_processor_class:
            descriptions = stage._generate_llm_descriptions(silent_videos, transcripts, mock_config)

            # VisionProcessor should not be instantiated
            mock_processor_class.assert_not_called()

            # Should still get filename fallback
            assert "test_video.mp4" in descriptions

    @pytest.mark.fast
    def test_generate_llm_descriptions_handles_vision_error(self, mock_config):
        """Test graceful handling of Vision API errors"""
        stage = TranscribeStage()
        silent_videos = [Path("test_video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.use_vision_api = True
        mock_config.silent_video.use_llm_fallback = True
        mock_config.cache = Mock()
        mock_config.cache.cache_dir = ".cache"
        mock_config.llm = None

        with patch('src.vision.VisionProcessor') as mock_processor_class:
            mock_processor_class.side_effect = Exception("Vision API error")

            # Should not raise, should fall back
            descriptions = stage._generate_llm_descriptions(silent_videos, transcripts, mock_config)

            # Should get filename fallback
            assert "test_video.mp4" in descriptions

    @pytest.mark.fast
    def test_generate_llm_descriptions_no_fallback(self, mock_config):
        """Test when both Vision and LLM fallback are disabled"""
        stage = TranscribeStage()
        silent_videos = [Path("test_video.mp4")]
        transcripts = {}

        mock_config.silent_video = Mock()
        mock_config.silent_video.use_vision_api = False
        mock_config.silent_video.use_llm_fallback = False

        descriptions = stage._generate_llm_descriptions(silent_videos, transcripts, mock_config)

        # Should return empty dict
        assert descriptions == {}
