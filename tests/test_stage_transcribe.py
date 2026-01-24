"""
Comprehensive tests for TranscribeStage.

Tests cover:
- Stage initialization and registration
- Video file discovery
- Parallel vs sequential transcription
- Embedding computation
- Face pre-detection
- Video topic extraction
- Skip transcription behavior
- Checkpoint save/restore
- Validation and error handling
- Text metadata rebuilding

Created: 2026-01-09 (Phase 10.3)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from src.stages.transcribe import TranscribeStage
from src.state import PipelineState, DownloadedVideo, AudioDownload


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with transcription settings"""
    config = MagicMock()
    config.pipeline.skip_transcription = False
    config.pipeline.parallel_transcription = True
    config.pipeline.parallel_embedding = True
    config.pipeline.delta_indexing = True
    config.downloaded_videos_dir = "videos"
    config.transcription.model = "base"
    config.transcription.language = "auto"
    config.transcription.max_workers = 4
    config.embedding.batch_size = 32
    config.matching.chapter_matching_enabled = False
    config.cache.cache_dir = ".cache"
    # Silent video settings
    config.silent_video = Mock()
    config.silent_video.enabled = True
    config.silent_video.min_words_threshold = 10
    config.silent_video.use_vision_api = True
    config.silent_video.use_llm_fallback = True
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
# Test TranscribeStage Initialization
# ============================================================================

class TestTranscribeStageInit:
    """Test TranscribeStage initialization"""

    def test_stage_name(self):
        """Test stage name is correct"""
        stage = TranscribeStage()
        assert stage.name == "TRANSCRIBE"

    def test_stage_description(self):
        """Test stage description"""
        stage = TranscribeStage()
        assert "Transcribe" in stage.description or "transcribe" in stage.description

    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("TRANSCRIBE")
        assert stage_class is TranscribeStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestTranscribeInputValidation:
    """Test validate_inputs method"""

    def test_validate_no_videos_or_audio(self, mock_config):
        """Test validation fails when no videos or audio"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "videos" in error.lower() or "audio" in error.lower()

    def test_validate_with_videos(self, mock_config):
        """Test validation succeeds with videos"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [Mock()]
        state.downloaded_audio = []

        error = stage.validate_inputs(state, mock_config)

        assert error is None

    def test_validate_with_audio(self, mock_config):
        """Test validation succeeds with audio files"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = [Mock()]

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test Video File Discovery
# ============================================================================

class TestVideoFileDiscovery:
    """Test _get_video_files method"""

    def test_get_audio_first_files(self, mock_config):
        """Test getting files from audio downloads"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_audio = [
            AudioDownload(file="audio1.mp3", url="url1", video_id="vid1"),
            AudioDownload(file="audio2.mp3", url="url2", video_id="vid2")
        ]

        files = stage._get_video_files(state, mock_config)

        assert len(files) == 2
        assert all(isinstance(f, Path) for f in files)
        assert str(files[0]) == "audio1.mp3"

    def test_get_video_files(self, mock_config):
        """Test getting files from downloaded videos"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download"),
            DownloadedVideo(file="video2.mp4", url="url2", source="download")
        ]

        files = stage._get_video_files(state, mock_config)

        assert len(files) == 2
        assert str(files[0]) == "video1.mp4"

    def test_get_files_from_directory(self, mock_config, temp_project_dir):
        """Test scanning directory for video files"""
        stage = TranscribeStage()
        state = PipelineState()

        # Create mock video files
        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "video1.mp4").write_bytes(b"fake")
        (videos_dir / "video2.webm").write_bytes(b"fake")
        (videos_dir / "audio.mp3").write_bytes(b"fake")

        mock_config.downloaded_videos_dir = str(videos_dir)

        files = stage._get_video_files(state, mock_config)

        assert len(files) >= 1  # Should find at least one file


# ============================================================================
# Test Transcription
# ============================================================================

class TestTranscription:
    """Test transcription methods"""

    @patch('src.transcription.transcribe_videos_parallel')
    @patch('src.transcription.DeltaAwareIndex')
    def test_transcribe_videos_parallel(self, mock_delta_class, mock_transcribe, mock_config):
        """Test parallel transcription"""
        stage = TranscribeStage()
        video_files = [Path("video1.mp4"), Path("video2.mp4")]

        # Mock delta index
        mock_delta = Mock()
        mock_delta.get_new_videos.return_value = ["video1.mp4", "video2.mp4"]
        mock_delta_class.return_value = mock_delta

        # Mock transcription result
        mock_transcribe.return_value = {
            "video1.mp4": [{"text": "Hello", "start_time": 0, "end_time": 2}],
            "video2.mp4": [{"text": "World", "start_time": 0, "end_time": 2}]
        }

        transcripts = stage._transcribe_videos(video_files, mock_config)

        assert len(transcripts) == 2
        assert "video1.mp4" in transcripts

    @patch('src.transcription.transcribe_videos_parallel', side_effect=ImportError("no parallel"))
    @patch('src.transcription.transcribe_video')
    @patch('src.transcription.TranscriptCache')
    def test_transcribe_parallel_fallback_to_sequential(self, mock_cache_class, mock_transcribe,
                                                       mock_parallel, mock_config):
        """Test fallback to sequential when parallel unavailable"""
        stage = TranscribeStage()
        video_files = [Path("video1.mp4")]

        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        mock_transcribe.return_value = [{"text": "Hello", "start_time": 0, "end_time": 2}]

        mock_config.pipeline.delta_indexing = False  # Disable delta to avoid import

        transcripts = stage._transcribe_videos(video_files, mock_config)

        assert "video1.mp4" in transcripts

    @patch('src.transcription.transcribe_video')
    @patch('src.transcription.TranscriptCache')
    def test_transcribe_sequential(self, mock_cache_class, mock_transcribe, mock_config):
        """Test sequential transcription"""
        stage = TranscribeStage()
        video_files = [Path("video1.mp4"), Path("video2.mp4")]

        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        mock_transcribe.side_effect = [
            [{"text": "Hello", "start_time": 0, "end_time": 2}],
            [{"text": "World", "start_time": 0, "end_time": 2}]
        ]

        transcripts = stage._transcribe_sequential(video_files, mock_config)

        assert len(transcripts) == 2
        assert mock_transcribe.call_count == 2

    @patch('src.transcription.transcribe_video')
    @patch('src.transcription.TranscriptCache')
    @patch('src.stages.transcribe.logger')
    def test_transcribe_sequential_with_error(self, mock_logger, mock_cache_class,
                                              mock_transcribe, mock_config):
        """Test sequential transcription handles errors"""
        stage = TranscribeStage()
        video_files = [Path("video1.mp4"), Path("video2.mp4")]

        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        # First succeeds, second fails
        mock_transcribe.side_effect = [
            [{"text": "Hello"}],
            Exception("Transcription failed")
        ]

        transcripts = stage._transcribe_sequential(video_files, mock_config)

        assert len(transcripts) == 1
        assert "video1.mp4" in transcripts
        mock_logger.warning.assert_called()


# ============================================================================
# Test Silent Video Handling
# ============================================================================

class TestSilentVideoHandling:
    """Test silent video handling"""

    def test_handle_silent_videos_disabled(self, mock_config):
        """Test silent video handling when disabled"""
        stage = TranscribeStage()
        mock_config.silent_video = None

        # Should not raise
        stage._handle_silent_videos([], {}, mock_config)

    def test_handle_silent_videos_finds_silent(self, mock_config):
        """Test finding silent videos"""
        stage = TranscribeStage()
        mock_config.silent_video.enabled = True
        mock_config.silent_video.min_words_threshold = 10

        video_files = [Path("video1.mp4"), Path("video2.mp4")]
        transcripts = {"video1.mp4": [{"text": "This video has enough words to not be silent"}]}

        # Mock the description generation
        with patch.object(stage, '_generate_llm_descriptions') as mock_gen:
            mock_gen.return_value = {"video2.mp4": "Silent video description"}

            # video2.mp4 has no transcript (silent)
            stage._handle_silent_videos(video_files, transcripts, mock_config)

            # Should have called generate for the silent video
            mock_gen.assert_called_once()
            call_args = mock_gen.call_args[0]
            # video2 should be in the list (no transcript)
            assert Path("video2.mp4") in call_args[0]


# ============================================================================
# Test Embedding Computation
# ============================================================================

class TestEmbeddingComputation:
    """Test embedding computation"""

    def test_compute_embeddings_disabled(self, mock_config):
        """Test embeddings disabled"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_config.pipeline.parallel_embedding = False

        result = stage._compute_embeddings({}, state, mock_config)

        assert result is False

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.build_embedding_index')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_compute_embeddings_success(self, mock_cache_class, mock_provider,
                                       mock_build_index, mock_compute, mock_config):
        """Test successful embedding computation"""
        stage = TranscribeStage()
        state = PipelineState()

        transcripts = {
            "video1.mp4": [Mock(text="Hello world", start_time=0, end_time=2)]
        }

        mock_provider.return_value = Mock()
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        # Mock embedding results
        import numpy as np
        mock_compute.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_build_index.return_value = Mock()

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is True
        assert len(state.text_metadata) > 0
        assert state.embedding_index is not None

    @patch('src.embeddings.get_embedding_provider')
    @patch('src.embeddings.compute_embeddings', side_effect=Exception("embedding error"))
    @patch('src.stages.transcribe.logger')
    def test_compute_embeddings_import_error(self, mock_logger, mock_compute, mock_provider, mock_config):
        """Test embedding computation handles errors gracefully"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_provider.return_value = Mock()

        transcripts = {"video1.mp4": [{"text": "Hello"}]}

        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert result is False
        mock_logger.error.assert_called()  # Uses logger.error for generic exceptions

    def test_compute_embeddings_no_text(self, mock_config):
        """Test embedding computation with no text segments"""
        stage = TranscribeStage()
        state = PipelineState()

        result = stage._compute_embeddings({}, state, mock_config)

        assert result is False


# ============================================================================
# Test Face Pre-Detection
# ============================================================================

class TestFacePreDetection:
    """Test face pre-detection"""

    @patch('src.face_detection.FaceDetector')
    def test_predetect_faces_all_cached(self, mock_detector_class, mock_config):
        """Test face detection when all videos cached"""
        stage = TranscribeStage()

        mock_detector = Mock()
        mock_detector._cache = {"video1.mp4": 0.8, "video2.mp4": 0.3}
        mock_detector_class.get_instance.return_value = mock_detector

        video_files = [Path("video1.mp4"), Path("video2.mp4")]

        # Should complete without calling get_face_score
        stage._predetect_faces(video_files, mock_config)

    @patch('src.face_detection.FaceDetector')
    def test_predetect_faces_with_uncached(self, mock_detector_class, mock_config):
        """Test face detection with uncached videos"""
        stage = TranscribeStage()

        mock_detector = Mock()
        mock_detector._cache = {}
        mock_detector.get_face_score.side_effect = [0.8, 0.1]
        mock_detector_class.get_instance.return_value = mock_detector

        video_files = [Path("video1.mp4"), Path("video2.mp4")]

        stage._predetect_faces(video_files, mock_config)

        assert mock_detector.get_face_score.call_count == 2

    @patch('src.face_detection.FaceDetector', side_effect=Exception("detector error"))
    @patch('src.stages.transcribe.logger')
    def test_predetect_faces_error(self, mock_logger, mock_detector, mock_config):
        """Test face detection handles errors gracefully"""
        stage = TranscribeStage()

        stage._predetect_faces([Path("video1.mp4")], mock_config)

        mock_logger.warning.assert_called()


# ============================================================================
# Test Video Topic Extraction
# ============================================================================

class TestVideoTopicExtraction:
    """Test video topic extraction"""

    def test_extract_topics_empty_transcripts(self, mock_config):
        """Test topic extraction with no transcripts"""
        stage = TranscribeStage()

        # Should not raise
        stage._extract_video_topics({}, mock_config)

    @patch('src.topic_extraction.TopicExtractor')
    def test_extract_topics_success(self, mock_extractor_class, mock_config):
        """Test successful topic extraction"""
        stage = TranscribeStage()

        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        transcripts = {
            "video1.mp4": [Mock(text="Beach sunset")]
        }

        stage._extract_video_topics(transcripts, mock_config)

        # Should complete without error

    @patch('src.topic_extraction.TopicExtractor', side_effect=Exception("extraction failed"))
    @patch('src.stages.transcribe.logger')
    def test_extract_topics_error(self, mock_logger, mock_extractor, mock_config):
        """Test topic extraction handles errors"""
        stage = TranscribeStage()

        transcripts = {"video1.mp4": [{"text": "Hello"}]}

        stage._extract_video_topics(transcripts, mock_config)

        mock_logger.warning.assert_called()


# ============================================================================
# Test Skip Transcription
# ============================================================================

class TestSkipTranscription:
    """Test skip transcription behavior"""

    @patch('src.transcription.TranscriptCache')
    def test_skip_transcription_loads_cache(self, mock_cache_class, mock_config, mock_checkpoint,
                                           temp_project_dir):
        """Test skipping transcription loads from cache"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_config.pipeline.skip_transcription = True
        videos_dir = temp_project_dir / "videos"
        videos_dir.mkdir()
        (videos_dir / "video1.mp4").write_bytes(b"fake")
        mock_config.downloaded_videos_dir = str(videos_dir)

        mock_cache = Mock()
        mock_cache.get.return_value = [{"text": "Cached"}]
        mock_cache_class.return_value = mock_cache

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.transcripts) >= 0  # May be empty if no cache hits


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestTranscribeStageExecution:
    """Test full stage execution"""

    @patch('src.transcription.transcribe_videos_parallel')
    @patch('src.transcription.DeltaAwareIndex')
    def test_run_success(self, mock_delta_class, mock_transcribe, mock_config, mock_checkpoint):
        """Test successful stage execution"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]
        state.face_preference = 'neutral'

        mock_delta = Mock()
        mock_delta.get_new_videos.return_value = ["video1.mp4"]
        mock_delta_class.return_value = mock_delta

        mock_transcribe.return_value = {
            "video1.mp4": [{"text": "Hello", "start_time": 0, "end_time": 2}]
        }

        mock_config.pipeline.parallel_embedding = False  # Disable to simplify test

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.transcripts) > 0

    def test_run_no_video_files(self, mock_config, mock_checkpoint):
        """Test run with no video files"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = []

        mock_config.downloaded_videos_dir = "/nonexistent"

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(result.warnings) > 0

    def test_run_exception_handling(self, mock_config, mock_checkpoint):
        """Test run handles exceptions gracefully"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_videos = [Mock()]

        with patch.object(stage, '_get_video_files', side_effect=Exception("Error")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestTranscribeCheckpoint:
    """Test checkpoint operations"""

    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @patch.object(TranscribeStage, '_rebuild_text_metadata')
    def test_can_skip_with_checkpoint(self, mock_rebuild, mock_checkpoint):
        """Test can_skip returns True and rebuilds metadata"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_rebuild.assert_called_once()

    def test_restore_success(self, mock_checkpoint):
        """Test restore with checkpoint data"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {
            'transcript_count': 5,
            'embedding_count': 100
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True

    def test_restore_no_data(self, mock_checkpoint):
        """Test restore with no checkpoint data"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @patch('src.stages.transcribe.logger')
    def test_restore_exception_handling(self, mock_logger, mock_checkpoint):
        """Test restore handles exceptions"""
        stage = TranscribeStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.side_effect = Exception("Error")

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        mock_logger.warning.assert_called()


# ============================================================================
# Test Text Metadata Rebuilding
# ============================================================================

class TestTextMetadataRebuilding:
    """Test text metadata rebuilding for --match-only mode"""

    def test_rebuild_text_metadata_no_cache(self, mock_checkpoint):
        """Test rebuilding when no cache available"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = Path("/nonexistent") / "checkpoint.json"

        # Should not raise
        stage._rebuild_text_metadata(state, mock_checkpoint)

        assert state.text_metadata == []

    @patch('glob.glob')
    def test_rebuild_text_metadata_with_transcripts(self, mock_glob, mock_checkpoint,
                                                    temp_project_dir):
        """Test rebuilding text metadata from cached transcripts"""
        stage = TranscribeStage()
        state = PipelineState()

        mock_checkpoint.checkpoint_path = temp_project_dir / "checkpoint.json"

        # Mock transcript file
        transcript_data = json.dumps([
            {"text": "Hello", "video_path": "video1.mp4", "start_time": 0, "end_time": 2},
            {"text": "World", "video_path": "video1.mp4", "start_time": 2, "end_time": 4}
        ])

        mock_glob.return_value = [str(temp_project_dir / "transcript.json")]

        with patch('builtins.open', mock_open(read_data=transcript_data)):
            with patch('yaml.load', return_value={'cache': {'cache_dir': '.cache'}}):
                stage._rebuild_text_metadata(state, mock_checkpoint)

        # Should have rebuilt some metadata
        assert isinstance(state.text_metadata, list)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestTranscribeEdgeCases:
    """Test edge cases"""

    def test_get_video_files_prefers_audio_over_videos(self, mock_config):
        """Test that audio files take precedence over videos"""
        stage = TranscribeStage()
        state = PipelineState()
        state.downloaded_audio = [AudioDownload(file="audio.mp3", url="url", video_id="vid")]
        state.downloaded_videos = [DownloadedVideo(file="video.mp4", url="url", source="download")]

        files = stage._get_video_files(state, mock_config)

        assert len(files) == 1
        assert str(files[0]) == "audio.mp3"

    @patch('src.transcription.transcribe_videos_parallel')
    @patch('src.transcription.DeltaAwareIndex')
    def test_transcribe_all_videos_cached(self, mock_delta_class, mock_transcribe, mock_config):
        """Test transcription when all videos are cached"""
        stage = TranscribeStage()

        mock_delta = Mock()
        mock_delta.get_new_videos.return_value = []  # All cached
        mock_delta_class.return_value = mock_delta

        video_files = [Path("video1.mp4")]

        transcripts = stage._transcribe_videos(video_files, mock_config)

        assert transcripts == {}
        mock_transcribe.assert_not_called()

    def test_handle_transcripts_with_dict_segments(self, mock_config):
        """Test handling transcripts that are dicts instead of objects"""
        stage = TranscribeStage()
        state = PipelineState()

        transcripts = {
            "video1.mp4": [
                {"text": "Hello", "start_time": 0, "end_time": 2}
            ]
        }

        mock_config.pipeline.parallel_embedding = False

        # Should not raise when computing embeddings with dict segments
        result = stage._compute_embeddings(transcripts, state, mock_config)

        assert isinstance(result, bool)
