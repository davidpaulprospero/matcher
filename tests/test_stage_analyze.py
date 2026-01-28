"""
Comprehensive tests for AnalyzeStage.

Tests cover:
- Stage initialization and registration
- Voiceover loading (SRT, audio, video)
- Keyword extraction with LLM mocking
- Topic detection
- Entity extraction
- Chapter detection
- Location chapter detection
- Checkpoint save/restore
- Validation and error handling

Created: 2026-01-09 (Phase 10.1)
"""

from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest
import tempfile
from pathlib import Path

from src.stages.analyze import AnalyzeStage
from src.stages import StageResult
from src.state import PipelineState, VoiceoverSegment
from src.checkpoint import CheckpointManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for AnalyzeStage"""
    config = Mock()
    config.keyword = Mock()
    config.keyword.max_keywords = 10
    config.matching = Mock()
    config.matching.chapter_matching_enabled = False
    config.matching.location_matching = Mock()
    config.matching.location_matching.enabled = False
    config.transcription = Mock()
    config.transcription.model = 'base'
    config.transcription.compute_type = 'int8'
    return config


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_checkpoint(temp_project_dir):
    """Create mock checkpoint manager"""
    return CheckpointManager(temp_project_dir, config_hash='test_hash')


@pytest.fixture
def sample_state():
    """Create sample pipeline state"""
    state = PipelineState()
    state.voiceover_path = ""
    return state


@pytest.fixture
def sample_srt_content():
    """Sample SRT file content"""
    return """1
00:00:00,000 --> 00:00:03,500
This is the first subtitle.

2
00:00:03,500 --> 00:00:07,000
This is the second subtitle with more text.

3
00:00:07,000 --> 00:00:10,500
And here's the third one about beaches.
"""


# ============================================================================
# Test Stage Registration and Initialization
# ============================================================================

class TestAnalyzeStageInit:
    """Test AnalyzeStage initialization and metadata"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name is ANALYZE"""
        stage = AnalyzeStage()
        assert stage.name == "ANALYZE"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage has description"""
        stage = AnalyzeStage()
        assert "voiceover" in stage.description.lower()
        assert "keyword" in stage.description.lower()

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("ANALYZE")
        assert stage_class is not None
        assert stage_class == AnalyzeStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test validate_inputs method"""

    @pytest.mark.fast
    def test_validate_no_voiceover_path(self, mock_config):
        """Test validation fails when no voiceover path"""
        stage = AnalyzeStage()
        state = PipelineState()
        state.voiceover_path = ""

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover path" in error.lower()

    @pytest.mark.fast
    def test_validate_missing_file(self, mock_config):
        """Test validation fails when file doesn't exist"""
        stage = AnalyzeStage()
        state = PipelineState()
        state.voiceover_path = "/nonexistent/file.srt"

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "not found" in error.lower()

    @pytest.mark.fast
    def test_validate_success(self, mock_config, temp_project_dir):
        """Test validation succeeds with valid file"""
        stage = AnalyzeStage()
        state = PipelineState()

        # Create temp file
        srt_file = temp_project_dir / "test.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
        state.voiceover_path = str(srt_file)

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test SRT Parsing
# ============================================================================

class TestSRTParsing:
    """Test _parse_srt method"""

    @pytest.mark.fast
    def test_parse_srt_success(self, sample_srt_content, temp_project_dir):
        """Test successful SRT parsing"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "test.srt"
        srt_file.write_text(sample_srt_content)

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 3
        assert segments[0].text == "This is the first subtitle."
        assert segments[0].start == 0.0
        assert segments[0].end == 3.5
        assert segments[1].start == 3.5
        assert segments[2].text == "And here's the third one about beaches."

    @pytest.mark.fast
    def test_parse_srt_with_utf8_bom(self, temp_project_dir):
        """Test parsing SRT with UTF-8 BOM"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "test_bom.srt"

        # Write with BOM
        content = "1\n00:00:00,000 --> 00:00:02,000\nTest with BOM\n"
        srt_file.write_text(content, encoding='utf-8-sig')

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 1
        assert segments[0].text == "Test with BOM"

    @pytest.mark.fast
    def test_parse_srt_multiline_text(self, temp_project_dir):
        """Test parsing SRT with multiline subtitles"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "multiline.srt"
        content = """1
00:00:00,000 --> 00:00:03,000
First line
Second line
Third line
"""
        srt_file.write_text(content)

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 1
        assert "First line Second line Third line" in segments[0].text

    @pytest.mark.fast
    def test_parse_srt_malformed_skips_invalid(self, temp_project_dir):
        """Test parsing SRT skips malformed entries"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "malformed.srt"
        content = """1
00:00:00,000 --> 00:00:02,000
Valid entry

2
INVALID TIMESTAMP
This should be skipped

3
00:00:02,000 --> 00:00:04,000
Another valid entry
"""
        srt_file.write_text(content)

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 2  # Only valid entries
        assert segments[0].text == "Valid entry"
        assert segments[1].text == "Another valid entry"

    @pytest.mark.fast
    def test_parse_srt_empty_file(self, temp_project_dir):
        """Test parsing empty SRT file"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "empty.srt"
        srt_file.write_text("")

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 0


# ============================================================================
# Test Audio Transcription
# ============================================================================

class TestAudioTranscription:
    """Test _transcribe_audio method"""

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_transcribe_audio_success(self, mock_logger, mock_config, temp_project_dir):
        """Test successful audio transcription"""
        stage = AnalyzeStage()
        audio_file = temp_project_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock transcription at import location inside method
        mock_transcribe = MagicMock(return_value=[
            {'start': 0.0, 'end': 2.5, 'text': 'Hello world'},
            {'start': 2.5, 'end': 5.0, 'text': 'This is a test'}
        ])
        mock_write_srt = MagicMock()

        # The source code imports 'transcribe_voiceover_audio' from transcription module
        import src.transcription as transcription_module
        original_func = getattr(transcription_module, 'transcribe_voiceover_audio', None)
        original_write = getattr(transcription_module, 'write_srt', None)
        transcription_module.transcribe_voiceover_audio = mock_transcribe
        transcription_module.write_srt = mock_write_srt

        try:
            segments = stage._transcribe_audio(audio_file, mock_config)
        finally:
            # Restore original state
            if original_func is None:
                if hasattr(transcription_module, 'transcribe_voiceover_audio'):
                    delattr(transcription_module, 'transcribe_voiceover_audio')
            else:
                transcription_module.transcribe_voiceover_audio = original_func
            if original_write is None:
                if hasattr(transcription_module, 'write_srt'):
                    delattr(transcription_module, 'write_srt')
            else:
                transcription_module.write_srt = original_write

        assert len(segments) == 2
        assert segments[0].text == 'Hello world'
        assert segments[0].start == 0.0
        assert segments[1].end == 5.0

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_transcribe_audio_failure(self, mock_logger, mock_config, temp_project_dir):
        """Test audio transcription failure handling"""
        stage = AnalyzeStage()
        audio_file = temp_project_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock transcription to raise exception
        mock_transcribe = MagicMock(side_effect=Exception("Transcription failed"))

        # Add the mock function to transcription module temporarily
        import src.transcription as transcription_module
        original_func = getattr(transcription_module, 'transcribe_voiceover_audio', None)
        transcription_module.transcribe_voiceover_audio = mock_transcribe

        try:
            segments = stage._transcribe_audio(audio_file, mock_config)
        finally:
            # Restore original state
            if original_func is None:
                if hasattr(transcription_module, 'transcribe_voiceover_audio'):
                    delattr(transcription_module, 'transcribe_voiceover_audio')
            else:
                transcription_module.transcribe_voiceover_audio = original_func

        assert len(segments) == 0
        mock_logger.error.assert_called_once()


# ============================================================================
# Test Voiceover Loading
# ============================================================================

class TestVoiceoverLoading:
    """Test _load_voiceover_segments method"""

    @pytest.mark.fast
    def test_load_srt_file(self, mock_config, sample_srt_content, temp_project_dir):
        """Test loading SRT file"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "test.srt"
        srt_file.write_text(sample_srt_content)

        segments = stage._load_voiceover_segments(str(srt_file), mock_config)

        assert len(segments) == 3
        assert all(isinstance(s, VoiceoverSegment) for s in segments)

    @patch('src.stages.analyze.AnalyzeStage._transcribe_audio')
    @pytest.mark.fast
    def test_load_audio_file_mp3(self, mock_transcribe, mock_config, temp_project_dir):
        """Test loading MP3 audio file"""
        stage = AnalyzeStage()
        audio_file = temp_project_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_transcribe.return_value = [
            VoiceoverSegment(index=0, start=0.0, end=2.0, text="Test")
        ]

        segments = stage._load_voiceover_segments(str(audio_file), mock_config)

        assert len(segments) == 1
        mock_transcribe.assert_called_once()

    @patch('src.stages.analyze.AnalyzeStage._transcribe_audio')
    @pytest.mark.fast
    def test_load_audio_file_wav(self, mock_transcribe, mock_config, temp_project_dir):
        """Test loading WAV audio file"""
        stage = AnalyzeStage()
        audio_file = temp_project_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        mock_transcribe.return_value = []

        segments = stage._load_voiceover_segments(str(audio_file), mock_config)

        mock_transcribe.assert_called_once()

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_load_unknown_format(self, mock_logger, mock_config, temp_project_dir):
        """Test loading unknown file format"""
        stage = AnalyzeStage()
        file = temp_project_dir / "test.xyz"
        file.write_bytes(b"fake data")

        segments = stage._load_voiceover_segments(str(file), mock_config)

        assert len(segments) == 0
        mock_logger.warning.assert_called_once()


# ============================================================================
# Test Keyword Extraction
# ============================================================================

class TestKeywordExtraction:
    """Test _extract_keywords method"""

    @pytest.mark.fast
    def test_extract_keywords_success(self, mock_config):
        """Test successful keyword extraction"""
        stage = AnalyzeStage()
        segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Visit the beautiful beach"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="Explore ocean wildlife")
        ]

        # Mock keyword extractor
        mock_result = Mock()
        mock_result.keywords = ['beach', 'ocean', 'wildlife']
        mock_result.entities = [{'name': 'Beach', 'type': 'location'}]
        mock_result.topic = 'Travel'

        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = mock_result
        mock_extractor.extract_keyword_per_segment.return_value = ['beach', 'ocean', 'wildlife', 'beach']

        with patch('src.keyword_extractor.LLMKeywordExtractor', return_value=mock_extractor):
            keywords, entities, topic = stage._extract_keywords(segments, 10, mock_config)

        assert len(keywords) > 0
        assert 'beach' in keywords
        assert len(entities) > 0
        assert topic == 'Travel'

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_extract_keywords_llm_failure_tfidf_fallback(self, mock_logger, mock_config):
        """Test TF-IDF fallback when LLM extraction fails"""
        stage = AnalyzeStage()
        segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="beach ocean sunset"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="wildlife nature animals")
        ]

        # Mock LLM extractor to raise exception
        with patch('src.keyword_extractor.LLMKeywordExtractor', side_effect=Exception("LLM failed")):
            keywords, entities, topic = stage._extract_keywords(segments, 5, mock_config)

        # Should use TF-IDF fallback
        assert isinstance(keywords, list)
        assert isinstance(entities, list)
        mock_logger.error.assert_called()


# ============================================================================
# Test TF-IDF Fallback
# ============================================================================

class TestTFIDFFallback:
    """Test _tfidf_fallback method"""

    @pytest.mark.fast
    def test_tfidf_fallback_success(self, mock_config):
        """Test TF-IDF fallback keyword extraction"""
        stage = AnalyzeStage()
        segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="beautiful beach sunset ocean"),
            VoiceoverSegment(index=1, start=3.0, end=6.0, text="beach waves surfing ocean")
        ]

        keywords, entities, topic = stage._tfidf_fallback(segments, 5, mock_config)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert len(entities) == 0  # TF-IDF doesn't extract entities
        assert topic == ''  # TF-IDF doesn't detect topic

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_tfidf_fallback_failure(self, mock_logger, mock_config):
        """Test TF-IDF fallback handles sklearn import error"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="test")]

        with patch('sklearn.feature_extraction.text.TfidfVectorizer', side_effect=ImportError("sklearn not found")):
            keywords, entities, topic = stage._tfidf_fallback(segments, 5, mock_config)

        assert keywords == []
        assert entities == []
        assert topic == ''
        mock_logger.error.assert_called()


# ============================================================================
# Test Topic Detection
# ============================================================================

class TestTopicDetection:
    """Test _detect_topic_from_keywords method"""

    @pytest.mark.fast
    def test_detect_topic_from_keywords(self, mock_config):
        """Test topic detection from keywords"""
        stage = AnalyzeStage()
        keywords = ['beach', 'ocean', 'sunset', 'waves', 'surfing', 'travel']

        topic = stage._detect_topic_from_keywords(keywords, mock_config)

        assert 'beach' in topic.lower()
        assert 'ocean' in topic.lower()

    @pytest.mark.fast
    def test_detect_topic_empty_keywords(self, mock_config):
        """Test topic detection with empty keywords"""
        stage = AnalyzeStage()

        topic = stage._detect_topic_from_keywords([], mock_config)

        assert topic == ''


# ============================================================================
# Test Chapter Detection
# ============================================================================

class TestChapterDetection:
    """Test _detect_chapters method"""

    @pytest.mark.fast
    def test_detect_chapters_success(self, mock_config):
        """Test successful chapter detection"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Chapter 1")]

        mock_detector = Mock()
        mock_detector.detect_chapters.return_value = [
            {'title': 'Chapter 1', 'start_index': 0, 'end_index': 5}
        ]

        with patch('src.topic_extraction.ChapterDetector', return_value=mock_detector):
            chapters = stage._detect_chapters(segments, "Travel", mock_config)

        assert len(chapters) == 1
        assert chapters[0]['title'] == 'Chapter 1'

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_detect_chapters_failure(self, mock_logger, mock_config):
        """Test chapter detection failure handling"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Test")]

        with patch('src.topic_extraction.ChapterDetector', side_effect=Exception("Detection failed")):
            chapters = stage._detect_chapters(segments, "Travel", mock_config)

        assert chapters == []
        mock_logger.warning.assert_called()


# ============================================================================
# Test Location Chapter Detection
# ============================================================================

class TestLocationChapterDetection:
    """Test _detect_location_chapters method"""

    @pytest.mark.fast
    def test_location_chapters_disabled(self, mock_config):
        """Test location chapter detection when disabled"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Test")]

        mock_config.matching.location_matching.enabled = False

        location_chapters = stage._detect_location_chapters(segments, "Travel", mock_config)

        assert location_chapters == []

    @pytest.mark.fast
    def test_location_chapters_no_config(self, mock_config):
        """Test location chapter detection with no location config"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Test")]

        mock_config.matching.location_matching = None

        location_chapters = stage._detect_location_chapters(segments, "Travel", mock_config)

        assert location_chapters == []

    @pytest.mark.fast
    def test_location_chapters_success(self, mock_config):
        """Test successful location chapter detection"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Paris France")]

        mock_config.matching.location_matching.enabled = True

        # Mock the enhanced chapter detector (used by default)
        mock_enhanced_detector = Mock()
        mock_enhanced_detector.detect_chapters.return_value = [
            {'location': 'Paris', 'start_index': 0, 'title': 'Paris', 'confidence': 0.9}
        ]

        mock_location_service = Mock()

        with patch('src.stages.analyze.EnhancedChapterDetector', return_value=mock_enhanced_detector, create=True):
            with patch('src.location_service.create_location_service', return_value=mock_location_service):
                # Patch the import inside the method
                with patch.dict('sys.modules', {'src.chapter_detection': Mock(EnhancedChapterDetector=lambda *args, **kwargs: mock_enhanced_detector)}):
                    location_chapters = stage._detect_location_chapters(segments, "Travel", mock_config)

        assert len(location_chapters) == 1
        assert location_chapters[0]['location'] == 'Paris'

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_location_chapters_failure(self, mock_logger, mock_config):
        """Test location chapter detection failure handling"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="Test")]

        mock_config.matching.location_matching.enabled = True

        with patch('src.topic_extraction.ChapterDetector', side_effect=Exception("Detection failed")):
            location_chapters = stage._detect_location_chapters(segments, "Travel", mock_config)

        assert location_chapters == []
        mock_logger.warning.assert_called()


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestStageExecution:
    """Test run() method"""

    @pytest.mark.fast
    def test_run_no_voiceover_path(self, mock_config, mock_checkpoint):
        """Test run fails when no voiceover path"""
        stage = AnalyzeStage()
        state = PipelineState()
        state.voiceover_path = ""

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "voiceover path" in result.error.lower()

    @pytest.mark.fast
    def test_run_file_not_found(self, mock_config, mock_checkpoint):
        """Test run fails when file doesn't exist"""
        stage = AnalyzeStage()
        state = PipelineState()
        state.voiceover_path = "/nonexistent/file.srt"

        result = stage.run(state, mock_checkpoint, mock_checkpoint)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.fast
    def test_run_no_segments(self, mock_config, mock_checkpoint, temp_project_dir):
        """Test run fails when no segments found"""
        stage = AnalyzeStage()
        state = PipelineState()

        # Create empty SRT file
        srt_file = temp_project_dir / "empty.srt"
        srt_file.write_text("")
        state.voiceover_path = str(srt_file)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "no segments" in result.error.lower()

    @pytest.mark.fast
    def test_run_success(self, mock_config, mock_checkpoint, sample_srt_content, temp_project_dir):
        """Test successful stage execution"""
        stage = AnalyzeStage()
        state = PipelineState()

        # Create SRT file
        srt_file = temp_project_dir / "test.srt"
        srt_file.write_text(sample_srt_content)
        state.voiceover_path = str(srt_file)

        # Mock keyword extraction
        mock_result = Mock()
        mock_result.keywords = ['beach', 'ocean']
        mock_result.entities = []
        mock_result.topic = 'Travel'

        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = mock_result
        mock_extractor.extract_keyword_per_segment.return_value = ['beach', 'ocean']

        with patch('src.keyword_extractor.LLMKeywordExtractor', return_value=mock_extractor):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.voiceover_segments) == 3
        assert len(state.keywords) > 0
        assert state.topic_context == 'Travel'
        assert result.data is not None

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_logger, mock_config, mock_checkpoint, temp_project_dir):
        """Test run handles exceptions gracefully"""
        stage = AnalyzeStage()
        state = PipelineState()

        srt_file = temp_project_dir / "test.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
        state.voiceover_path = str(srt_file)

        # Mock to raise exception during keyword extraction
        # Also mock TF-IDF fallback to fail so stage actually fails
        with patch('src.keyword_extractor.LLMKeywordExtractor', side_effect=Exception("Extraction failed")):
            with patch.object(stage, '_tfidf_fallback', side_effect=Exception("TF-IDF also failed")):
                result = stage.run(state, mock_config, mock_checkpoint)

        # Should fail but not crash
        assert result.success is False
        mock_logger.exception.assert_called()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestCheckpointOperations:
    """Test can_skip and restore methods"""

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = AnalyzeStage()
        state = PipelineState()

        can_skip = stage.can_skip(state, mock_checkpoint)

        assert can_skip is False

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = AnalyzeStage()
        state = PipelineState()

        restored = stage.restore(state, mock_checkpoint)

        assert restored is False

    @pytest.mark.fast
    def test_restore_success(self, temp_project_dir):
        """Test successful restore from checkpoint"""
        stage = AnalyzeStage()
        state = PipelineState()

        # Create checkpoint with data
        checkpoint = CheckpointManager(temp_project_dir, config_hash='test')
        checkpoint_data = {
            'keywords': ['beach', 'ocean'],
            'topic_context': 'Travel',
            'entities': [{'name': 'Beach', 'type': 'location'}],
            'segments': [
                {'index': 0, 'start': 0.0, 'end': 3.0, 'text': 'Test segment'}
            ],
            'location_chapters': []
        }
        checkpoint.save('ANALYZE', checkpoint_data)

        restored = stage.restore(state, checkpoint)

        assert restored is True
        assert state.keywords == ['beach', 'ocean']
        assert state.topic_context == 'Travel'
        assert len(state.voiceover_segments) == 1
        assert state.voiceover_segments[0].text == 'Test segment'

    @patch('src.stages.analyze.logger')
    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_logger, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        stage = AnalyzeStage()
        state = PipelineState()

        # Mock checkpoint to return invalid data
        mock_checkpoint.get_stage_data = Mock(side_effect=Exception("Invalid data"))

        restored = stage.restore(state, mock_checkpoint)

        assert restored is False
        mock_logger.warning.assert_called()


# ============================================================================
# Test Helper Methods
# ============================================================================

class TestHelperMethods:
    """Test helper conversion methods"""

    @pytest.mark.fast
    def test_segment_to_dict(self):
        """Test segment to dict conversion"""
        stage = AnalyzeStage()
        segment = VoiceoverSegment(
            index=5,
            start=10.0,
            end=15.0,
            text="Test segment"
        )

        segment_dict = stage._segment_to_dict(segment)

        assert segment_dict['index'] == 5
        assert segment_dict['start'] == 10.0
        assert segment_dict['end'] == 15.0
        assert segment_dict['text'] == "Test segment"
        assert 'duration' in segment_dict

    @pytest.mark.fast
    def test_location_chapter_to_dict_with_dataclass(self):
        """Test location chapter to dict with dataclass"""
        stage = AnalyzeStage()

        mock_chapter = Mock()
        mock_chapter.__dict__ = {'location': 'Paris', 'start_index': 0}

        chapter_dict = stage._location_chapter_to_dict(mock_chapter)

        assert chapter_dict['location'] == 'Paris'

    @pytest.mark.fast
    def test_location_chapter_to_dict_with_dict(self):
        """Test location chapter to dict with dict input"""
        stage = AnalyzeStage()
        chapter = {'location': 'London', 'start_index': 5}

        chapter_dict = stage._location_chapter_to_dict(chapter)

        assert chapter_dict == chapter


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.fast
    def test_srt_with_hours(self, temp_project_dir):
        """Test SRT parsing with hours in timestamp"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "long.srt"
        content = "1\n01:30:00,000 --> 01:30:05,000\nLong video segment\n"
        srt_file.write_text(content)

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 1
        assert segments[0].start == 5400.0  # 1h 30m = 5400s
        assert segments[0].end == 5405.0

    @pytest.mark.fast
    def test_srt_with_period_separator(self, temp_project_dir):
        """Test SRT parsing with period as millisecond separator"""
        stage = AnalyzeStage()
        srt_file = temp_project_dir / "period.srt"
        content = "1\n00:00:00.000 --> 00:00:02.500\nPeriod separator\n"
        srt_file.write_text(content)

        segments = stage._parse_srt(srt_file)

        assert len(segments) == 1
        assert segments[0].end == 2.5

    @pytest.mark.fast
    def test_max_keywords_limit(self, mock_config):
        """Test keyword extraction respects max_keywords limit"""
        stage = AnalyzeStage()
        segments = [VoiceoverSegment(index=0, start=0.0, end=3.0, text="test")]

        mock_result = Mock()
        mock_result.keywords = ['a', 'b', 'c']
        mock_result.entities = []
        mock_result.topic = ''

        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = mock_result
        # Return many keywords
        mock_extractor.extract_keyword_per_segment.return_value = list('abcdefghijklmnopqrstuvwxyz')

        with patch('src.keyword_extractor.LLMKeywordExtractor', return_value=mock_extractor):
            keywords, _, _ = stage._extract_keywords(segments, max_keywords=5, config=mock_config)

        assert len(keywords) <= 5

    @pytest.mark.fast
    def test_empty_segment_text(self):
        """Test handling segments with empty text"""
        stage = AnalyzeStage()
        segment = VoiceoverSegment(index=0, start=0.0, end=1.0, text="")

        segment_dict = stage._segment_to_dict(segment)

        assert segment_dict['text'] == ""
        assert 'duration' in segment_dict
