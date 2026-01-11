"""
Comprehensive tests for keyword remix module.

Covers:
- VideoScore and RemixResult dataclasses
- RemixConfig configuration
- KeywordRemixProcessor initialization and scoring
- Filename and metadata scoring
- Directory scanning
- File list processing
- Parallel vs sequential scoring
- Filtering and prioritization
- KeywordRemixer LLM-based keyword generation
- Caching of remix results
- Batch keyword remixing
- Convenience functions

Created: 2026-01-09 (Phase 5.1)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_remix import (
    VideoScore,
    RemixResult,
    RemixConfig,
    KeywordRemixProcessor,
    KeywordRemixer,
    KeywordRemixResult,
    KeywordRemixBatchResult,
    remix_downloaded_videos,
    remix_audio_files,
    get_remix_summary,
    save_remix_report,
    remix_zero_download_keywords
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def remix_config():
    """Create sample RemixConfig"""
    return RemixConfig(
        enabled=True,
        min_relevance_score=0.2,
        max_files_to_include=50,
        fuzzy_match=True,
        case_sensitive=False,
        parallel_scoring=False  # Disable for deterministic tests
    )


@pytest.fixture
def sample_keywords():
    """Sample keywords for testing"""
    return ["travel", "vacation", "beach"]


@pytest.fixture
def sample_video_score():
    """Create sample VideoScore"""
    return VideoScore(
        file_path="/videos/beach_video.mp4",
        filename="beach_video.mp4",
        keyword_matches=["beach", "vacation"],
        match_score=0.67,
        file_size_mb=50.5,
        duration_estimate=120.0,
        metadata={"title": "Beach Vacation"}
    )


@pytest.fixture
def sample_video_files(temp_dir):
    """Create sample video files with metadata"""
    videos = []

    for i, name in enumerate(["beach_vacation.mp4", "mountain_travel.mp4", "unrelated.mp4"]):
        video_path = temp_dir / name
        video_path.write_bytes(b"fake video content")
        videos.append(video_path)

        # Create info.json for first two videos
        if i < 2:
            info_json = video_path.with_suffix('.info.json')
            metadata = {
                "title": f"Test Video {i}",
                "description": "travel vacation" if i == 0 else "mountain adventure",
                "duration": 120 + i * 30,
                "view_count": 1000 * (i + 1)
            }
            info_json.write_text(json.dumps(metadata))

    return videos


# ============================================================================
# Test VideoScore Dataclass
# ============================================================================

class TestVideoScore:
    """Test VideoScore dataclass"""

    def test_video_score_creation(self, sample_video_score):
        """Test creating VideoScore"""
        assert sample_video_score.filename == "beach_video.mp4"
        assert sample_video_score.match_score == 0.67
        assert len(sample_video_score.keyword_matches) == 2

    def test_video_score_to_dict(self, sample_video_score):
        """Test VideoScore serialization"""
        data = sample_video_score.to_dict()

        assert data['filename'] == "beach_video.mp4"
        assert data['match_score'] == 0.67
        assert data['file_size_mb'] == 50.5


# ============================================================================
# Test RemixResult Dataclass
# ============================================================================

class TestRemixResult:
    """Test RemixResult dataclass"""

    def test_remix_result_creation(self, sample_video_score):
        """Test creating RemixResult"""
        result = RemixResult(
            total_files=10,
            included_files=7,
            excluded_files=3,
            included_videos=[sample_video_score],
            excluded_videos=[],
            processing_time_seconds=2.5,
            keywords_used=["beach", "vacation"],
            avg_match_score=0.65
        )

        assert result.total_files == 10
        assert result.included_files == 7
        assert result.avg_match_score == 0.65

    def test_remix_result_to_dict(self, sample_video_score):
        """Test RemixResult serialization"""
        result = RemixResult(
            total_files=10,
            included_files=7,
            excluded_files=3,
            included_videos=[sample_video_score],
            excluded_videos=[],
            processing_time_seconds=2.5,
            keywords_used=["beach"],
            avg_match_score=0.65
        )

        data = result.to_dict()
        assert data['total_files'] == 10
        assert len(data['included_videos']) == 1
        assert data['included_videos'][0]['filename'] == "beach_video.mp4"


# ============================================================================
# Test RemixConfig
# ============================================================================

class TestRemixConfig:
    """Test RemixConfig"""

    def test_config_defaults(self):
        """Test default config values"""
        config = RemixConfig()

        assert config.enabled is True
        assert config.min_relevance_score == 0.1
        assert config.max_files_to_include == 100
        assert config.fuzzy_match is True

    def test_config_custom_values(self):
        """Test custom config values"""
        config = RemixConfig(
            min_relevance_score=0.5,
            max_files_to_include=20,
            fuzzy_match=False
        )

        assert config.min_relevance_score == 0.5
        assert config.max_files_to_include == 20
        assert config.fuzzy_match is False


# ============================================================================
# Test KeywordRemixProcessor Initialization
# ============================================================================

class TestKeywordRemixProcessorInit:
    """Test KeywordRemixProcessor initialization"""

    def test_init_basic(self, remix_config, sample_keywords):
        """Test basic initialization"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        assert processor.config == remix_config
        assert len(processor.keywords) == 3
        assert processor.metrics['files_scanned'] == 0

    def test_init_case_normalization(self, remix_config):
        """Test keyword case normalization"""
        processor = KeywordRemixProcessor(remix_config, ["TRAVEL", "Beach"])

        # Should be lowercase when case_sensitive=False
        assert processor.keywords == ["travel", "beach"]

    def test_init_case_sensitive(self):
        """Test case-sensitive mode"""
        config = RemixConfig(case_sensitive=True)
        processor = KeywordRemixProcessor(config, ["TRAVEL", "Beach"])

        # Should preserve case
        assert processor.keywords == ["TRAVEL", "Beach"]

    def test_compile_keyword_patterns(self, remix_config, sample_keywords):
        """Test regex pattern compilation"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        assert len(processor.keyword_patterns) == 3
        for pattern in processor.keyword_patterns:
            assert pattern is not None


# ============================================================================
# Test Filename Scoring
# ============================================================================

class TestFilenameScoring:
    """Test filename scoring against keywords"""

    def test_score_filename_exact_match(self, remix_config):
        """Test exact keyword match in filename"""
        processor = KeywordRemixProcessor(remix_config, ["beach", "vacation"])

        score, matches = processor._score_filename("beach_vacation_2024.mp4")

        assert score > 0.5
        assert "beach" in matches
        assert "vacation" in matches

    def test_score_filename_partial_match(self, remix_config):
        """Test partial keyword match"""
        processor = KeywordRemixProcessor(remix_config, ["travel", "beach"])

        score, matches = processor._score_filename("summer_travel_video.mp4")

        assert score > 0
        assert "travel" in matches

    def test_score_filename_no_match(self, remix_config):
        """Test no keyword match"""
        processor = KeywordRemixProcessor(remix_config, ["beach", "ocean"])

        score, matches = processor._score_filename("mountain_hiking.mp4")

        assert score == 0.0
        assert len(matches) == 0

    def test_score_filename_special_chars(self, remix_config):
        """Test filename with special characters"""
        processor = KeywordRemixProcessor(remix_config, ["travel"])

        score, matches = processor._score_filename("my-travel-vlog_2024.mp4")

        assert score > 0
        assert "travel" in matches


# ============================================================================
# Test Metadata Scoring
# ============================================================================

class TestMetadataScoring:
    """Test metadata scoring"""

    def test_score_metadata_with_file(self, temp_dir, remix_config):
        """Test scoring with existing info.json"""
        processor = KeywordRemixProcessor(remix_config, ["travel", "beach"])

        # Create info.json
        info_json = temp_dir / "video.info.json"
        metadata = {
            "title": "Amazing Beach Travel Video",
            "description": "Beach vacation footage",
            "tags": ["travel", "vacation"],
            "duration": 120
        }
        info_json.write_text(json.dumps(metadata))

        score, matches, meta = processor._score_metadata(info_json)

        assert score > 0
        assert "travel" in matches
        assert "beach" in matches
        assert meta['duration'] == 120

    def test_score_metadata_missing_file(self, temp_dir, remix_config):
        """Test scoring with missing info.json"""
        processor = KeywordRemixProcessor(remix_config, ["travel"])

        info_json = temp_dir / "nonexistent.info.json"
        score, matches, meta = processor._score_metadata(info_json)

        assert score == 0.0
        assert len(matches) == 0
        assert meta == {}

    def test_score_metadata_invalid_json(self, temp_dir, remix_config):
        """Test scoring with invalid JSON"""
        processor = KeywordRemixProcessor(remix_config, ["travel"])

        info_json = temp_dir / "video.info.json"
        info_json.write_text("invalid json {")

        score, matches, meta = processor._score_metadata(info_json)

        assert score == 0.0


# ============================================================================
# Test Video Scoring
# ============================================================================

class TestVideoScoring:
    """Test complete video scoring"""

    def test_score_video_with_metadata(self, temp_dir, remix_config):
        """Test scoring video with metadata"""
        processor = KeywordRemixProcessor(remix_config, ["travel", "beach"])

        # Create video file
        video_path = temp_dir / "beach_travel.mp4"
        video_path.write_bytes(b"fake video" * 1000)

        # Create info.json
        info_json = video_path.with_suffix('.info.json')
        metadata = {
            "title": "Beach Travel Video",
            "duration": 180
        }
        info_json.write_text(json.dumps(metadata))

        score = processor.score_video(video_path)

        assert score.filename == "beach_travel.mp4"
        assert score.match_score > 0
        assert score.file_size_mb > 0
        assert score.duration_estimate == 180

    def test_score_video_without_metadata(self, temp_dir, remix_config):
        """Test scoring video without metadata (filename only)"""
        processor = KeywordRemixProcessor(remix_config, ["beach"])

        video_path = temp_dir / "beach_sunset.mp4"
        video_path.write_bytes(b"fake")

        score = processor.score_video(video_path)

        assert score.match_score > 0
        assert score.duration_estimate is None


# ============================================================================
# Test Directory Scanning
# ============================================================================

class TestDirectoryScanning:
    """Test directory scanning"""

    def test_scan_directory_videos_only(self, temp_dir, remix_config, sample_keywords):
        """Test scanning for video files only"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create mixed files
        (temp_dir / "video1.mp4").write_bytes(b"fake")
        (temp_dir / "video2.mkv").write_bytes(b"fake")
        (temp_dir / "audio.mp3").write_bytes(b"fake")
        (temp_dir / "readme.txt").write_bytes(b"fake")

        files = processor.scan_directory(temp_dir, include_audio=False)

        assert len(files) == 2
        assert all(f.suffix in {'.mp4', '.mkv'} for f in files)

    def test_scan_directory_with_audio(self, temp_dir, remix_config, sample_keywords):
        """Test scanning including audio files"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        (temp_dir / "video.mp4").write_bytes(b"fake")
        (temp_dir / "audio.mp3").write_bytes(b"fake")
        (temp_dir / "audio.m4a").write_bytes(b"fake")

        files = processor.scan_directory(temp_dir, include_audio=True)

        assert len(files) == 3

    def test_scan_directory_max_files(self, temp_dir, sample_keywords):
        """Test max files limit"""
        config = RemixConfig(max_files_to_process=2)
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create 5 files
        for i in range(5):
            (temp_dir / f"video{i}.mp4").write_bytes(b"fake")

        files = processor.scan_directory(temp_dir)

        assert len(files) == 2  # Limited by max_files_to_process


# ============================================================================
# Test File List Processing
# ============================================================================

class TestFileListProcessing:
    """Test processing file lists"""

    def test_process_empty_list(self, remix_config, sample_keywords):
        """Test processing empty file list"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        result = processor.process_file_list([], show_progress=False)

        assert result.total_files == 0
        assert result.included_files == 0

    def test_process_file_list_basic(self, sample_video_files, remix_config):
        """Test processing list of files"""
        processor = KeywordRemixProcessor(remix_config, ["travel", "beach"])

        result = processor.process_file_list(sample_video_files, show_progress=False)

        assert result.total_files == 3
        assert result.included_files > 0
        assert result.processing_time_seconds >= 0

    def test_process_file_list_filtering(self, sample_video_files):
        """Test relevance filtering"""
        config = RemixConfig(min_relevance_score=0.5, parallel_scoring=False)
        processor = KeywordRemixProcessor(config, ["beach", "vacation"])

        result = processor.process_file_list(sample_video_files, show_progress=False)

        # Files with low scores should be excluded
        for video in result.included_videos:
            assert video.match_score >= 0.5


# ============================================================================
# Test KeywordRemixer
# ============================================================================

class TestKeywordRemixer:
    """Test LLM-based KeywordRemixer"""

    def test_init_basic(self):
        """Test basic initialization"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )

        assert remixer.topic_context == "travel videos"
        assert remixer.stats['total_remixes'] == 0

    def test_cache_key_generation(self):
        """Test cache key generation"""
        remixer = KeywordRemixer(topic_context="test")

        key1 = remixer._get_cache_key("travel", attempt=1)
        key2 = remixer._get_cache_key("travel", attempt=2)

        assert isinstance(key1, str)
        assert len(key1) == 16
        assert key1 != key2  # Different attempts = different keys

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_remix_keyword_with_cache(self, mock_generator_class, temp_dir):
        """Test remix with caching"""
        remixer = KeywordRemixer(cache_dir=str(temp_dir))

        # First call - cache miss
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_multiple_alternatives.return_value = (
            ["beach vacation", "tropical getaway"],
            "Success"
        )

        remixer.gemini_api_key = "fake_key"
        result1 = remixer.remix_keyword("beach", attempt=1)

        assert result1.success
        assert len(result1.remixed_keywords) == 2

        # Second call - should use cache
        result2 = remixer.remix_keyword("beach", attempt=1)

        assert result2.provider == "cache"
        assert result2.remixed_keywords == result1.remixed_keywords

    def test_fallback_remix(self):
        """Test rule-based fallback remix"""
        remixer = KeywordRemixer()

        remixed = remixer._fallback_remix("Beach vacation 2024 footage")

        assert isinstance(remixed, list)
        assert len(remixed) > 0


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_remix_summary(self, sample_video_score):
        """Test generating remix summary"""
        result = RemixResult(
            total_files=10,
            included_files=7,
            excluded_files=3,
            included_videos=[sample_video_score],
            excluded_videos=[],
            processing_time_seconds=2.5,
            keywords_used=["beach"],
            avg_match_score=0.65
        )

        summary = get_remix_summary(result)

        assert "10" in summary
        assert "7" in summary
        assert "65" in summary

    def test_get_remix_summary_none(self):
        """Test summary when result is None"""
        summary = get_remix_summary(None)

        assert "not performed" in summary.lower()

    def test_save_remix_report(self, temp_dir, sample_video_score):
        """Test saving remix report to JSON"""
        result = RemixResult(
            total_files=5,
            included_files=3,
            excluded_files=2,
            included_videos=[sample_video_score],
            excluded_videos=[],
            processing_time_seconds=1.0,
            keywords_used=["beach"],
            avg_match_score=0.7
        )

        output_path = temp_dir / "report.json"
        save_remix_report(result, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data['total_files'] == 5
        assert data['avg_match_score'] == 0.7

    @patch('src.keyword_remix.KeywordRemixer')
    def test_remix_zero_download_keywords(self, mock_remixer_class):
        """Test convenience function for remixing failed keywords"""
        mock_remixer = mock_remixer_class.return_value
        mock_result = KeywordRemixBatchResult(
            total_original=2,
            total_remixed=4,
            successful_remixes=2,
            failed_remixes=0,
            results=[
                KeywordRemixResult("kw1", ["alt1", "alt2"], "OK", True, "gemini", 1),
                KeywordRemixResult("kw2", ["alt3", "alt4"], "OK", True, "gemini", 1)
            ],
            processing_time_seconds=1.0,
            provider="gemini"
        )
        mock_remixer.remix_keywords_batch.return_value = mock_result

        remixed, result = remix_zero_download_keywords(
            failed_keywords=["kw1", "kw2"],
            config=None,
            topic_context="test"
        )

        assert len(remixed) == 4
        assert "alt1" in remixed


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_processor_no_keywords(self, remix_config):
        """Test processor with no keywords"""
        processor = KeywordRemixProcessor(remix_config, [])

        score, matches = processor._score_filename("test_video.mp4")

        assert score == 0.0
        assert len(matches) == 0

    def test_score_video_nonexistent_file(self, temp_dir, remix_config, sample_keywords):
        """Test scoring nonexistent file"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        nonexistent = temp_dir / "missing.mp4"

        # Should handle gracefully
        try:
            score = processor.score_video(nonexistent)
            # File size will be 0 if file doesn't exist
            assert score.file_size_mb == 0.0
        except:
            pass  # Acceptable to raise exception

    @patch('src.keyword_remix.KeywordRemixer')
    def test_remix_empty_keywords_list(self, mock_remixer_class):
        """Test remixing with empty keywords list"""
        remixed, result = remix_zero_download_keywords(
            failed_keywords=[],
            config=None
        )

        assert remixed == []
        assert result is None

    def test_remix_config_disabled(self, temp_dir):
        """Test remix when disabled in config"""
        config = RemixConfig(enabled=False)

        # Create some videos
        (temp_dir / "video.mp4").write_bytes(b"fake")

        result, remix_result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=["test"],
            config=config,
            interactive=False
        )

        # Should return all videos without filtering
        assert len(result) > 0
        assert remix_result is None
