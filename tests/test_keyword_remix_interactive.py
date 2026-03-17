"""
Tests for keyword_remix.py Interactive & LLM Methods

Targets uncovered lines:
- Lines 625-678: Interactive curation prompts in remix_downloaded_videos()
- Lines 708-771: Interactive curation prompts in remix_audio_files()
- Lines 1112-1157: remix_zero_download_keywords() orchestration
- Lines 1161-1212: _batch_remix_gemini() LLM integration

Created: 2026-01-10 (Session 12 - keyword_remix expansion)
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
    remix_audio_files
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
def sample_keywords():
    """Sample keywords for testing"""
    return ["travel", "beach", "vacation", "sunset"]


# ============================================================================
# Test Interactive Curation in remix_downloaded_videos() (Lines 625-678)
# ============================================================================

class TestRemixDownloadedVideosInteractive:
    """Test interactive curation prompts and auto-accept modes"""

    @pytest.mark.fast
    def test_auto_accept_filtered(self, temp_dir, sample_keywords):
        """Test auto_accept_filter='filtered' mode (lines 650-656)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='filtered',
            log_file_processing=False
        )

        # Create test videos
        (temp_dir / "travel_beach.mp4").write_text("travel beach")
        (temp_dir / "random.mp4").write_text("random")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should auto-accept filtered videos without prompting
        assert result is not None
        assert len(selected) == result.included_files
        # Filtered videos should be returned, not all videos
        assert len(selected) < result.total_files or result.total_files == result.included_files

    @pytest.mark.fast
    def test_auto_accept_all(self, temp_dir, sample_keywords):
        """Test auto_accept_filter='all' mode (lines 657-662)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='all',
            log_file_processing=False,
            min_relevance_score=0.8  # High threshold to exclude some
        )

        # Create test videos
        (temp_dir / "travel_beach.mp4").write_text("travel beach")
        (temp_dir / "random.mp4").write_text("random")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should return ALL videos (included + excluded)
        assert result is not None
        assert len(selected) == result.total_files
        # Should bypass filtering
        assert len(selected) >= result.included_files

    @pytest.mark.fast
    def test_interactive_prompt_yes(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt with 'Y' response (lines 664-678)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',  # Force prompt
            log_file_processing=False
        )

        # Create test videos
        (temp_dir / "travel_beach.mp4").write_text("travel beach")

        # Mock user input to return 'Y'
        inputs = iter(['Y'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should use filtered videos
        assert result is not None
        assert len(selected) == result.included_files

    @pytest.mark.fast
    def test_interactive_prompt_all(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt with 'A' response (lines 672-675)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',
            log_file_processing=False,
            min_relevance_score=0.9  # High threshold
        )

        # Create test videos
        (temp_dir / "travel.mp4").write_text("travel")
        (temp_dir / "random.mp4").write_text("random")

        # Mock user input to return 'A' (use all)
        inputs = iter(['A'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should use ALL videos (bypassing filter)
        assert result is not None
        assert len(selected) == result.total_files

    @pytest.mark.fast
    def test_interactive_prompt_cancel(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt with 'N' response (lines 676-678)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',
            log_file_processing=False
        )

        # Create test videos
        (temp_dir / "travel.mp4").write_text("travel")

        # Mock user input to return 'N' (cancel)
        inputs = iter(['N'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should return empty list (user cancelled)
        assert result is not None
        assert len(selected) == 0

    @pytest.mark.fast
    def test_show_progress_output(self, temp_dir, sample_keywords, capsys):
        """Test show_progress output (lines 625-645)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=False,
            log_file_processing=False,
            show_excluded=True
        )

        # Create test videos
        (temp_dir / "travel_beach_vacation.mp4").write_text("travel beach vacation")
        (temp_dir / "random.mp4").write_text("random")

        remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=False,
            show_progress=True  # Enable progress output
        )

        captured = capsys.readouterr()
        # Check for expected output elements
        assert "Remix Results:" in captured.out
        assert "Total videos scanned:" in captured.out
        assert "Included" in captured.out
        assert "Top 5 matches:" in captured.out or "Excluded" in captured.out


# ============================================================================
# Test Interactive Curation in remix_audio_files() (Lines 708-771)
# ============================================================================

class TestRemixAudioFilesInteractive:
    """Test interactive curation for audio file remixing"""

    @pytest.mark.fast
    def test_auto_accept_filtered_audio(self, temp_dir, sample_keywords):
        """Test auto_accept_filter='filtered' for audio (lines 746-751)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='filtered',
            log_file_processing=False
        )

        # Create audio files
        audio1 = temp_dir / "travel_podcast.mp3"
        audio2 = temp_dir / "random.mp3"
        audio1.write_text("travel")
        audio2.write_text("random")

        audio_files = [str(audio1), str(audio2)]

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should auto-accept filtered audio
        assert result is not None
        assert len(selected) == result.included_files

    @pytest.mark.fast
    def test_auto_accept_all_audio(self, temp_dir, sample_keywords):
        """Test auto_accept_filter='all' for audio (lines 752-756)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='all',
            log_file_processing=False,
            min_relevance_score=0.9
        )

        # Create audio files
        audio1 = temp_dir / "travel.mp3"
        audio2 = temp_dir / "random.mp3"
        audio1.write_text("travel")
        audio2.write_text("random")

        audio_files = [str(audio1), str(audio2)]

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should return ALL audio files
        assert result is not None
        assert len(selected) == result.total_files

    @pytest.mark.fast
    def test_interactive_prompt_yes_audio(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt 'Y' for audio (lines 758-771)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',
            log_file_processing=False
        )

        audio1 = temp_dir / "travel.mp3"
        audio1.write_text("travel")
        audio_files = [str(audio1)]

        # Mock user input
        inputs = iter(['Y'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        assert result is not None
        assert len(selected) == result.included_files

    @pytest.mark.fast
    def test_interactive_prompt_all_audio(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt 'A' for audio (lines 765-768)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',
            log_file_processing=False,
            min_relevance_score=0.9
        )

        audio1 = temp_dir / "travel.mp3"
        audio2 = temp_dir / "random.mp3"
        audio1.write_text("travel")
        audio2.write_text("random")
        audio_files = [str(audio1), str(audio2)]

        # Mock user input to select ALL
        inputs = iter(['A'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should use ALL files
        assert result is not None
        assert len(selected) == result.total_files

    @pytest.mark.fast
    def test_interactive_prompt_cancel_audio(self, temp_dir, sample_keywords, monkeypatch):
        """Test interactive prompt 'N' for audio (lines 769-771)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,
            auto_accept_filter='prompt',
            log_file_processing=False
        )

        audio1 = temp_dir / "travel.mp3"
        audio1.write_text("travel")
        audio_files = [str(audio1)]

        # Mock user input to cancel
        inputs = iter(['N'])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=True,
            show_progress=False
        )

        # Should return empty list
        assert result is not None
        assert len(selected) == 0

    @pytest.mark.fast
    def test_show_progress_audio(self, temp_dir, sample_keywords, capsys):
        """Test show_progress output for audio (lines 721-742)"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=False,
            log_file_processing=False,
            show_excluded=True
        )

        audio1 = temp_dir / "travel_beach.mp3"
        audio2 = temp_dir / "random.mp3"
        audio1.write_text("travel beach")
        audio2.write_text("random")
        audio_files = [str(audio1), str(audio2)]

        remix_audio_files(
            audio_files=audio_files,
            keywords=sample_keywords,
            config=config,
            interactive=False,
            show_progress=True
        )

        captured = capsys.readouterr()
        # Check for expected audio-specific output
        assert "Remix Results (Audio-First Mode):" in captured.out
        assert "Total audio files scored:" in captured.out


# ============================================================================
# Test KeywordRemixer LLM Methods (Lines 1112-1212)
# ============================================================================

class TestKeywordRemixerLLM:
    """Test LLM-based keyword remixing methods"""

    @pytest.mark.fast
    def test_remix_keywords_batch_basic(self):
        """Test remix_keywords_batch orchestration (lines 1112-1157)"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )

        # Mock the remix_keyword method to avoid actual LLM calls
        remixer.remix_keyword = Mock(return_value=KeywordRemixResult(
            original_keyword="travel",
            remixed_keywords=["travel vlog", "travel guide"],
            reasoning="Mock remix",
            success=True,
            provider="mock",
            attempt=1
        ))

        keywords = ["travel", "beach"]
        result = remixer.remix_keywords_batch(keywords, attempt=1)

        # Verify batch result structure
        assert result.total_original == 2
        assert result.successful_remixes >= 0
        assert result.failed_remixes >= 0
        assert len(result.results) == 2
        assert result.processing_time_seconds >= 0

    @pytest.mark.fast
    def test_remix_keywords_batch_with_gemini_batch(self):
        """Test batch processing path with Gemini (lines 1119-1126)"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )
        remixer.gemini_api_key = "test_key"

        # Mock batch Gemini method
        mock_batch_results = [
            KeywordRemixResult(
                original_keyword="travel",
                remixed_keywords=["travel vlog"],
                reasoning="Gemini batch",
                success=True,
                provider="gemini",
                attempt=1
            )
        ]
        remixer._batch_remix_gemini = Mock(return_value=mock_batch_results)

        keywords = ["travel", "beach"]
        result = remixer.remix_keywords_batch(keywords, attempt=1)

        # Should call batch remix
        remixer._batch_remix_gemini.assert_called_once_with(keywords, 1)
        assert result.total_original == 2

    @pytest.mark.fast
    def test_remix_keywords_batch_fallback_individual(self):
        """Test fallback to individual processing (lines 1127-1134)"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )
        remixer.gemini_api_key = "test_key"

        # Mock batch to return partial results
        remixer._batch_remix_gemini = Mock(return_value=[
            KeywordRemixResult(
                original_keyword="travel",
                remixed_keywords=["travel vlog"],
                reasoning="Gemini",
                success=True,
                provider="gemini",
                attempt=1
            )
        ])

        # Mock individual remix for remaining keyword
        remixer.remix_keyword = Mock(return_value=KeywordRemixResult(
            original_keyword="beach",
            remixed_keywords=["beach vacation"],
            reasoning="Individual",
            success=True,
            provider="mock",
            attempt=1
        ))

        keywords = ["travel", "beach"]
        result = remixer.remix_keywords_batch(keywords, attempt=1)

        # Should process "beach" individually (not in batch results)
        remixer.remix_keyword.assert_called_once_with("beach", 1)
        assert result.total_original == 2

    @pytest.mark.fast
    def test_batch_remix_gemini_success(self):
        """Test _batch_remix_gemini with successful response (lines 1161-1208)"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )
        remixer.gemini_api_key = "test_gemini_key"

        # Mock LLM client - patch at import location inside the method
        mock_response = Mock()
        mock_response.parsed_data = {
            "travel": ["travel vlog", "travel guide"],
            "beach": ["beach vacation", "beach sunset"]
        }

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            keywords = ["travel", "beach"]
            results = remixer._batch_remix_gemini(keywords, attempt=1)

            # Should return 2 successful results
            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].original_keyword == "travel"
            assert results[1].original_keyword == "beach"
            assert remixer.stats['api_calls'] == 1

    @pytest.mark.fast
    def test_batch_remix_gemini_no_api_key(self):
        """Test _batch_remix_gemini without API key (lines 1161-1162)"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )
        remixer.gemini_api_key = None

        results = remixer._batch_remix_gemini(["travel"], attempt=1)

        # Should return empty list
        assert results == []

    @pytest.mark.fast
    def test_batch_remix_gemini_error_handling(self):
        """Test _batch_remix_gemini error handling"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )
        remixer.gemini_api_key = "test_key"

        # Mock LLM client to raise exception
        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("API error")
            mock_create_client.return_value = mock_client

            # Should handle exception gracefully and return empty list
            keywords = ["travel"]
            results = remixer._batch_remix_gemini(keywords, attempt=1)

            # Should return empty list on error (exception is caught)
            assert results == []


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.mark.fast
    def test_remix_with_disabled_config(self, temp_dir):
        """Test remix_downloaded_videos with disabled config (lines 708-709 in remix_audio_files)"""
        config = RemixConfig(enabled=False)

        (temp_dir / "test.mp4").write_text("test")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=["travel"],
            config=config
        )

        # Should return all videos when disabled
        assert result is None
        assert len(selected) >= 1

    @pytest.mark.fast
    def test_remix_audio_files_empty_list(self):
        """Test remix_audio_files with empty list (lines 711-713)"""
        config = RemixConfig(enabled=True)

        selected, result = remix_audio_files(
            audio_files=[],
            keywords=["travel"],
            config=config
        )

        # Should return empty list
        assert selected == []
        assert result is None

    @pytest.mark.fast
    def test_interactive_non_interactive_fallthrough(self, temp_dir, sample_keywords):
        """Test when interactive=False, should skip prompts"""
        config = RemixConfig(
            enabled=True,
            interactive_curation=True,  # Config says yes
            log_file_processing=False
        )

        (temp_dir / "travel.mp4").write_text("travel")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=sample_keywords,
            config=config,
            interactive=False,  # But parameter says no
            show_progress=False
        )

        # Should skip interactive prompts
        assert result is not None
        assert len(selected) == result.included_files
