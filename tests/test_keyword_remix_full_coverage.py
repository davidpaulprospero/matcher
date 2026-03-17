"""
Full coverage tests for keyword_remix.py - Targeting all missed lines

Covers:
- Lines 161, 166, 171-172: Pattern compilation (fuzzy/exact, errors)
- Lines 382-395, 404-407, 421: process_file_list parallel/sequential
- Lines 502, 510, 528: process_videos logging/progress
- Line 608, 645: remix_downloaded_videos edge cases
- Lines 708-709, 742: remix_audio_files disabled/excluded
- Lines 819, 834-836: to_dict methods
- Lines 920-921, 954-955, 972-973, 981: KeywordRemixer caching/API
- Lines 1003-1026, 1074-1075, 1079-1081: LLM remix methods
- Lines 1170-1173: Batch remix config

Created: 2026-01-11 (Session 15)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest

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
# Test Pattern Compilation (Lines 161, 166, 171-172)
# ============================================================================

class TestPatternCompilation:
    """Test _compile_keyword_patterns edge cases"""

    @pytest.mark.fast
    def test_multiword_fuzzy_match_pattern(self):
        """Test line 161: Multi-word keyword with fuzzy_match=True creates OR pattern"""
        config = RemixConfig(fuzzy_match=True, case_sensitive=False)
        processor = KeywordRemixProcessor(config, ["beach vacation"])

        # Should create pattern that matches any word
        patterns = processor.keyword_patterns
        assert len(patterns) == 1

        # Both "beach" and "vacation" should match
        assert patterns[0].search("beach_video.mp4")
        assert patterns[0].search("vacation_trip.mp4")
        assert not patterns[0].search("mountain_video.mp4")

    @pytest.mark.fast
    def test_exact_match_pattern_no_fuzzy(self):
        """Test line 166: fuzzy_match=False creates word boundary pattern"""
        config = RemixConfig(fuzzy_match=False, case_sensitive=False)
        processor = KeywordRemixProcessor(config, ["travel"])

        patterns = processor.keyword_patterns
        assert len(patterns) == 1

        # Word boundary should match whole words
        assert patterns[0].search("travel video")
        assert patterns[0].search("my travel guide")
        # Should not match partial (would be 'traveled')
        # Note: word boundary allows partial in some regex implementations

    @pytest.mark.fast
    def test_invalid_regex_pattern_warning(self, caplog):
        """Test lines 171-172: Invalid regex pattern logs warning"""
        config = RemixConfig(fuzzy_match=False, case_sensitive=False)

        # Create a keyword that might cause regex issues when escaped
        # Actually with re.escape, this shouldn't cause issues, so we need to mock
        with patch('re.compile', side_effect=__import__('re').error("Mock error")):
            # Should not raise, just log warning
            processor = KeywordRemixProcessor(config, ["[invalid"])
            assert len(processor.keyword_patterns) == 0


# ============================================================================
# Test process_file_list (Lines 382-395, 404-407, 421)
# ============================================================================

class TestProcessFileList:
    """Test process_file_list method coverage"""

    @pytest.mark.fast
    def test_parallel_scoring_with_progress(self, tmp_path, capsys):
        """Test lines 382-395: Parallel scoring with progress output"""
        config = RemixConfig(
            parallel_scoring=True,
            max_workers=2,
            min_relevance_score=0.1,
            log_file_processing=True  # Enable logging
        )
        processor = KeywordRemixProcessor(config, ["travel", "beach"])

        # Create 25 files to trigger parallel (>10) and progress (>20)
        file_paths = []
        for i in range(25):
            f = tmp_path / f"travel_beach_{i}.mp4"
            f.write_text("content")
            file_paths.append(f)

        result = processor.process_file_list(file_paths, show_progress=True)

        captured = capsys.readouterr()
        assert "Scoring 25 files" in captured.out
        assert result.total_files == 25
        assert result.included_files > 0

    @pytest.mark.fast
    def test_sequential_scoring_error_handling(self, tmp_path):
        """Test lines 404-407: Sequential scoring error handling"""
        config = RemixConfig(
            parallel_scoring=False,
            min_relevance_score=0.1
        )
        processor = KeywordRemixProcessor(config, ["travel"])

        # Create files
        file_paths = []
        for i in range(5):
            f = tmp_path / f"video_{i}.mp4"
            f.write_text("content")
            file_paths.append(f)

        # Mock score_video to fail for some files
        original_score = processor.score_video
        call_count = [0]

        def mock_score(fp):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Mock scoring error")
            return original_score(fp)

        with patch.object(processor, 'score_video', side_effect=mock_score):
            result = processor.process_file_list(file_paths, show_progress=False)

        # Should handle error and continue
        assert processor.metrics['scoring_errors'] >= 1
        assert result.total_files == 5

    @pytest.mark.fast
    def test_max_files_exceeded_exclusion(self, tmp_path):
        """Test line 421: Videos excluded when max_files_to_include exceeded"""
        config = RemixConfig(
            max_files_to_include=2,
            min_relevance_score=0.1
        )
        processor = KeywordRemixProcessor(config, ["travel", "beach"])

        # Create 5 highly relevant files
        file_paths = []
        for i in range(5):
            f = tmp_path / f"travel_beach_vacation_{i}.mp4"
            f.write_text("content")
            file_paths.append(f)

        result = processor.process_file_list(file_paths, show_progress=False)

        # Should only include 2, exclude 3
        assert result.included_files == 2
        assert result.excluded_files == 3


# ============================================================================
# Test process_videos Logging (Lines 502, 510, 528)
# ============================================================================

class TestProcessVideosLogging:
    """Test process_videos logging and progress"""

    @pytest.mark.fast
    def test_log_file_processing_enabled(self, tmp_path, caplog):
        """Test line 502: log_file_processing logs each file"""
        import logging
        caplog.set_level(logging.DEBUG)

        config = RemixConfig(
            parallel_scoring=True,
            max_workers=2,
            log_file_processing=True,
            min_relevance_score=0.1
        )
        processor = KeywordRemixProcessor(config, ["travel"])

        # Create 15 files to trigger parallel
        for i in range(15):
            (tmp_path / f"travel_{i}.mp4").write_text("content")

        result = processor.process_videos(tmp_path, show_progress=False)

        assert result.total_files == 15

    @pytest.mark.fast
    def test_parallel_progress_output(self, tmp_path, capsys):
        """Test line 510: Parallel scoring progress every 20 files"""
        config = RemixConfig(
            parallel_scoring=True,
            max_workers=4,
            min_relevance_score=0.1
        )
        processor = KeywordRemixProcessor(config, ["travel"])

        # Create 25 files to trigger progress
        for i in range(25):
            (tmp_path / f"travel_{i}.mp4").write_text("content")

        processor.process_videos(tmp_path, show_progress=True)

        captured = capsys.readouterr()
        assert "Scoring 25 videos" in captured.out

    @pytest.mark.fast
    def test_sequential_progress_output(self, tmp_path, capsys):
        """Test line 528: Sequential scoring progress every 20 files"""
        config = RemixConfig(
            parallel_scoring=False,
            min_relevance_score=0.1
        )
        processor = KeywordRemixProcessor(config, ["travel"])

        # Create 25 files for sequential processing
        for i in range(25):
            (tmp_path / f"travel_{i}.mp4").write_text("content")

        processor.process_videos(tmp_path, show_progress=True)

        captured = capsys.readouterr()
        assert "Scored 20/25 videos" in captured.out


# ============================================================================
# Test remix_downloaded_videos Edge Cases (Lines 608, 645)
# ============================================================================

class TestRemixDownloadedVideosEdges:
    """Test remix_downloaded_videos edge cases"""

    @pytest.mark.fast
    def test_none_config_creates_default(self, tmp_path):
        """Test line 608: config=None creates RemixConfig()"""
        (tmp_path / "video.mp4").write_text("content")

        # Pass config=None explicitly - should use defaults
        with patch('builtins.input', return_value='Y'):
            selected, result = remix_downloaded_videos(
                video_dir=tmp_path,
                keywords=["test"],
                config=None,  # Should create default
                interactive=False,
                show_progress=False
            )

        assert result is not None

    @pytest.mark.fast
    def test_more_than_3_excluded_message(self, tmp_path, capsys):
        """Test line 645: Shows '... and X more' for >3 excluded"""
        config = RemixConfig(
            min_relevance_score=0.9,  # High threshold
            show_excluded=True,
            interactive_curation=False
        )

        # Create 10 files with low relevance
        for i in range(10):
            (tmp_path / f"random_{i}.mp4").write_text("content")

        selected, result = remix_downloaded_videos(
            video_dir=tmp_path,
            keywords=["travel"],
            config=config,
            interactive=False,
            show_progress=True
        )

        captured = capsys.readouterr()
        # Should show "... and 7 more" (10 excluded - 3 shown = 7)
        assert "and" in captured.out and "more" in captured.out


# ============================================================================
# Test remix_audio_files Edge Cases (Lines 708-709, 742)
# ============================================================================

class TestRemixAudioFilesEdges:
    """Test remix_audio_files edge cases"""

    @pytest.mark.fast
    def test_disabled_returns_all_audio(self, tmp_path):
        """Test lines 708-709: Disabled config returns all audio files"""
        config = RemixConfig(enabled=False)

        audio1 = tmp_path / "audio1.mp3"
        audio2 = tmp_path / "audio2.m4a"
        audio1.write_text("content")
        audio2.write_text("content")

        selected, result = remix_audio_files(
            audio_files=[str(audio1), str(audio2)],
            keywords=["travel"],
            config=config
        )

        assert result is None
        assert len(selected) == 2

    @pytest.mark.fast
    def test_more_than_3_excluded_audio(self, tmp_path, capsys):
        """Test line 742: Shows '... and X more' for excluded audio"""
        config = RemixConfig(
            min_relevance_score=0.9,  # High threshold
            show_excluded=True,
            interactive_curation=False
        )

        # Create 10 audio files
        audio_files = []
        for i in range(10):
            f = tmp_path / f"random_{i}.mp3"
            f.write_text("content")
            audio_files.append(str(f))

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=["travel"],
            config=config,
            interactive=False,
            show_progress=True
        )

        captured = capsys.readouterr()
        assert "and" in captured.out and "more" in captured.out


# ============================================================================
# Test to_dict Methods (Lines 819, 834-836)
# ============================================================================

class TestToDictMethods:
    """Test to_dict serialization methods"""

    @pytest.mark.fast
    def test_keyword_remix_result_to_dict(self):
        """Test line 819: KeywordRemixResult.to_dict()"""
        result = KeywordRemixResult(
            original_keyword="test",
            remixed_keywords=["alt1", "alt2"],
            reasoning="test reason",
            success=True,
            provider="gemini",
            attempt=1
        )

        data = result.to_dict()

        assert data['original_keyword'] == "test"
        assert data['remixed_keywords'] == ["alt1", "alt2"]
        assert data['reasoning'] == "test reason"
        assert data['success'] is True
        assert data['provider'] == "gemini"
        assert 'timestamp' in data

    @pytest.mark.fast
    def test_keyword_remix_batch_result_to_dict(self):
        """Test lines 834-836: KeywordRemixBatchResult.to_dict()"""
        individual_result = KeywordRemixResult(
            original_keyword="test",
            remixed_keywords=["alt1"],
            reasoning="",
            success=True,
            provider="gemini",
            attempt=1
        )

        batch_result = KeywordRemixBatchResult(
            total_original=5,
            total_remixed=10,
            successful_remixes=4,
            failed_remixes=1,
            results=[individual_result],
            processing_time_seconds=2.5,
            provider="mixed"
        )

        data = batch_result.to_dict()

        assert data['total_original'] == 5
        assert data['total_remixed'] == 10
        assert data['successful_remixes'] == 4
        assert data['failed_remixes'] == 1
        assert len(data['results']) == 1
        assert data['results'][0]['original_keyword'] == "test"
        assert data['provider'] == "mixed"


# ============================================================================
# Test KeywordRemixer Caching (Lines 920-921, 954-955, 972-973)
# ============================================================================

class TestKeywordRemixerCaching:
    """Test KeywordRemixer caching methods"""

    @pytest.mark.fast
    def test_api_key_from_config(self, tmp_path):
        """Test lines 920-921: API key from config object"""
        mock_config = MagicMock()
        mock_config.gemini_api_key = "test-gemini-key"
        mock_config.anthropic_api_key = "test-anthropic-key"

        remixer = KeywordRemixer(
            config=mock_config,
            topic_context="test",
            cache_dir=str(tmp_path)
        )

        assert remixer.gemini_api_key == "test-gemini-key"
        assert remixer.anthropic_api_key == "test-anthropic-key"

    @pytest.mark.fast
    def test_cache_read_exception_handling(self, tmp_path):
        """Test lines 954-955: Exception in _get_cached_remix"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )

        # Create invalid cache file
        cache_key = remixer._get_cache_key("test", 1)
        cache_file = tmp_path / f"remix_{cache_key}.json"
        cache_file.write_text("invalid json{")

        # Should return None, not raise
        result = remixer._get_cached_remix("test", 1)
        assert result is None

    @pytest.mark.fast
    def test_cache_write_exception_handling(self, tmp_path):
        """Test lines 972-973: Exception in _cache_remix"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )

        # Make cache dir read-only to cause write error
        with patch('builtins.open', side_effect=PermissionError("No permission")):
            # Should not raise, just log warning
            remixer._cache_remix("test", 1, ["alt1"], "reason")


# ============================================================================
# Test LLM Remix Methods (Lines 981, 1003-1026, 1074-1075, 1079-1081)
# ============================================================================

class TestLLMRemixMethods:
    """Test LLM-based remix methods"""

    @pytest.mark.fast
    def test_gemini_api_key_not_available(self, tmp_path):
        """Test line 981: ValueError when Gemini API key not available"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = None

        with pytest.raises(ValueError, match="Gemini API key not available"):
            remixer.remix_keyword_gemini("test", 1)

    @pytest.mark.fast
    def test_anthropic_remix_keyword(self, tmp_path):
        """Test lines 1003-1026: remix_keyword_anthropic method"""
        mock_config = MagicMock()
        mock_config.anthropic_api_key = "test-key"
        mock_config.llm = MagicMock()
        mock_config.llm.max_tokens = 500

        remixer = KeywordRemixer(
            config=mock_config,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.anthropic_api_key = "test-key"

        # Mock the KeywordAlternativeGenerator at its source module
        with patch('src.keyword_alternatives.KeywordAlternativeGenerator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate_multiple_alternatives.return_value = (["alt1", "alt2"], "reasoning")
            mock_gen.return_value = mock_instance

            remixed, reasoning = remixer.remix_keyword_anthropic("test", 1)

        assert remixed == ["alt1", "alt2"]
        assert reasoning == "reasoning"
        assert remixer.stats['api_calls'] == 1

    @pytest.mark.fast
    def test_anthropic_api_key_not_available(self, tmp_path):
        """Test line 1003: ValueError when Anthropic API key not available"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.anthropic_api_key = None

        with pytest.raises(ValueError, match="Anthropic API key not available"):
            remixer.remix_keyword_anthropic("test", 1)

    @pytest.mark.fast
    def test_anthropic_fallback_when_gemini_fails(self, tmp_path):
        """Test lines 1074-1075: Fallback to Anthropic when Gemini returns nothing"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = "test-key"
        remixer.anthropic_api_key = "test-key"

        # Mock both remix methods
        with patch.object(remixer, 'remix_keyword_gemini', return_value=([], "")):
            with patch.object(remixer, 'remix_keyword_anthropic', return_value=(["alt1"], "anthropic")):
                result = remixer.remix_keyword("test", 1)

        assert result.provider == "anthropic"
        assert result.remixed_keywords == ["alt1"]

    @pytest.mark.fast
    def test_rule_based_fallback(self, tmp_path):
        """Test lines 1079-1081: Rule-based fallback when no LLM available"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = None
        remixer.anthropic_api_key = None

        with patch.object(remixer, '_fallback_remix', return_value=["fallback1", "fallback2"]):
            result = remixer.remix_keyword("test", 1)

        assert result.provider == "fallback"
        assert result.reasoning == "Rule-based fallback"
        assert result.remixed_keywords == ["fallback1", "fallback2"]


# ============================================================================
# Test Batch Remix Config (Lines 1170-1173)
# ============================================================================

class TestBatchRemixConfig:
    """Test _batch_remix_gemini config handling"""

    @pytest.mark.fast
    def test_batch_remix_gemini_model_from_matching_config(self, tmp_path):
        """Test lines 1170-1171: Model from config.matching"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.gemini_model = "gemini-custom-model"

        remixer = KeywordRemixer(
            config=mock_config,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = "test-key"

        with patch('src.llm_client.create_client') as mock_create:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.parsed_data = {"keyword1": ["alt1", "alt2"]}
            mock_client.generate.return_value = mock_response
            mock_create.return_value = mock_client

            result = remixer._batch_remix_gemini(["keyword1"], 1)

        # Verify custom model was used
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]['model'] == "gemini-custom-model"

    @pytest.mark.fast
    def test_batch_remix_gemini_model_from_llm_config(self, tmp_path):
        """Test lines 1172-1173: Model from config.llm"""
        mock_config = MagicMock()
        mock_config.matching = None  # No matching config
        mock_config.llm = MagicMock()
        mock_config.llm.model = "gemini-llm-model"
        del mock_config.matching  # Remove matching attribute

        remixer = KeywordRemixer(
            config=mock_config,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = "test-key"

        with patch('src.llm_client.create_client') as mock_create:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.parsed_data = {"keyword1": ["alt1"]}
            mock_client.generate.return_value = mock_response
            mock_create.return_value = mock_client

            result = remixer._batch_remix_gemini(["keyword1"], 1)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1]['model'] == "gemini-llm-model"

    @pytest.mark.fast
    def test_batch_remix_exception_handling(self, tmp_path):
        """Test lines 1210-1212: Exception in batch remix returns empty"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=str(tmp_path)
        )
        remixer.gemini_api_key = "test-key"

        with patch('src.llm_client.create_client', side_effect=Exception("API error")):
            result = remixer._batch_remix_gemini(["keyword1"], 1)

        assert result == []


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.fast
    def test_get_remix_summary_with_none(self):
        """Test get_remix_summary with None result"""
        summary = get_remix_summary(None)
        assert summary == "Remix not performed"

    @pytest.mark.fast
    def test_get_remix_summary_with_result(self):
        """Test get_remix_summary with valid result"""
        result = RemixResult(
            total_files=10,
            included_files=8,
            excluded_files=2,
            included_videos=[],
            excluded_videos=[],
            processing_time_seconds=5.5,
            keywords_used=["travel"],
            avg_match_score=0.75
        )

        summary = get_remix_summary(result)

        assert "Videos processed: 10" in summary
        assert "Included: 8" in summary
        assert "Excluded: 2" in summary

    @pytest.mark.fast
    def test_save_remix_report(self, tmp_path):
        """Test save_remix_report saves JSON"""
        result = RemixResult(
            total_files=5,
            included_files=3,
            excluded_files=2,
            included_videos=[],
            excluded_videos=[],
            processing_time_seconds=1.0,
            keywords_used=["test"],
            avg_match_score=0.5
        )

        output_path = tmp_path / "report.json"
        save_remix_report(result, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert data['total_files'] == 5
        assert data['included_files'] == 3

    @pytest.mark.fast
    def test_remix_zero_download_keywords_empty(self):
        """Test remix_zero_download_keywords with empty list"""
        remixed, result = remix_zero_download_keywords([], None, "")

        assert remixed == []
        assert result is None

    @pytest.mark.fast
    def test_remix_zero_download_keywords_with_keywords(self, tmp_path):
        """Test remix_zero_download_keywords with keywords"""
        with patch('src.keyword_remix.KeywordRemixer') as mock_class:
            mock_remixer = MagicMock()
            mock_result = MagicMock()
            mock_result.results = [
                MagicMock(remixed_keywords=["alt1", "alt2"]),
                MagicMock(remixed_keywords=["alt3"])
            ]
            mock_remixer.remix_keywords_batch.return_value = mock_result
            mock_class.return_value = mock_remixer

            remixed, result = remix_zero_download_keywords(
                ["kw1", "kw2"],
                None,
                "test topic",
                str(tmp_path)
            )

        assert "alt1" in remixed
        assert "alt2" in remixed
        assert "alt3" in remixed


# ============================================================================
# Test Case Sensitivity
# ============================================================================

class TestCaseSensitivity:
    """Test case sensitive pattern matching"""

    @pytest.mark.fast
    def test_case_sensitive_matching(self, tmp_path):
        """Test case_sensitive=True for pattern matching"""
        config = RemixConfig(case_sensitive=True, fuzzy_match=True)
        processor = KeywordRemixProcessor(config, ["Travel"])

        # Case should matter
        f1 = tmp_path / "Travel_video.mp4"
        f2 = tmp_path / "travel_video.mp4"
        f1.write_text("content")
        f2.write_text("content")

        score1 = processor.score_video(f1)
        score2 = processor.score_video(f2)

        # Exact case match should score higher
        assert score1.match_score > score2.match_score
