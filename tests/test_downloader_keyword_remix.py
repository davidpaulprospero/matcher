"""
Comprehensive test suite for src/downloader/keyword_remix.py - SearchOptimizer class.

Tests coverage for:
- SearchOptimizer initialization
- Keyword remix generation (LLM-based and simple)
- Keyword categorization
- Adaptive search pool sizing
- Search pass rate tracking
- Source diversity tracking and reporting
- Retry keyword generation

Created: January 10, 2026
Session: 13 Phase 3
Target Coverage: 30% → 75%
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.keyword_remix import SearchOptimizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock Config object"""
    config = Mock()

    # Download config with defaults
    config.download = Mock()
    config.download.search_pool_multiplier = 5
    config.download.min_search_pool = 30
    config.download.max_search_pool = 100

    # Zero download remix config
    config.download.zero_download_remix = Mock()
    config.download.zero_download_remix.enabled = True
    config.download.zero_download_remix.use_llm = False

    # LLM title filter config
    config.download.llm_title_filter = Mock()
    config.download.llm_title_filter.provider = 'gemini'
    config.download.llm_title_filter.model = 'gemini-2.0-flash'

    return config


@pytest.fixture
def mock_llm_gemini():
    """Mock Gemini LLM call"""
    return Mock(return_value="alternative keyword")


@pytest.fixture
def mock_llm_anthropic():
    """Mock Anthropic LLM call"""
    return Mock(return_value="anthropic alternative")


@pytest.fixture
def optimizer(mock_config, mock_llm_gemini, mock_llm_anthropic):
    """Create SearchOptimizer instance"""
    return SearchOptimizer(
        config=mock_config,
        llm_call_gemini_func=mock_llm_gemini,
        llm_call_anthropic_func=mock_llm_anthropic
    )


# ============================================================================
# Test Initialization
# ============================================================================

class TestSearchOptimizerInit:
    """Test SearchOptimizer initialization"""

    def test_init_basic(self, mock_config, mock_llm_gemini, mock_llm_anthropic):
        """Test basic initialization"""
        optimizer = SearchOptimizer(
            config=mock_config,
            llm_call_gemini_func=mock_llm_gemini,
            llm_call_anthropic_func=mock_llm_anthropic
        )

        assert optimizer.config is mock_config
        assert optimizer.download_config is mock_config.download
        assert optimizer._call_gemini is mock_llm_gemini
        assert optimizer._call_anthropic is mock_llm_anthropic
        assert optimizer._search_pass_rates == {}
        assert optimizer._keyword_sources == {}
        assert optimizer._source_keywords == {}

    def test_init_stores_llm_functions(self, optimizer, mock_llm_gemini, mock_llm_anthropic):
        """Test that LLM functions are stored correctly"""
        assert callable(optimizer._call_gemini)
        assert callable(optimizer._call_anthropic)
        assert optimizer._call_gemini is mock_llm_gemini
        assert optimizer._call_anthropic is mock_llm_anthropic


# ============================================================================
# Test Keyword Categorization
# ============================================================================

class TestKeywordCategorization:
    """Test keyword categorization for adaptive pool sizing"""

    def test_category_stock_footage(self, optimizer):
        """Test stock footage category detection"""
        assert optimizer.get_keyword_category("beach footage") == "stock_footage"
        assert optimizer.get_keyword_category("STOCK nature") == "stock_footage"
        assert optimizer.get_keyword_category("b-roll sunset") == "stock_footage"
        assert optimizer.get_keyword_category("broll city") == "stock_footage"

    def test_category_documentary(self, optimizer):
        """Test documentary category detection"""
        assert optimizer.get_keyword_category("history documentary") == "documentary"
        assert optimizer.get_keyword_category("science explained") == "documentary"

    def test_category_travel(self, optimizer):
        """Test travel category detection"""
        assert optimizer.get_keyword_category("paris tour") == "travel"
        assert optimizer.get_keyword_category("walkthrough museum") == "travel"
        assert optimizer.get_keyword_category("walk through city") == "travel"
        assert optimizer.get_keyword_category("travel vlog") == "travel"

    def test_category_cinematic(self, optimizer):
        """Test cinematic category detection"""
        assert optimizer.get_keyword_category("aerial view") == "cinematic"
        assert optimizer.get_keyword_category("drone flight") == "cinematic"  # Not "footage"
        assert optimizer.get_keyword_category("4k nature") == "cinematic"
        assert optimizer.get_keyword_category("timelapse sunset") == "cinematic"

    def test_category_interview(self, optimizer):
        """Test interview category detection"""
        assert optimizer.get_keyword_category("interview CEO") == "interview"
        assert optimizer.get_keyword_category("speech president") == "interview"
        assert optimizer.get_keyword_category("talk show") == "interview"

    def test_category_general(self, optimizer):
        """Test general/default category"""
        assert optimizer.get_keyword_category("random keyword") == "general"
        assert optimizer.get_keyword_category("something else") == "general"
        assert optimizer.get_keyword_category("cats") == "general"

    def test_category_case_insensitive(self, optimizer):
        """Test categorization is case-insensitive"""
        assert optimizer.get_keyword_category("TRAVEL TOUR") == "travel"
        assert optimizer.get_keyword_category("Drone View") == "cinematic"  # Not "footage"
        assert optimizer.get_keyword_category("StOcK VIDEO") == "stock_footage"


# ============================================================================
# Test Simple Keyword Remix
# ============================================================================

class TestSimpleKeywordRemix:
    """Test simple (non-LLM) keyword remixing"""

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_simple_remix_delegates_to_generator(self, mock_generator_class, optimizer):
        """Test that simple_remix_keyword delegates to KeywordAlternativeGenerator"""
        mock_generator = Mock()
        mock_generator.simple_remix_keyword.return_value = "remixed keyword"
        mock_generator_class.return_value = mock_generator

        result = optimizer.simple_remix_keyword("original keyword")

        # Verify generator was created
        mock_generator_class.assert_called_once_with(optimizer.config)

        # Verify simple_remix_keyword was called
        mock_generator.simple_remix_keyword.assert_called_once_with("original keyword")

        assert result == "remixed keyword"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_simple_remix_returns_none_if_generator_returns_none(self, mock_generator_class, optimizer):
        """Test handling when generator returns None"""
        mock_generator = Mock()
        mock_generator.simple_remix_keyword.return_value = None
        mock_generator_class.return_value = mock_generator

        result = optimizer.simple_remix_keyword("keyword")

        assert result is None


# ============================================================================
# Test Get Remix Keyword (LLM-based)
# ============================================================================

class TestGetRemixKeyword:
    """Test LLM-based keyword remixing"""

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_disabled_returns_none(self, mock_generator_class, optimizer):
        """Test that disabled remix returns None"""
        optimizer.download_config.zero_download_remix.enabled = False

        result = optimizer.get_remix_keyword("keyword", "topic")

        assert result is None

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_no_config_uses_simple(self, mock_generator_class, optimizer):
        """Test fallback to simple remix when config missing"""
        optimizer.download_config.zero_download_remix = None

        mock_generator = Mock()
        mock_generator.simple_remix_keyword.return_value = "simple remix"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword")

        mock_generator.simple_remix_keyword.assert_called_once_with("keyword")
        assert result == "simple remix"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_use_llm_false(self, mock_generator_class, optimizer):
        """Test simple remix when use_llm is False"""
        optimizer.download_config.zero_download_remix.use_llm = False

        mock_generator = Mock()
        mock_generator.simple_remix_keyword.return_value = "simple result"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword", "topic")

        mock_generator.simple_remix_keyword.assert_called_once_with("keyword")
        assert result == "simple result"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_llm_success(self, mock_generator_class, optimizer):
        """Test LLM-based remix when enabled"""
        optimizer.download_config.zero_download_remix.use_llm = True

        mock_generator = Mock()
        mock_generator.generate_single_alternative.return_value = "llm remix"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword", "travel")

        # Verify LLM generation was attempted
        mock_generator.generate_single_alternative.assert_called_once_with(
            keyword="keyword",
            topic="travel",
            provider="gemini",
            model="gemini-2.0-flash"
        )

        assert result == "llm remix"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_llm_fallback_to_simple(self, mock_generator_class, optimizer):
        """Test fallback to simple when LLM fails"""
        optimizer.download_config.zero_download_remix.use_llm = True

        mock_generator = Mock()
        mock_generator.generate_single_alternative.return_value = None  # LLM failed
        mock_generator.simple_remix_keyword.return_value = "simple fallback"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword")

        # Both methods should be called
        mock_generator.generate_single_alternative.assert_called_once()
        mock_generator.simple_remix_keyword.assert_called_once_with("keyword")

        assert result == "simple fallback"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_no_llm_config(self, mock_generator_class, optimizer):
        """Test fallback when LLM config missing"""
        optimizer.download_config.zero_download_remix.use_llm = True
        optimizer.download_config.llm_title_filter = None  # No LLM config

        mock_generator = Mock()
        mock_generator.simple_remix_keyword.return_value = "simple"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword")

        # Should fall back to simple
        mock_generator.simple_remix_keyword.assert_called_once_with("keyword")
        assert result == "simple"

    @patch('src.keyword_alternatives.KeywordAlternativeGenerator')
    def test_get_remix_keyword_dict_config(self, mock_generator_class, optimizer):
        """Test handling dict-based config (backward compatibility)"""
        # Replace mock object with dict
        optimizer.download_config.zero_download_remix = {
            'enabled': True,
            'use_llm': True
        }

        mock_generator = Mock()
        mock_generator.generate_single_alternative.return_value = "llm result"
        mock_generator_class.return_value = mock_generator

        result = optimizer.get_remix_keyword("keyword")

        # Should use LLM path
        mock_generator.generate_single_alternative.assert_called_once()
        assert result == "llm result"


# ============================================================================
# Test Adaptive Search Pool Sizing
# ============================================================================

class TestAdaptiveSearchPool:
    """Test adaptive search pool calculations"""

    def test_get_adaptive_pool_no_history(self, optimizer):
        """Test default pool when no history"""
        # max_downloads=10, multiplier=5 → 50, but min=30
        pool = optimizer.get_adaptive_search_pool("travel", max_downloads=10)
        assert pool == 50

    def test_get_adaptive_pool_below_minimum(self, optimizer):
        """Test minimum pool size enforcement"""
        # max_downloads=2, multiplier=5 → 10, but min=30
        pool = optimizer.get_adaptive_search_pool("travel", max_downloads=2)
        assert pool == 30

    def test_get_adaptive_pool_insufficient_data(self, optimizer):
        """Test default when insufficient history (< 2 entries)"""
        optimizer._search_pass_rates['travel'] = [0.3]  # Only 1 entry

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)
        assert pool == 50  # Default

    def test_get_adaptive_pool_very_low_pass_rate(self, optimizer):
        """Test pool expansion for very low pass rates (<= 0.01)"""
        optimizer._search_pass_rates['travel'] = [0.005, 0.008, 0.01]

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)

        # Should triple: 50 * 3 = 150, capped at max_pool=100
        assert pool == 100

    def test_get_adaptive_pool_low_pass_rate(self, optimizer):
        """Test pool expansion for low pass rates (< 0.2)"""
        optimizer._search_pass_rates['travel'] = [0.1, 0.15, 0.12]

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)

        # Should double: 50 * 2 = 100
        assert pool == 100

    def test_get_adaptive_pool_high_pass_rate(self, optimizer):
        """Test pool reduction for high pass rates (> 0.6)"""
        optimizer._search_pass_rates['travel'] = [0.7, 0.8, 0.75]

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)

        # Should reduce: 50 * 0.7 = 35
        assert pool == 35

    def test_get_adaptive_pool_normal_pass_rate(self, optimizer):
        """Test default pool for normal pass rates"""
        optimizer._search_pass_rates['travel'] = [0.3, 0.4, 0.5]

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)

        # Should use default
        assert pool == 50

    def test_get_adaptive_pool_uses_last_10_entries(self, optimizer):
        """Test that only last 10 entries are used"""
        # Create 20 entries: first 10 are low (0.1), last 10 are high (0.8)
        rates = [0.1] * 10 + [0.8] * 10
        optimizer._search_pass_rates['travel'] = rates

        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)

        # Should use last 10 (high pass rate) → smaller pool
        # 50 * 0.7 = 35
        assert pool == 35

    def test_get_adaptive_pool_respects_min_max(self, optimizer):
        """Test min/max pool enforcement"""
        optimizer.download_config.min_search_pool = 40
        optimizer.download_config.max_search_pool = 80

        # Test max cap (low pass rate would expand to 150)
        optimizer._search_pass_rates['travel'] = [0.01, 0.01]
        pool = optimizer.get_adaptive_search_pool("travel tour", max_downloads=10)
        assert pool == 80  # Capped at max

        # Test min floor (high pass rate would reduce to 35)
        optimizer._search_pass_rates['general'] = [0.8, 0.8]
        pool = optimizer.get_adaptive_search_pool("random", max_downloads=10)
        assert pool >= 40  # At least min


# ============================================================================
# Test Search Pass Rate Tracking
# ============================================================================

class TestSearchPassRateTracking:
    """Test pass rate recording and tracking"""

    def test_record_pass_rate_basic(self, optimizer):
        """Test basic pass rate recording"""
        optimizer.record_search_pass_rate("travel tour", searched=100, approved=30)

        assert 'travel' in optimizer._search_pass_rates
        assert len(optimizer._search_pass_rates['travel']) == 1
        assert optimizer._search_pass_rates['travel'][0] == 0.3

    def test_record_pass_rate_multiple_entries(self, optimizer):
        """Test recording multiple pass rates"""
        optimizer.record_search_pass_rate("travel tour", searched=100, approved=30)
        optimizer.record_search_pass_rate("paris tour", searched=50, approved=25)
        optimizer.record_search_pass_rate("aerial drone", searched=80, approved=40)

        # travel tour and paris tour → 'travel' category
        assert len(optimizer._search_pass_rates['travel']) == 2
        assert optimizer._search_pass_rates['travel'][0] == 0.3
        assert optimizer._search_pass_rates['travel'][1] == 0.5

        # aerial drone → 'cinematic' category
        assert len(optimizer._search_pass_rates['cinematic']) == 1
        assert optimizer._search_pass_rates['cinematic'][0] == 0.5

    def test_record_pass_rate_zero_searched_ignored(self, optimizer):
        """Test that zero searched count is ignored"""
        optimizer.record_search_pass_rate("travel", searched=0, approved=0)

        assert 'travel' not in optimizer._search_pass_rates

    def test_record_pass_rate_keeps_last_50(self, optimizer):
        """Test that only last 50 entries are kept"""
        # Record 60 entries
        for i in range(60):
            optimizer.record_search_pass_rate("travel tour", searched=100, approved=i)

        # Should keep only last 50
        assert len(optimizer._search_pass_rates['travel']) == 50

        # Should have the last 50 entries (10-59)
        assert optimizer._search_pass_rates['travel'][0] == 0.10  # Entry 10
        assert optimizer._search_pass_rates['travel'][-1] == 0.59  # Entry 59


# ============================================================================
# Test Source Diversity Tracking
# ============================================================================

class TestSourceDiversityTracking:
    """Test source diversity tracking and reporting"""

    def test_record_source_initial(self, optimizer):
        """Test recording first source for keyword"""
        optimizer.record_source_for_keyword("travel", "vid123")

        assert "travel" in optimizer._keyword_sources
        assert "vid123" in optimizer._keyword_sources["travel"]
        assert "vid123" in optimizer._source_keywords
        assert "travel" in optimizer._source_keywords["vid123"]

    def test_record_source_multiple_keywords_same_source(self, optimizer):
        """Test recording same source for multiple keywords"""
        optimizer.record_source_for_keyword("travel", "vid123")
        optimizer.record_source_for_keyword("vacation", "vid123")

        # Both keywords reference same video
        assert "vid123" in optimizer._keyword_sources["travel"]
        assert "vid123" in optimizer._keyword_sources["vacation"]

        # Video referenced by both keywords
        assert set(optimizer._source_keywords["vid123"]) == {"travel", "vacation"}

    def test_record_source_multiple_sources_same_keyword(self, optimizer):
        """Test recording multiple sources for same keyword"""
        optimizer.record_source_for_keyword("travel", "vid123")
        optimizer.record_source_for_keyword("travel", "vid456")

        # Keyword has multiple sources
        assert optimizer._keyword_sources["travel"] == {"vid123", "vid456"}

    def test_record_source_no_duplicates(self, optimizer):
        """Test that recording same source twice doesn't duplicate"""
        optimizer.record_source_for_keyword("travel", "vid123")
        optimizer.record_source_for_keyword("travel", "vid123")  # Duplicate

        # Should still be only one entry
        assert len(optimizer._keyword_sources["travel"]) == 1
        assert optimizer._source_keywords["vid123"] == ["travel"]

    def test_get_source_diversity_report_empty(self, optimizer):
        """Test diversity report with no data"""
        report = optimizer.get_source_diversity_report()

        assert report['total_keywords'] == 0
        assert report['total_unique_sources'] == 0
        assert report['sources_per_keyword'] == {}
        assert report['overlap_warnings'] == []
        assert report['heavily_reused_sources'] == []

    def test_get_source_diversity_report_no_overlap(self, optimizer):
        """Test diversity report with no overlap (each keyword has unique sources)"""
        optimizer.record_source_for_keyword("travel", "vid1")
        optimizer.record_source_for_keyword("vacation", "vid2")
        optimizer.record_source_for_keyword("beach", "vid3")

        report = optimizer.get_source_diversity_report()

        assert report['total_keywords'] == 3
        assert report['total_unique_sources'] == 3
        assert report['sources_per_keyword'] == {"travel": 1, "vacation": 1, "beach": 1}
        assert report['overlap_warnings'] == []
        assert report['heavily_reused_sources'] == []

    def test_get_source_diversity_report_with_overlap(self, optimizer):
        """Test diversity report with source overlap"""
        # vid1 used by 2 keywords
        optimizer.record_source_for_keyword("travel", "vid1")
        optimizer.record_source_for_keyword("vacation", "vid1")

        # vid2 unique
        optimizer.record_source_for_keyword("beach", "vid2")

        report = optimizer.get_source_diversity_report()

        assert report['total_keywords'] == 3
        assert report['total_unique_sources'] == 2
        assert len(report['overlap_warnings']) == 1

        # Check overlap details
        overlap = report['overlap_warnings'][0]
        assert overlap['video_id'] == "vid1"
        assert set(overlap['keywords']) == {"travel", "vacation"}
        assert overlap['count'] == 2

    def test_get_source_diversity_report_heavy_reuse(self, optimizer):
        """Test detection of heavily reused sources (3+ keywords)"""
        # vid1 used by 4 keywords
        for kw in ["travel", "vacation", "beach", "sunset"]:
            optimizer.record_source_for_keyword(kw, "vid1")

        # vid2 used by 2 keywords (not heavy)
        optimizer.record_source_for_keyword("mountain", "vid2")
        optimizer.record_source_for_keyword("hiking", "vid2")

        report = optimizer.get_source_diversity_report()

        assert len(report['overlap_warnings']) == 2  # Both overlaps detected
        assert len(report['heavily_reused_sources']) == 1  # Only vid1 is heavy

        heavy = report['heavily_reused_sources'][0]
        assert heavy['video_id'] == "vid1"
        assert heavy['count'] == 4

    def test_get_source_diversity_report_sorting(self, optimizer):
        """Test that overlaps are sorted by count (descending)"""
        # Create overlaps with different counts
        optimizer.record_source_for_keyword("kw1", "vid1")
        optimizer.record_source_for_keyword("kw2", "vid1")  # 2 keywords

        for kw in ["kw3", "kw4", "kw5"]:
            optimizer.record_source_for_keyword(kw, "vid2")  # 3 keywords

        for kw in ["kw6", "kw7", "kw8", "kw9"]:
            optimizer.record_source_for_keyword(kw, "vid3")  # 4 keywords

        report = optimizer.get_source_diversity_report()

        # Should be sorted: vid3 (4), vid2 (3), vid1 (2)
        assert report['overlap_warnings'][0]['count'] == 4
        assert report['overlap_warnings'][1]['count'] == 3
        assert report['overlap_warnings'][2]['count'] == 2


# ============================================================================
# Test Diversity Report Logging
# ============================================================================

class TestDiversityReportLogging:
    """Test diversity report logging"""

    def test_log_diversity_report_empty(self, optimizer, caplog):
        """Test logging with no data (should skip)"""
        import logging
        caplog.set_level(logging.INFO)

        optimizer.log_source_diversity_report()

        # Should not log anything when no data
        assert "INTER-KEYWORD SOURCE DIVERSITY REPORT" not in caplog.text

    def test_log_diversity_report_basic(self, optimizer, caplog):
        """Test basic diversity report logging"""
        import logging
        caplog.set_level(logging.INFO)

        optimizer.record_source_for_keyword("travel", "vid1")
        optimizer.record_source_for_keyword("vacation", "vid2")

        optimizer.log_source_diversity_report()

        # Check key log entries
        assert "INTER-KEYWORD SOURCE DIVERSITY REPORT" in caplog.text
        assert "Total keywords: 2" in caplog.text
        assert "Total unique video sources: 2" in caplog.text

    def test_log_diversity_report_with_overlap(self, optimizer, caplog):
        """Test logging with overlap warnings"""
        import logging
        caplog.set_level(logging.WARNING)

        # Create overlap
        optimizer.record_source_for_keyword("travel", "vid1")
        optimizer.record_source_for_keyword("vacation", "vid1")

        optimizer.log_source_diversity_report()

        # Check warning messages
        assert "Source overlap detected" in caplog.text
        assert "vid1" in caplog.text

    def test_log_diversity_report_heavy_reuse(self, optimizer, caplog):
        """Test logging with heavy reuse warnings"""
        import logging
        caplog.set_level(logging.WARNING)

        # Create heavy reuse
        for kw in ["kw1", "kw2", "kw3", "kw4"]:
            optimizer.record_source_for_keyword(kw, "vid1")

        optimizer.log_source_diversity_report()

        # Check heavy reuse warning
        assert "Heavily reused sources" in caplog.text
        assert "Consider diversifying keywords" in caplog.text


# ============================================================================
# Test Retry Keyword Generation
# ============================================================================

class TestRetryKeywordGeneration:
    """Test retry keyword generation for timeouts"""

    def test_get_retry_keyword_first_retry_long_keyword(self, optimizer):
        """Test first retry simplifies long keywords"""
        result = optimizer.get_retry_keyword("beautiful mountain landscape drone aerial view", retry_count=0)

        # Should take first 3 words + "footage"
        assert result == "beautiful mountain landscape footage"

    def test_get_retry_keyword_first_retry_short_keyword(self, optimizer):
        """Test first retry adds 'footage' to short keywords"""
        result = optimizer.get_retry_keyword("mountain view", retry_count=0)

        # Should add "footage"
        assert result == "mountain view footage"

    def test_get_retry_keyword_first_retry_already_has_footage(self, optimizer):
        """Test first retry when keyword already has 'footage'"""
        result = optimizer.get_retry_keyword("mountain footage", retry_count=0)

        # Should not duplicate "footage"
        assert result == "mountain footage"

    def test_get_retry_keyword_second_retry(self, optimizer):
        """Test second retry takes first 2 words"""
        result = optimizer.get_retry_keyword("beautiful mountain landscape", retry_count=1)

        # Should take first 2 words
        assert result == "beautiful mountain"

    def test_get_retry_keyword_second_retry_short(self, optimizer):
        """Test second retry with short keyword"""
        result = optimizer.get_retry_keyword("mountain", retry_count=1)

        # Less than 2 words, return as-is
        assert result == "mountain"

    def test_get_retry_keyword_third_retry(self, optimizer):
        """Test third+ retry returns original"""
        result = optimizer.get_retry_keyword("mountain landscape", retry_count=2)

        # No more modifications
        assert result == "mountain landscape"

    def test_get_retry_keyword_preserves_spacing(self, optimizer):
        """Test that spacing is preserved correctly"""
        result = optimizer.get_retry_keyword("one two three four", retry_count=0)
        assert result == "one two three footage"

        result = optimizer.get_retry_keyword("one two three four", retry_count=1)
        assert result == "one two"
