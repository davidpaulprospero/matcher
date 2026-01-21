#!/usr/bin/env python3
"""
Integration Tests for High Matches Mode and Cross-Project Learning

Tests the following recent additions:
1. Coverage Analyzer - analyze match results and identify weak segments
2. Recovery Keywords - generate targeted keywords for weak segments
3. Client Profiles - cross-project learning and profile management
4. High Matches Logger - dedicated logging for iteration tracking
5. Iterative Match Stage - stop conditions and state management

Usage:
    python tests/test_high_matches_integration.py
    python tests/test_high_matches_integration.py --verbose
    pytest tests/test_high_matches_integration.py -v
"""

import json
import os
import sys
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_box(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(text: str):
    print(f"\n  --- {text} ---")


def print_result(name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


# =============================================================================
# Test 1: Coverage Analyzer
# =============================================================================

class TestCoverageAnalyzer:
    """Tests for coverage_analyzer.py"""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_coverage_'))

    def teardown_method(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_analyze_coverage_all_high_confidence(self):
        """Test coverage analysis when all matches are high confidence."""
        from src.matching.coverage_analyzer import analyze_coverage, CoverageReport
        from src.state import VoiceoverSegment

        # Create test segments
        segments = [
            VoiceoverSegment(index=i, start=i*10.0, end=(i+1)*10.0, text=f"Segment {i}")
            for i in range(5)
        ]

        # Create mock matches with high confidence (using dict format)
        matches = [
            {
                'segment_index': i,
                'confidence': 0.95,
                'video_file': f'video_{i}.mp4'
            }
            for i in range(5)
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.total_segments == 5
        assert report.high_confidence == 5
        assert report.medium_confidence == 0
        assert report.low_confidence == 0
        assert report.coverage_ratio == 1.0
        assert len(report.weak_segments) == 0
        print_result("All high confidence coverage", True)

    def test_analyze_coverage_mixed_confidence(self):
        """Test coverage analysis with mixed confidence levels."""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        segments = [
            VoiceoverSegment(index=i, start=i*10.0, end=(i+1)*10.0, text=f"Segment {i}")
            for i in range(10)
        ]

        # Mix of confidences: 3 high, 4 medium, 3 low
        confidences = [0.95, 0.92, 0.91,  # high (>= 0.90)
                       0.85, 0.80, 0.75, 0.72,  # medium (0.70-0.90)
                       0.65, 0.50, 0.30]  # low (< 0.70)

        matches = [
            {'segment_index': i, 'confidence': conf, 'video_file': f'v{i}.mp4'}
            for i, conf in enumerate(confidences)
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.total_segments == 10
        assert report.high_confidence == 3
        assert report.medium_confidence == 4
        assert report.low_confidence == 3
        assert report.coverage_ratio == 0.3  # 3/10
        assert len(report.weak_segments) == 7  # medium + low
        print_result("Mixed confidence coverage", True)

    def test_analyze_coverage_empty_segments(self):
        """Test coverage analysis with no segments."""
        from src.matching.coverage_analyzer import analyze_coverage

        report = analyze_coverage(matches=[], voiceover_segments=[])

        assert report.total_segments == 0
        assert report.coverage_ratio == 0.0
        assert len(report.weak_segments) == 0
        print_result("Empty segments coverage", True)

    def test_weak_segments_sorted_by_confidence(self):
        """Test that weak segments are sorted by confidence (lowest first)."""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        segments = [
            VoiceoverSegment(index=i, start=i*10.0, end=(i+1)*10.0, text=f"Seg {i}")
            for i in range(5)
        ]

        # All medium/low confidence, different values
        matches = [
            {'segment_index': 0, 'confidence': 0.80, 'video_file': 'v0.mp4'},
            {'segment_index': 1, 'confidence': 0.50, 'video_file': 'v1.mp4'},
            {'segment_index': 2, 'confidence': 0.65, 'video_file': 'v2.mp4'},
            {'segment_index': 3, 'confidence': 0.30, 'video_file': 'v3.mp4'},
            {'segment_index': 4, 'confidence': 0.75, 'video_file': 'v4.mp4'},
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        # Should be sorted by confidence ascending
        confidences = [ws.current_confidence for ws in report.weak_segments]
        assert confidences == sorted(confidences)
        assert confidences[0] == 0.30  # Lowest first
        print_result("Weak segments sorted by confidence", True)

    def test_get_improvement_delta(self):
        """Test improvement delta calculation."""
        from src.matching.coverage_analyzer import get_improvement_delta

        # Positive improvement
        delta, improving = get_improvement_delta(0.50, 0.60, min_improvement=0.02)
        assert abs(delta - 0.10) < 0.001  # Floating point tolerance
        assert improving is True

        # No improvement
        delta, improving = get_improvement_delta(0.50, 0.51, min_improvement=0.02)
        assert abs(delta - 0.01) < 0.001
        assert improving is False

        # Negative delta
        delta, improving = get_improvement_delta(0.60, 0.55, min_improvement=0.02)
        assert abs(delta - (-0.05)) < 0.001
        assert improving is False

        print_result("Improvement delta calculation", True)

    def test_coverage_report_summary(self):
        """Test CoverageReport summary method."""
        from src.matching.coverage_analyzer import CoverageReport

        report = CoverageReport(
            total_segments=10,
            high_confidence=6,
            medium_confidence=3,
            low_confidence=1,
            coverage_ratio=0.60,
            target_confidence=0.90,
        )

        summary = report.summary()
        assert "60.0%" in summary or "60%" in summary
        assert "6/10" in summary
        assert "High: 6" in summary
        print_result("Coverage report summary", True)

    def run_all(self):
        """Run all coverage analyzer tests."""
        print_box("Coverage Analyzer Tests")
        passed = 0
        total = 0

        for method in dir(self):
            if method.startswith('test_'):
                total += 1
                try:
                    self.setup_method()
                    getattr(self, method)()
                    passed += 1
                except AssertionError as e:
                    print_result(method, False, str(e))
                except Exception as e:
                    print_result(method, False, f"Error: {e}")
                finally:
                    self.teardown_method()

        print(f"\n  Coverage Analyzer: {passed}/{total} tests passed")
        return passed == total


# =============================================================================
# Test 2: Recovery Keywords
# =============================================================================

class TestRecoveryKeywords:
    """Tests for recovery_keywords.py"""

    def test_extract_keywords_from_text_basic(self):
        """Test basic keyword extraction."""
        from src.matching.recovery_keywords import extract_keywords_from_text

        text = "The beautiful sunset over the ocean waves creates a magical atmosphere"
        keywords = extract_keywords_from_text(text, max_keywords=5)

        # Should extract meaningful words, not stop words
        assert len(keywords) <= 5
        assert len(keywords) > 0
        assert "the" not in keywords
        # Should have content words (over may be included as it's not always filtered)
        assert any(kw in keywords for kw in ["beautiful", "sunset", "ocean", "waves", "magical", "atmosphere", "creates", "over"])
        print_result("Basic keyword extraction", True)

    def test_extract_keywords_filters_stopwords(self):
        """Test that stop words are filtered."""
        from src.matching.recovery_keywords import extract_keywords_from_text

        text = "the a an and or but this that these those"
        keywords = extract_keywords_from_text(text, max_keywords=10)

        assert len(keywords) == 0  # All stop words
        print_result("Stop word filtering", True)

    def test_extract_keywords_min_length(self):
        """Test minimum word length filtering."""
        from src.matching.recovery_keywords import extract_keywords_from_text

        text = "a big cat in the old box by an oak"
        keywords = extract_keywords_from_text(text, max_keywords=10, min_word_length=3)

        # 'big', 'cat', 'old', 'box', 'oak' have 3+ chars (but some are stop words)
        for kw in keywords:
            assert len(kw) >= 3
        print_result("Min word length filtering", True)

    def test_generate_recovery_keywords_from_weak_segments(self):
        """Test recovery keyword generation from weak segments."""
        from src.matching.recovery_keywords import generate_recovery_keywords
        from src.matching.coverage_analyzer import WeakSegment

        weak_segments = [
            WeakSegment(
                segment_id="S001",
                segment_index=1,
                text="Beautiful mountain landscape with snow peaks",
                current_confidence=0.50,
            ),
            WeakSegment(
                segment_id="S002",
                segment_index=2,
                text="Ocean waves crashing on sandy beach",
                current_confidence=0.45,
            ),
        ]

        keywords = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=["nature", "scenery"],
            strategy="weak_segments",
            max_keywords=10,
        )

        # Should generate new keywords not in existing
        assert len(keywords) > 0
        assert "nature" not in [k.lower() for k in keywords]
        assert "scenery" not in [k.lower() for k in keywords]
        # Keywords should have "footage" suffix added
        assert any("footage" in kw.lower() for kw in keywords)
        print_result("Recovery keywords from weak segments", True)

    def test_generate_recovery_keywords_empty_input(self):
        """Test recovery keywords with empty weak segments."""
        from src.matching.recovery_keywords import generate_recovery_keywords

        keywords = generate_recovery_keywords(
            weak_segments=[],
            existing_keywords=["test"],
            strategy="weak_segments",
        )

        assert keywords == []
        print_result("Empty weak segments handling", True)

    def test_generate_recovery_keywords_deduplication(self):
        """Test that duplicate keywords are removed."""
        from src.matching.recovery_keywords import generate_recovery_keywords
        from src.matching.coverage_analyzer import WeakSegment

        # Create segments with overlapping keywords
        weak_segments = [
            WeakSegment(
                segment_id="S001",
                segment_index=1,
                text="mountain mountain mountain landscape",
                current_confidence=0.50,
            ),
        ]

        keywords = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=10,
        )

        # Should not have duplicates
        lowercase_keywords = [k.lower() for k in keywords]
        assert len(lowercase_keywords) == len(set(lowercase_keywords))
        print_result("Keyword deduplication", True)

    def test_format_keywords_for_search(self):
        """Test keyword formatting for search."""
        from src.matching.recovery_keywords import format_keywords_for_search

        keywords = ["mountain", "ocean waves", "sunset"]
        formatted = format_keywords_for_search(keywords, add_suffix=True, suffix="footage")

        assert formatted[0] == "mountain footage"
        assert formatted[1] == "ocean waves footage"
        assert formatted[2] == "sunset footage"

        # Test without suffix
        formatted_no_suffix = format_keywords_for_search(keywords, add_suffix=False)
        assert formatted_no_suffix == keywords
        print_result("Keyword formatting for search", True)

    def test_format_keywords_skips_existing_suffix(self):
        """Test that formatting skips keywords that already have suffixes."""
        from src.matching.recovery_keywords import format_keywords_for_search

        # Use lowercase suffix to match the case-insensitive check
        keywords = ["mountain footage", "ocean video", "sunset"]
        formatted = format_keywords_for_search(keywords, add_suffix=True, suffix="footage")

        # First two already have suffixes (footage, video), should not double-add
        assert formatted[0] == "mountain footage"  # unchanged (has "footage")
        assert formatted[1] == "ocean video"  # unchanged (has "video")
        assert formatted[2] == "sunset footage"  # suffix added
        print_result("Skip existing suffix", True)

    def run_all(self):
        """Run all recovery keyword tests."""
        print_box("Recovery Keywords Tests")
        passed = 0
        total = 0

        for method in dir(self):
            if method.startswith('test_'):
                total += 1
                try:
                    getattr(self, method)()
                    passed += 1
                except AssertionError as e:
                    print_result(method, False, str(e))
                except Exception as e:
                    print_result(method, False, f"Error: {e}")

        print(f"\n  Recovery Keywords: {passed}/{total} tests passed")
        return passed == total


# =============================================================================
# Test 3: Client Profiles
# =============================================================================

class TestClientProfiles:
    """Tests for client_profiles.py"""

    def setup_method(self):
        """Setup test environment with temp directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_client_'))
        # Patch CLIENT_PROFILES_DIR to use temp directory
        import src.feedback.client_profiles as cp
        self._original_dir = cp.CLIENT_PROFILES_DIR
        cp.CLIENT_PROFILES_DIR = self.temp_dir

    def teardown_method(self):
        """Cleanup."""
        import src.feedback.client_profiles as cp
        cp.CLIENT_PROFILES_DIR = self._original_dir
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_new_client_profile(self):
        """Test creating a new client profile."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="test_client")

        assert profile.client_id == "test_client"
        assert profile.display_name == "Test_Client"
        assert profile.created != ""
        assert isinstance(profile.blacklist_channels, set)
        assert isinstance(profile.blacklist_keywords, set)
        print_result("Create new client profile", True)

    def test_profile_save_and_load(self):
        """Test saving and loading a profile."""
        from src.feedback.client_profiles import ClientProfile

        # Create and save
        profile = ClientProfile(client_id="save_test")
        profile.add_blacklist_channel("Bad Channel")
        profile.add_blacklist_keyword("spam")
        profile.add_project("/path/to/project1")
        profile.record_acceptance(5)
        profile.record_rejection(2)
        profile.save()

        # Load
        loaded = ClientProfile.load("save_test")

        assert loaded is not None
        assert loaded.client_id == "save_test"
        assert "Bad Channel" in loaded.blacklist_channels
        assert "spam" in loaded.blacklist_keywords
        assert "/path/to/project1" in loaded.projects
        assert loaded.total_videos_accepted == 5
        assert loaded.total_videos_rejected == 2
        print_result("Profile save and load", True)

    def test_channel_blacklist_case_insensitive(self):
        """Test that channel blacklist matching is case-insensitive."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="case_test")
        profile.add_blacklist_channel("Bad Channel")

        assert profile.is_channel_blacklisted("bad channel")
        assert profile.is_channel_blacklisted("BAD CHANNEL")
        assert profile.is_channel_blacklisted("Bad Channel")
        assert not profile.is_channel_blacklisted("Good Channel")
        print_result("Channel blacklist case-insensitive", True)

    def test_keyword_blacklist_matching(self):
        """Test keyword blacklist matching in titles."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="keyword_test")
        profile.add_blacklist_keyword("spam")
        profile.add_blacklist_keyword("scam")

        assert profile.is_keyword_blacklisted("This is spam content")
        assert profile.is_keyword_blacklisted("SPAM VIDEO")
        assert profile.is_keyword_blacklisted("Total Scam Alert")
        assert not profile.is_keyword_blacklisted("Good content here")
        print_result("Keyword blacklist matching", True)

    def test_whitelist_overrides_blacklist(self):
        """Test that whitelist overrides global rejection."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="whitelist_test")
        profile.whitelist_channels.add("Trusted Channel")

        assert profile.is_channel_whitelisted("trusted channel")
        assert profile.is_channel_whitelisted("TRUSTED CHANNEL")
        assert not profile.is_channel_whitelisted("Other Channel")
        print_result("Whitelist override check", True)

    def test_get_or_create_client_profile(self):
        """Test get_or_create helper function."""
        from src.feedback.client_profiles import get_or_create_client_profile

        # First call creates
        profile1 = get_or_create_client_profile("new_client")
        assert profile1.client_id == "new_client"

        # Second call loads existing
        profile2 = get_or_create_client_profile("new_client")
        assert profile2.client_id == "new_client"
        assert profile2.created == profile1.created
        print_result("Get or create profile", True)

    def test_list_client_profiles(self):
        """Test listing all client profiles."""
        from src.feedback.client_profiles import (
            ClientProfile, list_client_profiles
        )

        # Create several profiles
        ClientProfile(client_id="alice").save()
        ClientProfile(client_id="bob").save()
        ClientProfile(client_id="charlie").save()

        clients = list_client_profiles()

        assert len(clients) == 3
        assert "alice" in clients
        assert "bob" in clients
        assert "charlie" in clients
        print_result("List client profiles", True)

    def test_content_preferences_defaults(self):
        """Test ContentPreferences default values."""
        from src.feedback.client_profiles import ContentPreferences

        prefs = ContentPreferences()

        assert prefs.content_style == "documentary"
        assert prefs.preferred_duration_min == 30.0
        assert prefs.preferred_duration_max == 180.0
        assert prefs.avoid_trainers is True
        assert prefs.avoid_movies is True
        assert prefs.avoid_music is True
        print_result("Content preferences defaults", True)

    def test_quality_thresholds_defaults(self):
        """Test QualityThresholds default values."""
        from src.feedback.client_profiles import QualityThresholds

        thresholds = QualityThresholds()

        assert thresholds.min_channel_score == 0.3
        assert thresholds.min_llm_relevance == 0.6
        assert thresholds.min_embedding_score == 0.4
        assert thresholds.max_videos_per_keyword == 12
        print_result("Quality thresholds defaults", True)

    def test_profile_post_init_conversions(self):
        """Test that __post_init__ converts dicts to dataclasses."""
        from src.feedback.client_profiles import ClientProfile

        # Simulate loading from YAML (dicts)
        profile = ClientProfile(
            client_id="convert_test",
            preferences={"content_style": "raw", "avoid_music": False},
            thresholds={"min_channel_score": 0.5},
            blacklist_channels=["ch1", "ch2"],  # list instead of set
        )

        # Should be converted to proper types
        assert hasattr(profile.preferences, 'content_style')
        assert profile.preferences.content_style == "raw"
        assert profile.preferences.avoid_music is False
        assert hasattr(profile.thresholds, 'min_channel_score')
        assert profile.thresholds.min_channel_score == 0.5
        assert isinstance(profile.blacklist_channels, set)
        assert "ch1" in profile.blacklist_channels
        print_result("Post-init type conversions", True)

    def run_all(self):
        """Run all client profile tests."""
        print_box("Client Profiles Tests")
        passed = 0
        total = 0

        for method in dir(self):
            if method.startswith('test_'):
                total += 1
                try:
                    self.setup_method()
                    getattr(self, method)()
                    passed += 1
                except AssertionError as e:
                    print_result(method, False, str(e))
                except Exception as e:
                    print_result(method, False, f"Error: {e}")
                finally:
                    self.teardown_method()

        print(f"\n  Client Profiles: {passed}/{total} tests passed")
        return passed == total


# =============================================================================
# Test 4: High Matches Logger
# =============================================================================

class TestHighMatchesLogger:
    """Tests for high_matches_logger.py"""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_hmm_log_'))

    def teardown_method(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_logger_creates_files(self):
        """Test that logger creates log and JSON files."""
        from src.matching.high_matches_logger import HighMatchesLogger

        logger = HighMatchesLogger(self.temp_dir, session_id="test_session")

        assert logger.log_file.exists()
        assert logger.log_file.name == "high_matches_test_session.log"
        assert logger.json_file.name == "high_matches_test_session.json"
        print_result("Logger creates files", True)

    def test_logger_auto_generates_session_id(self):
        """Test that logger auto-generates session ID."""
        from src.matching.high_matches_logger import HighMatchesLogger

        logger = HighMatchesLogger(self.temp_dir)

        assert logger.session_id is not None
        assert len(logger.session_id) > 0
        assert "_" in logger.session_id  # Format: YYYYMMDD_HHMMSS
        print_result("Auto-generate session ID", True)

    def test_log_entry_to_dict(self):
        """Test HighMatchesLogEntry serialization."""
        from src.matching.high_matches_logger import HighMatchesLogEntry
        from datetime import datetime, timezone

        entry = HighMatchesLogEntry(
            timestamp=datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc),
            entry_type="test",
            iteration=1,
            data={"key": "value"},
            session_id="test_123"
        )

        d = entry.to_dict()

        assert d['entry_type'] == "test"
        assert d['iteration'] == 1
        assert d['data'] == {"key": "value"}
        assert "2024-01-15" in d['timestamp']
        print_result("Log entry to_dict", True)

    def test_log_coverage_analysis(self):
        """Test logging coverage analysis."""
        from src.matching.high_matches_logger import HighMatchesLogger
        from src.matching.coverage_analyzer import CoverageReport

        logger = HighMatchesLogger(self.temp_dir, session_id="coverage_test")

        report = CoverageReport(
            total_segments=10,
            high_confidence=6,
            medium_confidence=3,
            low_confidence=1,
            coverage_ratio=0.60,
            target_confidence=0.90,
        )

        # Mock config and state for session start
        mock_config = MagicMock()
        mock_config.matching.high_matches_mode.target_confidence = 0.90
        mock_config.matching.high_matches_mode.coverage_target = 0.85
        mock_config.matching.high_matches_mode.max_iterations = 3
        mock_config.matching.high_matches_mode.videos_per_iteration = 10
        mock_config.matching.high_matches_mode.keyword_strategy = "llm"

        mock_state = MagicMock()
        mock_state.voiceover_segments = [1, 2, 3]
        mock_state.matches = [1, 2]
        mock_state.downloaded_videos = [1]
        mock_state.keywords = ["test"]

        logger.log_session_start(mock_config, mock_state)
        logger.log_coverage_analysis(report, iteration=1)

        # Check log file contains coverage info
        content = logger.log_file.read_text()
        assert "60" in content or "0.6" in content
        print_result("Log coverage analysis", True)

    def test_get_log_paths(self):
        """Test getting log file paths."""
        from src.matching.high_matches_logger import HighMatchesLogger

        logger = HighMatchesLogger(self.temp_dir, session_id="paths_test")
        paths = logger.get_log_paths()

        assert 'text' in paths
        assert 'json' in paths
        assert paths['text'] == logger.log_file
        assert paths['json'] == logger.json_file
        print_result("Get log paths", True)

    def run_all(self):
        """Run all high matches logger tests."""
        print_box("High Matches Logger Tests")
        passed = 0
        total = 0

        for method in dir(self):
            if method.startswith('test_'):
                total += 1
                try:
                    self.setup_method()
                    getattr(self, method)()
                    passed += 1
                except AssertionError as e:
                    print_result(method, False, str(e))
                except Exception as e:
                    print_result(method, False, f"Error: {e}")
                finally:
                    self.teardown_method()

        print(f"\n  High Matches Logger: {passed}/{total} tests passed")
        return passed == total


# =============================================================================
# Test 5: Iterative Match Stage
# =============================================================================

class TestIterativeMatchStage:
    """Tests for iterative_match.py stop conditions and state management."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_iterative_'))

    def teardown_method(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_iterative_match_state_defaults(self):
        """Test IterativeMatchState default values."""
        from src.state import IterativeMatchState

        state = IterativeMatchState()

        assert state.iteration_count == 0
        assert state.coverage_history == []
        assert state.videos_added_per_iteration == []
        assert state.final_coverage == 0.0
        assert state.target_achieved is False
        assert state.weak_segment_count == 0
        print_result("IterativeMatchState defaults", True)

    def test_should_stop_coverage_target_met(self):
        """Test _should_stop returns True when coverage target met."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()

        # Mock config
        mock_config = MagicMock()
        mock_config.matching.high_matches_mode.coverage_target = 0.85
        mock_config.matching.high_matches_mode.max_iterations = 3

        # Coverage at target
        coverage = CoverageReport(
            total_segments=10,
            high_confidence=9,
            medium_confidence=1,
            low_confidence=0,
            coverage_ratio=0.90,  # Above 0.85 target
            target_confidence=0.90,
        )

        iter_state = IterativeMatchState(iteration_count=1)

        should_stop = stage._should_stop(coverage, iter_state, mock_config)
        assert should_stop is True
        print_result("Stop when coverage target met", True)

    def test_should_stop_max_iterations_reached(self):
        """Test _should_stop returns True at max iterations."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()

        mock_config = MagicMock()
        mock_config.matching.high_matches_mode.coverage_target = 0.90
        mock_config.matching.high_matches_mode.max_iterations = 3

        # Coverage below target
        coverage = CoverageReport(
            total_segments=10,
            high_confidence=5,
            medium_confidence=3,
            low_confidence=2,
            coverage_ratio=0.50,
            target_confidence=0.90,
            weak_segments=[MagicMock()],  # Has weak segments
        )

        # At max iterations
        iter_state = IterativeMatchState(iteration_count=3)

        should_stop = stage._should_stop(coverage, iter_state, mock_config)
        assert should_stop is True
        print_result("Stop at max iterations", True)

    def test_should_stop_no_weak_segments(self):
        """Test _should_stop returns True when no weak segments."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()

        mock_config = MagicMock()
        mock_config.matching.high_matches_mode.coverage_target = 0.90
        mock_config.matching.high_matches_mode.max_iterations = 5

        # No weak segments (even if below target somehow)
        coverage = CoverageReport(
            total_segments=10,
            high_confidence=7,
            medium_confidence=0,
            low_confidence=0,
            coverage_ratio=0.70,  # Below target
            target_confidence=0.90,
            weak_segments=[],  # Empty!
        )

        iter_state = IterativeMatchState(iteration_count=1)

        should_stop = stage._should_stop(coverage, iter_state, mock_config)
        assert should_stop is True
        print_result("Stop when no weak segments", True)

    def test_should_continue_when_conditions_allow(self):
        """Test _should_stop returns False when iteration should continue."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport, WeakSegment
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()

        mock_config = MagicMock()
        mock_config.matching.high_matches_mode.coverage_target = 0.90
        mock_config.matching.high_matches_mode.max_iterations = 5

        # Below target, has weak segments, within iteration limit
        coverage = CoverageReport(
            total_segments=10,
            high_confidence=5,
            medium_confidence=3,
            low_confidence=2,
            coverage_ratio=0.50,
            target_confidence=0.90,
            weak_segments=[WeakSegment("S001", 1, "text", 0.5)],
        )

        iter_state = IterativeMatchState(iteration_count=2)

        should_stop = stage._should_stop(coverage, iter_state, mock_config)
        assert should_stop is False
        print_result("Continue when conditions allow", True)

    def test_get_result_data(self):
        """Test _get_result_data for checkpointing."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()

        iter_state = IterativeMatchState(
            iteration_count=2,
            coverage_history=[0.50, 0.65, 0.80],
            videos_added_per_iteration=[10, 8],
            final_coverage=0.80,
            target_achieved=False,
            weak_segment_count=3,
        )

        data = stage._get_result_data(iter_state)

        assert data['iteration_count'] == 2
        assert data['coverage_history'] == [0.50, 0.65, 0.80]
        assert data['videos_added_per_iteration'] == [10, 8]
        assert data['final_coverage'] == 0.80
        assert data['target_achieved'] is False
        assert data['weak_segment_count'] == 3
        print_result("Result data for checkpoint", True)

    def test_stage_restore(self):
        """Test restoring stage from checkpoint."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import PipelineState

        stage = IterativeMatchStage()

        # Mock checkpoint manager
        mock_checkpoint = MagicMock()
        mock_checkpoint.get_stage_data.return_value = {
            'iteration_count': 2,
            'coverage_history': [0.50, 0.70],
            'videos_added_per_iteration': [10, 5],
            'final_coverage': 0.70,
            'target_achieved': False,
            'weak_segment_count': 5,
        }

        # Create minimal state
        state = MagicMock()
        state.iterative_match_state = None

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert state.iterative_match_state is not None
        assert state.iterative_match_state.iteration_count == 2
        assert state.iterative_match_state.coverage_history == [0.50, 0.70]
        print_result("Stage restore from checkpoint", True)

    def test_stage_restore_no_data(self):
        """Test restore returns False when no checkpoint data."""
        from src.stages.iterative_match import IterativeMatchStage

        stage = IterativeMatchStage()

        mock_checkpoint = MagicMock()
        mock_checkpoint.get_stage_data.return_value = None

        state = MagicMock()

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        print_result("Stage restore with no data", True)

    def run_all(self):
        """Run all iterative match stage tests."""
        print_box("Iterative Match Stage Tests")
        passed = 0
        total = 0

        for method in dir(self):
            if method.startswith('test_'):
                total += 1
                try:
                    self.setup_method()
                    getattr(self, method)()
                    passed += 1
                except AssertionError as e:
                    print_result(method, False, str(e))
                except Exception as e:
                    print_result(method, False, f"Error: {e}")
                finally:
                    self.teardown_method()

        print(f"\n  Iterative Match Stage: {passed}/{total} tests passed")
        return passed == total


# =============================================================================
# Test Runner
# =============================================================================

class HighMatchesIntegrationRunner:
    """Main test runner for high matches mode integration tests."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}

    def run_all_tests(self):
        """Run all test suites."""
        print_box("HIGH MATCHES MODE INTEGRATION TESTS")

        test_suites = [
            ("Coverage Analyzer", TestCoverageAnalyzer()),
            ("Recovery Keywords", TestRecoveryKeywords()),
            ("Client Profiles", TestClientProfiles()),
            ("High Matches Logger", TestHighMatchesLogger()),
            ("Iterative Match Stage", TestIterativeMatchStage()),
        ]

        all_passed = True
        for name, suite in test_suites:
            try:
                passed = suite.run_all()
                self.results[name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"\n  ERROR running {name}: {e}")
                self.results[name] = False
                all_passed = False

        # Print summary
        print_box("SUMMARY")
        total_suites = len(self.results)
        passed_suites = sum(1 for v in self.results.values() if v)

        for name, passed in self.results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {name}")

        print(f"\n  Overall: {passed_suites}/{total_suites} test suites passed")

        return all_passed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="High Matches Mode Integration Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    runner = HighMatchesIntegrationRunner(verbose=args.verbose)
    success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
