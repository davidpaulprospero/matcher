"""
Tests for High Matches Logger

Tests the dedicated logging system for tracking iterative matching
progress and coverage analysis.
"""

import pytest
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from src.matching.high_matches_logger import (
    HighMatchesLogEntry,
    HighMatchesLogger,
)
from src.matching.coverage_analyzer import CoverageReport, WeakSegment


class TestHighMatchesLogEntry:
    """Tests for HighMatchesLogEntry dataclass"""

    def test_basic_creation(self):
        """Test creating a log entry"""
        now = datetime.now(timezone.utc)
        entry = HighMatchesLogEntry(
            timestamp=now,
            entry_type="test_entry",
            iteration=1,
        )

        assert entry.timestamp == now
        assert entry.entry_type == "test_entry"
        assert entry.iteration == 1
        assert entry.data == {}
        assert entry.session_id == ""

    def test_with_data(self):
        """Test entry with additional data"""
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="coverage_analysis",
            iteration=2,
            data={"coverage": 0.85, "weak_count": 5},
            session_id="test_session",
        )

        assert entry.data["coverage"] == 0.85
        assert entry.data["weak_count"] == 5
        assert entry.session_id == "test_session"

    def test_to_dict(self):
        """Test serialization to dict"""
        now = datetime.now(timezone.utc)
        entry = HighMatchesLogEntry(
            timestamp=now,
            entry_type="iteration_start",
            iteration=3,
            data={"test": "value"},
            session_id="session123",
        )

        d = entry.to_dict()

        assert d["timestamp"] == now.isoformat()
        assert d["entry_type"] == "iteration_start"
        assert d["iteration"] == 3
        assert d["data"] == {"test": "value"}
        assert d["session_id"] == "session123"


class TestHighMatchesLogger:
    """Tests for HighMatchesLogger class"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing"""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.target_confidence = 0.90
        config.matching.high_matches_mode.coverage_target = 0.85
        config.matching.high_matches_mode.max_iterations = 5
        config.matching.high_matches_mode.videos_per_iteration = 10
        config.matching.high_matches_mode.keyword_strategy = "llm"
        return config

    @pytest.fixture
    def mock_state(self):
        """Create mock state for testing"""
        state = Mock()
        state.voiceover_segments = [Mock() for _ in range(10)]
        state.matches = [Mock() for _ in range(10)]
        state.downloaded_videos = [Mock() for _ in range(20)]
        state.keywords = ["test1", "test2"]
        return state

    def test_logger_creation(self, temp_log_dir):
        """Test creating a logger"""
        logger = HighMatchesLogger(temp_log_dir)

        assert logger.log_dir == temp_log_dir
        assert logger.session_id is not None
        assert len(logger.entries) == 0
        assert logger.log_file.exists()

    def test_auto_generated_session_id(self, temp_log_dir):
        """Test session ID auto-generation"""
        logger = HighMatchesLogger(temp_log_dir)

        # Session ID should be timestamp-based
        assert len(logger.session_id) > 0
        # Format: YYYYMMDD_HHMMSS
        assert "_" in logger.session_id

    def test_custom_session_id(self, temp_log_dir):
        """Test custom session ID"""
        logger = HighMatchesLogger(temp_log_dir, session_id="custom_123")

        assert logger.session_id == "custom_123"

    def test_log_files_created(self, temp_log_dir):
        """Test that log files are created"""
        logger = HighMatchesLogger(temp_log_dir)

        assert logger.log_file.exists()
        assert logger.log_file.suffix == ".log"

    def test_get_log_paths(self, temp_log_dir):
        """Test get_log_paths returns correct paths"""
        logger = HighMatchesLogger(temp_log_dir, session_id="test")

        paths = logger.get_log_paths()

        assert "text" in paths
        assert "json" in paths
        assert paths["text"].name == "high_matches_test.log"
        assert paths["json"].name == "high_matches_test.json"

    def test_log_session_start(self, temp_log_dir, mock_config, mock_state):
        """Test logging session start"""
        logger = HighMatchesLogger(temp_log_dir)
        logger.log_session_start(mock_config, mock_state)

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "session_start"
        assert "config" in logger.entries[0].data
        assert "initial_state" in logger.entries[0].data

    def test_log_coverage_analysis(self, temp_log_dir):
        """Test logging coverage analysis"""
        logger = HighMatchesLogger(temp_log_dir)

        report = CoverageReport(
            total_segments=100,
            high_confidence=70,
            medium_confidence=20,
            low_confidence=10,
            coverage_ratio=0.70,
            weak_segments=[
                WeakSegment("S001", 0, "test", 0.5),
            ],
        )

        logger.log_coverage_analysis(report, iteration=1)

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "coverage_analysis"
        assert logger.entries[0].iteration == 1
        assert logger.entries[0].data["total_segments"] == 100
        assert logger.entries[0].data["coverage_ratio"] == 0.70

    def test_log_recovery_keywords(self, temp_log_dir):
        """Test logging recovery keywords"""
        logger = HighMatchesLogger(temp_log_dir)

        keywords = ["ocean footage", "mountain 4K", "cityscape video"]
        logger.log_recovery_keywords(keywords, strategy="llm", weak_count=15, iteration=2)

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "recovery_keywords"
        assert logger.entries[0].data["keywords"] == keywords
        assert logger.entries[0].data["strategy"] == "llm"
        assert logger.entries[0].data["weak_segment_count"] == 15

    def test_log_iteration_start(self, temp_log_dir):
        """Test logging iteration start"""
        logger = HighMatchesLogger(temp_log_dir)

        logger.log_iteration_start(iteration=3, coverage=0.65, weak_count=35)

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "iteration_start"
        assert logger.entries[0].iteration == 3
        assert logger.entries[0].data["current_coverage"] == 0.65
        assert logger.entries[0].data["weak_segment_count"] == 35

    def test_log_download_result(self, temp_log_dir):
        """Test logging download results"""
        logger = HighMatchesLogger(temp_log_dir)

        logger.log_download_result(
            videos_added=15,
            total_videos=100,
            success=True,
            iteration=2,
        )

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "download_result"
        assert logger.entries[0].data["videos_added"] == 15
        assert logger.entries[0].data["total_videos"] == 100
        assert logger.entries[0].data["success"] is True

    def test_log_download_result_with_source(self, temp_log_dir):
        """Test logging download results with custom source"""
        logger = HighMatchesLogger(temp_log_dir)

        logger.log_download_result(
            videos_added=5,
            total_videos=50,
            success=True,
            iteration=3,
            source="pexels",
        )

        assert logger.entries[0].data["source"] == "pexels"

    def test_log_rematch_result(self, temp_log_dir):
        """Test logging rematch results"""
        logger = HighMatchesLogger(temp_log_dir)

        logger.log_rematch_result(
            new_coverage=0.80,
            previous_coverage=0.70,
            match_count=100,
            iteration=2,
        )

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "rematch_result"
        assert logger.entries[0].data["new_coverage"] == 0.80
        assert logger.entries[0].data["previous_coverage"] == 0.70
        assert logger.entries[0].data["improvement"] == pytest.approx(0.10)

    def test_log_iteration_end(self, temp_log_dir):
        """Test logging iteration end"""
        logger = HighMatchesLogger(temp_log_dir)

        logger.log_iteration_end(
            iteration=2,
            coverage=0.75,
            continue_iteration=True,
            reason="Improved by 5%",
        )

        assert len(logger.entries) == 1
        assert logger.entries[0].entry_type == "iteration_end"
        assert logger.entries[0].data["continue"] is True
        assert logger.entries[0].data["reason"] == "Improved by 5%"

    def test_log_final_report(self, temp_log_dir, mock_config, mock_state):
        """Test logging final report"""
        logger = HighMatchesLogger(temp_log_dir)
        logger.log_session_start(mock_config, mock_state)

        # Create mock iterative match state
        iter_state = Mock()
        iter_state.iteration_count = 3
        iter_state.coverage_history = [0.50, 0.65, 0.75, 0.82]
        iter_state.videos_added_per_iteration = [10, 8, 5]
        iter_state.final_coverage = 0.82
        iter_state.target_achieved = False

        report = CoverageReport(
            total_segments=100,
            high_confidence=82,
            medium_confidence=10,
            low_confidence=8,
            coverage_ratio=0.82,
            weak_segments=[WeakSegment("S001", 0, "test", 0.5)],
        )

        logger.log_final_report(iter_state, report)

        # Should have session_start + session_end
        assert len(logger.entries) == 2
        assert logger.entries[-1].entry_type == "session_end"
        assert "duration_seconds" in logger.entries[-1].data

    def test_final_json_written(self, temp_log_dir, mock_config, mock_state):
        """Test that final JSON is written correctly"""
        logger = HighMatchesLogger(temp_log_dir, session_id="test")
        logger.log_session_start(mock_config, mock_state)

        iter_state = Mock()
        iter_state.iteration_count = 1
        iter_state.coverage_history = [0.70, 0.80]
        iter_state.videos_added_per_iteration = [10]
        iter_state.final_coverage = 0.80
        iter_state.target_achieved = False

        report = CoverageReport(
            total_segments=50,
            high_confidence=40,
            medium_confidence=5,
            low_confidence=5,
            coverage_ratio=0.80,
            weak_segments=[],
        )

        logger.log_final_report(iter_state, report)

        # Check JSON file was created
        json_file = temp_log_dir / "high_matches_test.json"
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)

        assert data["session_id"] == "test"
        assert "config" in data
        assert "initial_state" in data
        assert "summary" in data
        assert "entries" in data

    def test_thread_safety(self, temp_log_dir):
        """Test that logging is thread-safe"""
        logger = HighMatchesLogger(temp_log_dir)
        errors = []

        def log_entries(thread_id):
            try:
                for i in range(10):
                    logger.log_iteration_start(
                        iteration=thread_id * 100 + i,
                        coverage=0.5,
                        weak_count=10,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=log_entries, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(logger.entries) == 50  # 5 threads * 10 entries each

    def test_text_log_content(self, temp_log_dir, mock_config, mock_state):
        """Test that text log contains expected content"""
        logger = HighMatchesLogger(temp_log_dir, session_id="test")
        logger.log_session_start(mock_config, mock_state)

        with open(logger.log_file) as f:
            content = f.read()

        assert "HIGH MATCHES MODE SESSION" in content
        assert "test" in content  # session_id
        assert "target_confidence" in content

    def test_multiple_coverage_analyses(self, temp_log_dir):
        """Test logging multiple coverage analyses"""
        logger = HighMatchesLogger(temp_log_dir)

        for i in range(3):
            report = CoverageReport(
                total_segments=100,
                high_confidence=50 + i * 10,
                medium_confidence=30 - i * 5,
                low_confidence=20 - i * 5,
                coverage_ratio=0.50 + i * 0.10,
            )
            logger.log_coverage_analysis(report, iteration=i)

        assert len(logger.entries) == 3
        coverages = [e.data["coverage_ratio"] for e in logger.entries]
        assert coverages == [0.50, 0.60, 0.70]

    def test_creates_log_directory_if_missing(self):
        """Test that logger creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "logs"
            # Directory doesn't exist yet

            logger = HighMatchesLogger(nested_dir)

            assert nested_dir.exists()
            assert logger.log_file.exists()

    def test_handles_unicode_content(self, temp_log_dir):
        """Test that logger handles unicode in content"""
        logger = HighMatchesLogger(temp_log_dir)

        report = CoverageReport(
            total_segments=10,
            high_confidence=5,
            medium_confidence=3,
            low_confidence=2,
            coverage_ratio=0.50,
            weak_segments=[
                WeakSegment("S001", 0, "日本語テスト content with émojis 🎬", 0.3),
            ],
        )

        logger.log_coverage_analysis(report, iteration=0)

        # Should not raise exception
        assert len(logger.entries) == 1

    def test_header_format(self, temp_log_dir):
        """Test log file header format"""
        logger = HighMatchesLogger(temp_log_dir, session_id="header_test")

        with open(logger.log_file) as f:
            content = f.read()

        # Header should contain session info
        assert "=" * 80 in content
        assert "header_test" in content
        assert "Started:" in content
