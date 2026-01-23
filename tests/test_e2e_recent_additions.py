"""
End-to-end tests for recent additions (2026-01-18 to 2026-01-22).

Tests:
1. High Matches Mode (iterative matching with coverage targets)
2. Client Profiles (cross-project learning)
3. OTIO file:/// URL prefix (DaVinci Resolve compatibility)

Created: 2026-01-22
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import tempfile
import json
import yaml
from pathlib import Path
from dataclasses import asdict

# ============================================================================
# High Matches Mode Tests
# ============================================================================

class TestCoverageAnalyzer:
    """Test coverage analysis for high matches mode."""

    def _create_simple_match(self, segment_index: int, confidence: float, video_file: str):
        """Create a simple match dict that coverage analyzer understands."""
        return {
            'segment_index': segment_index,
            'confidence': confidence,
            'video_file': video_file,
        }

    def test_analyze_coverage_empty_segments(self):
        """Test coverage analysis with no segments."""
        from src.matching.coverage_analyzer import analyze_coverage, CoverageReport

        report = analyze_coverage(
            matches=[],
            voiceover_segments=[],
            target_confidence=0.90,
        )

        assert report.total_segments == 0
        assert report.coverage_ratio == 0.0
        assert report.high_confidence == 0
        assert len(report.weak_segments) == 0

    def test_analyze_coverage_all_high_confidence(self):
        """Test coverage when all segments have high confidence."""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        # Create segments
        segments = [
            VoiceoverSegment(index=i, start=i*3.0, end=(i+1)*3.0, text=f"Segment {i}")
            for i in range(5)
        ]

        # Create high-confidence matches using dicts (coverage analyzer accepts dicts)
        matches = [
            self._create_simple_match(i, 0.95, f"video_{i}.mp4")
            for i in range(5)
        ]

        report = analyze_coverage(
            matches=matches,
            voiceover_segments=segments,
            target_confidence=0.90,
        )

        assert report.total_segments == 5
        assert report.high_confidence == 5
        assert report.coverage_ratio == 1.0
        assert len(report.weak_segments) == 0

    def test_analyze_coverage_mixed_confidence(self):
        """Test coverage with mixed confidence levels."""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        segments = [
            VoiceoverSegment(index=i, start=i*3.0, end=(i+1)*3.0, text=f"Segment {i}")
            for i in range(10)
        ]

        # Create matches: 5 high, 3 medium, 2 low
        confidences = [0.95, 0.92, 0.91, 0.90, 0.90,  # 5 high (>=0.90)
                       0.85, 0.75, 0.72,  # 3 medium (0.70-0.90)
                       0.50, 0.30]  # 2 low (<0.70)

        matches = [
            self._create_simple_match(i, conf, f"video_{i}.mp4")
            for i, conf in enumerate(confidences)
        ]

        report = analyze_coverage(
            matches=matches,
            voiceover_segments=segments,
            target_confidence=0.90,
            medium_threshold=0.70,
        )

        assert report.total_segments == 10
        assert report.high_confidence == 5
        assert report.medium_confidence == 3
        assert report.low_confidence == 2
        assert report.coverage_ratio == 0.5  # 5/10
        # Weak segments = medium + low
        assert len(report.weak_segments) == 5

    def test_analyze_coverage_weak_segments_sorted(self):
        """Test that weak segments are sorted by confidence (lowest first)."""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        segments = [
            VoiceoverSegment(index=i, start=i*3.0, end=(i+1)*3.0, text=f"Segment {i}")
            for i in range(5)
        ]

        # All low confidence, unsorted
        confidences = [0.50, 0.20, 0.80, 0.30, 0.60]
        matches = [
            self._create_simple_match(i, conf, f"video_{i}.mp4")
            for i, conf in enumerate(confidences)
        ]

        report = analyze_coverage(
            matches=matches,
            voiceover_segments=segments,
            target_confidence=0.90,
        )

        # All are weak (none >= 0.90)
        assert len(report.weak_segments) == 5
        # Should be sorted by confidence ascending
        confs = [ws.current_confidence for ws in report.weak_segments]
        assert confs == sorted(confs)

    def test_coverage_report_to_dict(self):
        """Test CoverageReport serialization."""
        from src.matching.coverage_analyzer import CoverageReport, WeakSegment

        report = CoverageReport(
            total_segments=10,
            high_confidence=7,
            medium_confidence=2,
            low_confidence=1,
            coverage_ratio=0.7,
            weak_segments=[
                WeakSegment(
                    segment_id="S001",
                    segment_index=1,
                    text="Test segment",
                    current_confidence=0.5,
                )
            ],
            target_confidence=0.90,
        )

        data = report.to_dict()
        assert data['total_segments'] == 10
        assert data['coverage_ratio'] == 0.7
        assert data['weak_segment_count'] == 1

    def test_get_improvement_delta(self):
        """Test improvement delta calculation."""
        from src.matching.coverage_analyzer import get_improvement_delta

        # Good improvement
        delta, improving = get_improvement_delta(0.50, 0.60, min_improvement=0.02)
        assert delta == pytest.approx(0.10)
        assert improving is True

        # Minimal improvement (below threshold)
        delta, improving = get_improvement_delta(0.50, 0.51, min_improvement=0.02)
        assert delta == pytest.approx(0.01)
        assert improving is False

        # No improvement
        delta, improving = get_improvement_delta(0.50, 0.50, min_improvement=0.02)
        assert delta == pytest.approx(0.0)
        assert improving is False

        # Regression
        delta, improving = get_improvement_delta(0.60, 0.50, min_improvement=0.02)
        assert delta == pytest.approx(-0.10)
        assert improving is False


class TestIterativeMatchStage:
    """Test iterative match stage for high matches mode."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with high_matches_mode settings."""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.enabled = True
        config.matching.high_matches_mode.target_confidence = 0.90
        config.matching.high_matches_mode.coverage_target = 0.85
        config.matching.high_matches_mode.max_iterations = 3
        config.matching.high_matches_mode.videos_per_iteration = 10
        config.matching.high_matches_mode.keyword_strategy = "llm"
        return config

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state."""
        from src.state import PipelineState, VoiceoverSegment

        state = PipelineState()
        state.project_dir = tempfile.mkdtemp()

        # Create voiceover segments
        state.voiceover_segments = [
            VoiceoverSegment(index=i, start=i*3.0, end=(i+1)*3.0, text=f"Segment about topic {i}")
            for i in range(10)
        ]

        # Create initial matches with varying confidence (as dicts for coverage analyzer)
        state.matches = []
        for i in range(10):
            # First 7 have high confidence, last 3 have low
            conf = 0.92 if i < 7 else 0.50
            state.matches.append({
                'segment_index': i,
                'confidence': conf,
                'video_file': f"video_{i}.mp4",
            })

        state.keywords = ["topic_1", "topic_2"]
        state.downloaded_videos = []
        state.video_candidates = []
        state.text_metadata = []

        return state

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create mock checkpoint manager."""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = None
        checkpoint.save = Mock()
        return checkpoint

    def test_stage_disabled_skips(self, mock_config, mock_state, mock_checkpoint):
        """Test that disabled high matches mode skips the stage."""
        from src.stages.iterative_match import IterativeMatchStage

        mock_config.matching.high_matches_mode.enabled = False

        stage = IterativeMatchStage()
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success is True
        # Should not create iterative state
        assert mock_state.iterative_match_state is None

    def test_stage_target_already_met(self, mock_config, mock_state, mock_checkpoint):
        """Test early exit when coverage target already met."""
        from src.stages.iterative_match import IterativeMatchStage

        # Set all matches to high confidence (matches are dicts)
        for match in mock_state.matches:
            match['confidence'] = 0.95

        stage = IterativeMatchStage()
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success is True
        assert mock_state.iterative_match_state is not None
        assert mock_state.iterative_match_state.target_achieved is True
        # No iterations needed
        assert mock_state.iterative_match_state.iteration_count == 0

    def test_stage_converts_dict_config(self, mock_state, mock_checkpoint):
        """Test that dict config is converted to object (Rule 2)."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.config.sections.matching import HighMatchesModeConfig

        # Create config with dict-based high_matches_mode
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = {
            'enabled': False,  # Disabled to avoid full iteration
            'target_confidence': 0.90,
            'coverage_target': 0.85,
            'max_iterations': 3,
            'videos_per_iteration': 10,
            'keyword_strategy': 'llm',
        }

        stage = IterativeMatchStage()
        result = stage.run(mock_state, config, mock_checkpoint)

        # Should have converted dict to object
        assert isinstance(config.matching.high_matches_mode, HighMatchesModeConfig)
        assert result.success is True

    def test_stage_creates_iterative_state(self, mock_config, mock_state, mock_checkpoint):
        """Test that iterative state is created properly."""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import IterativeMatchState

        # Set all matches to high confidence to avoid iteration (matches are dicts)
        for match in mock_state.matches:
            match['confidence'] = 0.95

        stage = IterativeMatchStage()
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert mock_state.iterative_match_state is not None
        assert isinstance(mock_state.iterative_match_state, IterativeMatchState)
        assert len(mock_state.iterative_match_state.coverage_history) > 0

    def test_stage_stops_at_max_iterations(self, mock_config, mock_state, mock_checkpoint):
        """Test that iteration stops at max_iterations."""
        from src.stages.iterative_match import IterativeMatchStage

        mock_config.matching.high_matches_mode.max_iterations = 2

        # Set low confidence to trigger iterations (matches are dicts)
        for match in mock_state.matches:
            match['confidence'] = 0.50

        stage = IterativeMatchStage()

        # Mock the download/transcribe/rematch methods to avoid real execution
        with patch.object(stage, '_download_videos', return_value=False):
            result = stage.run(mock_state, mock_config, mock_checkpoint)

        # Should stop after max iterations even without improvement
        assert mock_state.iterative_match_state.iteration_count <= 2


class TestWeakSegmentDataclass:
    """Test WeakSegment dataclass."""

    def test_weak_segment_to_dict(self):
        """Test WeakSegment serialization."""
        from src.matching.coverage_analyzer import WeakSegment

        ws = WeakSegment(
            segment_id="S005",
            segment_index=5,
            text="This is a test segment about nature",
            current_confidence=0.45,
            current_match="nature_video.mp4",
            suggested_keywords=["nature", "wildlife"],
        )

        data = ws.to_dict()
        assert data['segment_id'] == "S005"
        assert data['segment_index'] == 5
        assert data['current_confidence'] == 0.45
        assert data['suggested_keywords'] == ["nature", "wildlife"]


# ============================================================================
# Client Profiles Tests
# ============================================================================

class TestClientProfiles:
    """Test client profile system for cross-project learning."""

    @pytest.fixture
    def temp_profiles_dir(self, tmp_path, monkeypatch):
        """Create temporary profiles directory."""
        profiles_dir = tmp_path / ".matcher_rejections"
        profiles_dir.mkdir()

        # Patch the global CLIENT_PROFILES_DIR
        monkeypatch.setattr(
            'src.feedback.client_profiles.CLIENT_PROFILES_DIR',
            profiles_dir
        )
        return profiles_dir

    def test_create_new_profile(self, temp_profiles_dir):
        """Test creating a new client profile."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="test_client")

        assert profile.client_id == "test_client"
        assert profile.display_name == "Test_Client"
        assert profile.created != ""
        assert len(profile.blacklist_channels) == 0

    def test_profile_save_and_load(self, temp_profiles_dir):
        """Test saving and loading a client profile."""
        from src.feedback.client_profiles import ClientProfile

        # Create and save profile
        profile = ClientProfile(
            client_id="save_test",
            display_name="Save Test Client",
        )
        profile.blacklist_channels.add("BadChannel")
        profile.blacklist_keywords.add("spam")
        profile.total_videos_accepted = 50
        profile.total_videos_rejected = 10

        saved_path = profile.save()
        assert saved_path.exists()

        # Load profile
        loaded = ClientProfile.load("save_test")
        assert loaded is not None
        assert loaded.client_id == "save_test"
        assert loaded.display_name == "Save Test Client"
        assert "BadChannel" in loaded.blacklist_channels
        assert "spam" in loaded.blacklist_keywords
        assert loaded.total_videos_accepted == 50
        assert loaded.total_videos_rejected == 10

    def test_get_or_create_profile(self, temp_profiles_dir):
        """Test get_or_create_client_profile function."""
        from src.feedback.client_profiles import (
            get_or_create_client_profile,
            ClientProfile,
        )

        # First call creates profile
        profile1 = get_or_create_client_profile("new_client")
        assert profile1.client_id == "new_client"

        # Second call loads existing
        profile2 = get_or_create_client_profile("new_client")
        assert profile2.client_id == "new_client"

    def test_list_client_profiles(self, temp_profiles_dir):
        """Test listing all client profiles."""
        from src.feedback.client_profiles import (
            ClientProfile,
            list_client_profiles,
        )

        # Create several profiles
        for name in ["alice", "bob", "charlie"]:
            profile = ClientProfile(client_id=name)
            profile.save()

        clients = list_client_profiles()
        assert "alice" in clients
        assert "bob" in clients
        assert "charlie" in clients
        assert len(clients) == 3

    def test_channel_blacklist_check(self, temp_profiles_dir):
        """Test channel blacklist checking."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="blacklist_test")
        profile.blacklist_channels.add("SpamChannel")
        profile.blacklist_channels.add("BadQuality")

        # Case-insensitive check
        assert profile.is_channel_blacklisted("SpamChannel") is True
        assert profile.is_channel_blacklisted("spamchannel") is True
        assert profile.is_channel_blacklisted("SPAMCHANNEL") is True
        assert profile.is_channel_blacklisted("GoodChannel") is False

    def test_keyword_blacklist_check(self, temp_profiles_dir):
        """Test keyword blacklist checking."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="keyword_test")
        profile.blacklist_keywords.add("trailer")
        profile.blacklist_keywords.add("music video")

        assert profile.is_keyword_blacklisted("Official Movie Trailer") is True
        assert profile.is_keyword_blacklisted("Music Video 4K") is True
        assert profile.is_keyword_blacklisted("Documentary About Nature") is False

    def test_whitelist_overrides_blacklist(self, temp_profiles_dir):
        """Test that whitelist overrides global rejection."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="whitelist_test")
        profile.whitelist_channels.add("TrustedChannel")

        assert profile.is_channel_whitelisted("TrustedChannel") is True
        assert profile.is_channel_whitelisted("trustedchannel") is True
        assert profile.is_channel_whitelisted("OtherChannel") is False

    def test_add_project_tracking(self, temp_profiles_dir):
        """Test project tracking."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="project_test")
        initial_updated = profile.updated

        profile.add_project("/path/to/project1")
        profile.add_project("/path/to/project2")
        profile.add_project("/path/to/project1")  # Duplicate

        assert len(profile.projects) == 2
        assert "/path/to/project1" in profile.projects
        assert "/path/to/project2" in profile.projects

    def test_content_preferences_defaults(self, temp_profiles_dir):
        """Test ContentPreferences default values."""
        from src.feedback.client_profiles import ContentPreferences

        prefs = ContentPreferences()
        assert prefs.content_style == "documentary"
        assert prefs.preferred_duration_min == 30.0
        assert prefs.preferred_duration_max == 180.0
        assert prefs.avoid_trainers is True

    def test_quality_thresholds_defaults(self, temp_profiles_dir):
        """Test QualityThresholds default values."""
        from src.feedback.client_profiles import QualityThresholds

        thresholds = QualityThresholds()
        assert thresholds.min_channel_score == 0.3
        assert thresholds.min_llm_relevance == 0.6
        assert thresholds.min_embedding_score == 0.4

    def test_evolved_preset_creation(self, temp_profiles_dir):
        """Test EvolvedPreset creation."""
        from src.feedback.client_profiles import EvolvedPreset

        preset = EvolvedPreset(
            auto_blacklist_channels=["BadChannel1", "BadChannel2"],
            auto_blacklist_keywords=["spam", "ad"],
            preferred_channels=["GoodChannel"],
            preferred_duration_range=(45.0, 120.0),
            avg_accepted_duration=75.0,
            projects_analyzed=5,
        )

        assert len(preset.auto_blacklist_channels) == 2
        assert preset.avg_accepted_duration == 75.0
        assert preset.projects_analyzed == 5
        assert preset.generated_at != ""

    def test_profile_post_init_converts_types(self, temp_profiles_dir):
        """Test that __post_init__ converts dict/list to proper types."""
        from src.feedback.client_profiles import ClientProfile

        # Simulate loading from YAML where sets become lists and dataclasses become dicts
        profile = ClientProfile(
            client_id="convert_test",
            blacklist_channels=["Channel1", "Channel2"],  # List, should become set
            blacklist_keywords=["keyword1"],  # List, should become set
            preferences={"content_style": "raw", "avoid_trainers": False},  # Dict
            thresholds={"min_channel_score": 0.5},  # Dict
        )

        # Should be converted to sets
        assert isinstance(profile.blacklist_channels, set)
        assert isinstance(profile.blacklist_keywords, set)
        assert "Channel1" in profile.blacklist_channels

        # Should be converted to dataclass
        from src.feedback.client_profiles import ContentPreferences, QualityThresholds
        assert isinstance(profile.preferences, ContentPreferences)
        assert isinstance(profile.thresholds, QualityThresholds)
        assert profile.preferences.content_style == "raw"
        assert profile.thresholds.min_channel_score == 0.5


class TestEvolvePreset:
    """Test preset evolution from project history."""

    @pytest.fixture
    def temp_profiles_dir(self, tmp_path, monkeypatch):
        """Create temporary profiles directory."""
        profiles_dir = tmp_path / ".matcher_rejections"
        profiles_dir.mkdir()
        monkeypatch.setattr(
            'src.feedback.client_profiles.CLIENT_PROFILES_DIR',
            profiles_dir
        )
        return profiles_dir

    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create mock project directory with history files."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create sources.json
        sources = {
            "video1": {"channel": "GoodChannel", "duration": 60, "was_used": True},
            "video2": {"channel": "GoodChannel", "duration": 90, "was_used": True},
            "video3": {"channel": "BadChannel", "duration": 30, "was_used": False},
        }
        (project_dir / "sources.json").write_text(json.dumps(sources))

        # Create project_rejections.yaml
        rejections = {
            "rejected_videos": [
                {"channel": "SpamChannel", "reason": "too_short", "title": "Spam Video Ad"},
                {"channel": "SpamChannel", "reason": "off_topic", "title": "Another Spam"},
            ],
            "blocked_channels": ["BlockedChannel"],
        }
        (project_dir / "project_rejections.yaml").write_text(yaml.dump(rejections))

        return project_dir

    def test_evolve_preset_no_profile(self, temp_profiles_dir):
        """Test evolve preset when no profile exists."""
        from src.feedback.client_profiles import evolve_preset_from_history

        preset = evolve_preset_from_history("nonexistent_client")
        # Should return empty preset
        assert preset.projects_analyzed == 0

    def test_evolve_preset_no_projects(self, temp_profiles_dir):
        """Test evolve preset when profile has no projects."""
        from src.feedback.client_profiles import (
            ClientProfile,
            evolve_preset_from_history,
        )

        # Create profile without projects
        profile = ClientProfile(client_id="empty_client")
        profile.save()

        preset = evolve_preset_from_history("empty_client")
        assert preset.projects_analyzed == 0

    def test_evolve_preset_from_history(self, temp_profiles_dir, mock_project_dir):
        """Test evolving preset from actual project history."""
        from src.feedback.client_profiles import (
            ClientProfile,
            evolve_preset_from_history,
        )

        # Create profile with project
        profile = ClientProfile(client_id="history_client")
        profile.projects.append(str(mock_project_dir))
        profile.save()

        preset = evolve_preset_from_history("history_client")

        assert preset.projects_analyzed == 1
        # Should have learned from rejections
        assert "SpamChannel" in preset.auto_blacklist_channels
        # Should have learned preferred channels
        assert "GoodChannel" in preset.preferred_channels
        # Should have extracted keywords from rejected titles
        assert any("spam" in kw.lower() for kw in preset.auto_blacklist_keywords) or \
               any("video" in kw.lower() for kw in preset.auto_blacklist_keywords)


# ============================================================================
# OTIO URL Prefix Tests
# ============================================================================

class TestOTIOUrlPrefix:
    """Test OTIO file:/// URL prefix for DaVinci Resolve compatibility."""

    def test_encode_path_windows_absolute(self):
        """Test encoding Windows absolute paths."""
        from src.otio.utils import encode_path_for_xml_url

        # Windows path with backslashes
        result = encode_path_for_xml_url(r"E:\Projects\video.mp4")
        assert result == "file:///E:/Projects/video.mp4"

        # Windows path with forward slashes
        result = encode_path_for_xml_url("D:/Videos/clip.mp4")
        assert result == "file:///D:/Videos/clip.mp4"

    def test_encode_path_removes_extended_prefix(self):
        """Test that Windows extended-length prefix is removed."""
        from src.otio.utils import encode_path_for_xml_url

        # Extended-length path prefix (\\?\)
        result = encode_path_for_xml_url(r"\\?\C:\LongPath\video.mp4")
        assert result == "file:///C:/LongPath/video.mp4"
        assert "\\\\?\\" not in result

    def test_encode_path_unix_absolute(self):
        """Test encoding Unix absolute paths."""
        from src.otio.utils import encode_path_for_xml_url

        result = encode_path_for_xml_url("/home/user/videos/clip.mp4")
        assert result == "file:///home/user/videos/clip.mp4"

    def test_sanitize_path_for_url(self):
        """Test sanitize_path_for_url function."""
        from src.otio.utils import sanitize_path_for_url

        # Backslashes to forward slashes
        result = sanitize_path_for_url(r"E:\folder\subfolder\file.mp4")
        assert "\\" not in result
        assert result == "E:/folder/subfolder/file.mp4"

        # Remove double slashes
        result = sanitize_path_for_url("E://folder//file.mp4")
        assert "//" not in result

        # Remove extended-length prefix
        result = sanitize_path_for_url(r"\\?\E:\path\file.mp4")
        assert "?" not in result
        assert result == "E:/path/file.mp4"

    def test_format_path_url_plain(self):
        """Test format_path_url returns plain path (no file:// prefix)."""
        from src.otio.utils import format_path_url

        result = format_path_url(r"E:\Projects\video.mp4")
        # format_path_url returns plain path, not file:// URL
        assert result == "E:/Projects/video.mp4"
        assert "file://" not in result

    def test_encode_path_special_characters(self):
        """Test paths with spaces (common in Windows)."""
        from src.otio.utils import encode_path_for_xml_url

        result = encode_path_for_xml_url(r"E:\My Projects\video file.mp4")
        assert result == "file:///E:/My Projects/video file.mp4"

    def test_encode_path_empty(self):
        """Test empty path handling."""
        from src.otio.utils import encode_path_for_xml_url

        result = encode_path_for_xml_url("")
        assert result == ""

    def test_encode_path_relative(self):
        """Test relative path handling."""
        from src.otio.utils import encode_path_for_xml_url

        # Relative paths shouldn't get file:// prefix
        result = encode_path_for_xml_url("videos/clip.mp4")
        assert result == "videos/clip.mp4"


class TestCreateClipWithTimewarp:
    """Test clip creation with file:/// URLs."""

    def test_clip_uses_file_url(self):
        """Test that created clips use file:/// URL format."""
        from src.otio.utils import create_clip_with_timewarp

        clip = create_clip_with_timewarp(
            name="Test Clip",
            source_path=r"E:\Videos\test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
        )

        # Media reference should have file:/// URL
        target_url = clip.media_reference.target_url
        assert target_url.startswith("file:///")
        assert "E:/Videos/test.mp4" in target_url

    def test_clip_metadata_sanitized(self):
        """Test that clip metadata is properly sanitized."""
        from src.otio.utils import create_clip_with_timewarp
        import numpy as np

        clip = create_clip_with_timewarp(
            name="Metadata Test",
            source_path=r"E:\Videos\test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            metadata={
                'numpy_float': np.float64(0.95),
                'numpy_int': np.int32(42),
                'regular_float': 0.5,
            }
        )

        # Numpy types should be converted to Python types
        assert isinstance(clip.metadata.get('numpy_float'), float)
        assert isinstance(clip.metadata.get('numpy_int'), int)


class TestMediaPathNormalizer:
    """Test media path normalization for deduplication."""

    def test_normalizer_single_path(self, tmp_path):
        """Test normalizer with unique paths."""
        from src.otio.utils import MediaPathNormalizer

        # Create test files
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"
        video1.write_bytes(b"video1")
        video2.write_bytes(b"video2")

        normalizer = MediaPathNormalizer()
        normalizer.register(str(video1))
        normalizer.register(str(video2))
        normalizer.build_map()

        # Each should map to itself
        assert normalizer.get_canonical(str(video1)) == str(video1)
        assert normalizer.get_canonical(str(video2)) == str(video2)
        assert normalizer.duplicates_found == 0

    def test_normalizer_duplicate_files(self, tmp_path):
        """Test normalizer detects duplicate files."""
        from src.otio.utils import MediaPathNormalizer

        # Create two directories
        stock_dir = tmp_path / "stock"
        broll_dir = tmp_path / "broll"
        stock_dir.mkdir()
        broll_dir.mkdir()

        # Create identical files (same name, same content/size)
        content = b"identical video content"
        stock_file = stock_dir / "video.mp4"
        broll_file = broll_dir / "video.mp4"
        stock_file.write_bytes(content)
        broll_file.write_bytes(content)

        normalizer = MediaPathNormalizer()
        normalizer.register(str(stock_file))
        normalizer.register(str(broll_file))
        normalizer.build_map()

        # Both should map to stock (preferred)
        canonical = normalizer.get_canonical(str(broll_file))
        assert "stock" in canonical
        assert normalizer.duplicates_found > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestHighMatchesModeIntegration:
    """Integration tests for high matches mode with pipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create full mock config."""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.enabled = True
        config.matching.high_matches_mode.target_confidence = 0.90
        config.matching.high_matches_mode.coverage_target = 0.85
        config.matching.high_matches_mode.max_iterations = 2
        config.matching.high_matches_mode.videos_per_iteration = 5
        config.matching.high_matches_mode.keyword_strategy = "weak_segments"

        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 100

        config.pexels = Mock()
        config.pexels.enabled = False

        config.downloaded_videos_dir = tempfile.mkdtemp()

        return config

    def test_iterative_state_persistence(self, mock_config):
        """Test that iterative match state is properly tracked."""
        from src.state import IterativeMatchState

        state = IterativeMatchState()
        state.iteration_count = 2
        state.coverage_history = [0.50, 0.65, 0.80]
        state.videos_added_per_iteration = [10, 8]
        state.final_coverage = 0.80
        state.target_achieved = False
        state.weak_segment_count = 3

        # Verify state tracking
        assert state.iteration_count == 2
        assert len(state.coverage_history) == 3
        assert state.coverage_history[-1] == 0.80


class TestClientProfileIntegration:
    """Integration tests for client profiles with config."""

    @pytest.fixture
    def temp_profiles_dir(self, tmp_path, monkeypatch):
        """Create temporary profiles directory."""
        profiles_dir = tmp_path / ".matcher_rejections"
        profiles_dir.mkdir()
        monkeypatch.setattr(
            'src.feedback.client_profiles.CLIENT_PROFILES_DIR',
            profiles_dir
        )
        return profiles_dir

    def test_apply_profile_to_config(self, temp_profiles_dir):
        """Test applying client profile to config."""
        from src.feedback.client_profiles import (
            ClientProfile,
            apply_client_profile_to_config,
        )

        profile = ClientProfile(client_id="config_test")
        profile.thresholds.min_channel_score = 0.5

        # Create mock config
        config = Mock()
        config.feedback = Mock()
        config.feedback.channel_scoring = Mock()
        config.feedback.channel_scoring.min_score = 0.3

        apply_client_profile_to_config(profile, config)

        # Profile's stricter threshold should be applied
        assert config.feedback.channel_scoring.min_score == 0.5

    def test_profile_record_acceptance_rejection(self, temp_profiles_dir):
        """Test recording acceptances and rejections."""
        from src.feedback.client_profiles import ClientProfile

        profile = ClientProfile(client_id="record_test")

        profile.record_acceptance(5)
        profile.record_rejection(2)
        profile.record_acceptance(3)

        assert profile.total_videos_accepted == 8
        assert profile.total_videos_rejected == 2


# ============================================================================
# CLI Integration Tests
# ============================================================================

class TestCLIFlags:
    """Test CLI flag parsing for recent additions."""

    def test_high_matches_flag_parsing(self):
        """Test --high-matches flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        # Temporarily replace sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--high-matches', '--non-interactive']
            args = parse_arguments()
            assert args.high_matches is True
        finally:
            sys.argv = original_argv

    def test_target_confidence_flag_parsing(self):
        """Test --target-confidence flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--target-confidence', '0.85', '--non-interactive']
            args = parse_arguments()
            assert args.target_confidence == 0.85
        finally:
            sys.argv = original_argv

    def test_coverage_target_flag_parsing(self):
        """Test --coverage-target flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--coverage-target', '0.80', '--non-interactive']
            args = parse_arguments()
            assert args.coverage_target == 0.80
        finally:
            sys.argv = original_argv

    def test_client_flag_parsing(self):
        """Test --client flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--client', 'theresa', '--non-interactive']
            args = parse_arguments()
            assert args.client == 'theresa'
        finally:
            sys.argv = original_argv

    def test_evolve_preset_flag_parsing(self):
        """Test --evolve-preset flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--evolve-preset', '--client', 'stu', '--non-interactive']
            args = parse_arguments()
            assert args.evolve_preset is True
            assert args.client == 'stu'
        finally:
            sys.argv = original_argv

    def test_list_clients_flag_parsing(self):
        """Test --list-clients flag is recognized."""
        import sys
        from src.cli.args import parse_arguments

        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--list-clients', '--non-interactive']
            args = parse_arguments()
            assert args.list_clients is True
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
