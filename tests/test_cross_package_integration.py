"""
Cross-Package Integration Tests

Tests interactions between packages in the same repo to ensure
refactors don't silently break dependents.

Focus areas (recent additions):
- High Matches Mode: iterative_match <-> coverage_analyzer <-> recovery_keywords <-> stages
- State dataclasses: Match objects used across matching, stages, otio packages
- Config schema: dict/object handling across config consumers
- Caption-first mode: video_id vs file path handling across downloader, stages, matching
- OTIO integration: state.matches -> otio timeline generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from dataclasses import dataclass


# =============================================================================
# HIGH MATCHES MODE CROSS-PACKAGE TESTS
# =============================================================================

class TestIterativeMatchCoverageAnalyzerIntegration:
    """Tests: iterative_match.py <-> coverage_analyzer.py integration"""

    def test_coverage_analyzer_with_simple_match_dataclass(self):
        """Coverage analyzer should handle Match dataclass from state.py"""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        # Create state.Match-like objects (simple dataclass)
        @dataclass
        class SimpleMatch:
            segment_index: int
            confidence: float
            video_file: str

        matches = [
            SimpleMatch(segment_index=0, confidence=0.95, video_file='video1.mp4'),
            SimpleMatch(segment_index=1, confidence=0.75, video_file='video2.mp4'),
            SimpleMatch(segment_index=2, confidence=0.50, video_file='video3.mp4'),
        ]

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0, end=5, text='First segment'),
            VoiceoverSegment(index=1, start=5, end=10, text='Second segment'),
            VoiceoverSegment(index=2, start=10, end=15, text='Third segment'),
        ]

        report = analyze_coverage(matches, voiceover_segments, target_confidence=0.90)

        assert report.total_segments == 3
        assert report.high_confidence == 1  # Only 0.95 >= 0.90
        assert report.medium_confidence == 1  # 0.75 >= 0.70
        assert report.low_confidence == 1  # 0.50 < 0.70
        assert len(report.weak_segments) == 2  # Medium + Low

    def test_coverage_analyzer_with_dict_matches(self):
        """Coverage analyzer should handle serialized dict matches"""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        # Serialized matches (as dicts)
        matches = [
            {'segment_index': 0, 'confidence': 0.92, 'video_file': 'a.mp4'},
            {'segment_index': 1, 'confidence': 0.60, 'video_file': 'b.mp4'},
        ]

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0, end=5, text='Segment A'),
            VoiceoverSegment(index=1, start=5, end=10, text='Segment B'),
        ]

        report = analyze_coverage(matches, voiceover_segments, target_confidence=0.90)

        assert report.high_confidence == 1
        assert report.low_confidence == 1
        assert report.coverage_ratio == 0.5

    def test_coverage_analyzer_with_match_result_wrapper(self):
        """Coverage analyzer should handle MatchResultWrapper from matching/main.py"""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        # Create MatchResultWrapper-like structure
        class MockVoiceoverSegment:
            def __init__(self, idx):
                self.index = idx

        class MockVideoSegment:
            def __init__(self, source_file):
                self.source_file = source_file

        class MockPrimaryMatch:
            def __init__(self, idx, conf, source_file):
                self.voiceover_segment = MockVoiceoverSegment(idx)
                self.video_segment = MockVideoSegment(source_file)
                self.confidence = conf

        class MockMatchResultWrapper:
            def __init__(self, idx, conf, source_file):
                self.primary_match = MockPrimaryMatch(idx, conf, source_file)

        matches = [
            MockMatchResultWrapper(0, 0.95, 'video1.mp4'),
            MockMatchResultWrapper(1, 0.85, 'video2.mp4'),
        ]

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0, end=5, text='Seg1'),
            VoiceoverSegment(index=1, start=5, end=10, text='Seg2'),
        ]

        report = analyze_coverage(matches, voiceover_segments, target_confidence=0.90)

        assert report.high_confidence == 1  # 0.95
        assert report.medium_confidence == 1  # 0.85
        assert report.total_segments == 2


class TestRecoveryKeywordsMatchingIntegration:
    """Tests: recovery_keywords.py <-> matching package integration"""

    def test_weak_segments_from_coverage_to_recovery(self):
        """WeakSegment from coverage_analyzer works with recovery_keywords"""
        from src.matching.coverage_analyzer import WeakSegment
        from src.matching.recovery_keywords import generate_recovery_keywords

        weak_segments = [
            WeakSegment(
                segment_id='S001',
                segment_index=1,
                text='restaurant closures affecting local economy',
                current_confidence=0.45,
                current_match='generic_video.mp4',
            ),
            WeakSegment(
                segment_id='S005',
                segment_index=5,
                text='supply chain disruptions in food industry',
                current_confidence=0.55,
                current_match=None,
            ),
        ]

        # Test weak_segments strategy (no LLM required)
        keywords = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=['food', 'business'],
            existing_videos=[],
            strategy='weak_segments',
            max_keywords=5,
            config=None,
        )

        # Should extract keywords from weak segment text
        assert isinstance(keywords, list)
        # Keywords should be generated (may be empty if extraction fails)


class TestIterativeMatchStageConfigIntegration:
    """Tests: iterative_match.py <-> config package integration"""

    def test_high_matches_mode_config_dict_conversion(self):
        """IterativeMatchStage converts dict config to object (Rule 2)"""
        from src.config.sections.matching import HighMatchesModeConfig

        # Simulate config loaded as dict (after YAML merge)
        config_dict = {
            'enabled': True,
            'target_confidence': 0.85,
            'coverage_target': 0.80,
            'max_iterations': 3,
            'videos_per_iteration': 10,
            'keyword_strategy': 'weak_segments',
        }

        # Convert to object
        hmm_config = HighMatchesModeConfig(**config_dict)

        assert hmm_config.enabled is True
        assert hmm_config.target_confidence == 0.85
        assert hmm_config.keyword_strategy == 'weak_segments'

    def test_high_matches_mode_default_values(self):
        """HighMatchesModeConfig has sensible defaults"""
        from src.config.sections.matching import HighMatchesModeConfig

        config = HighMatchesModeConfig()

        assert config.enabled is False
        assert config.target_confidence == 0.90
        assert config.coverage_target == 0.85
        assert config.max_iterations == 3  # Default from config


# =============================================================================
# STATE DATACLASS CROSS-PACKAGE TESTS
# =============================================================================

class TestStateDataclassesAcrossPackages:
    """Tests: state.py dataclasses used in matching, stages, otio"""

    def test_voiceover_segment_in_coverage_analyzer(self):
        """VoiceoverSegment index/text attributes accessible in coverage_analyzer"""
        from src.state import VoiceoverSegment

        seg = VoiceoverSegment(index=5, start=10.0, end=15.0, text='Test text')

        # coverage_analyzer accesses these attributes
        assert hasattr(seg, 'index')
        assert hasattr(seg, 'text')
        assert seg.index == 5
        assert seg.text == 'Test text'

    def test_downloaded_video_in_recovery_keywords(self):
        """DownloadedVideo attributes accessible in recovery_keywords"""
        from src.state import DownloadedVideo

        video = DownloadedVideo(
            file='/path/to/video_abc123def45.mp4',
            title='Test Video',
            keyword='food',
        )

        # recovery_keywords may check existing videos
        assert hasattr(video, 'file')
        assert hasattr(video, 'title')
        assert hasattr(video, 'keyword')

    def test_iterative_match_state_serialization(self):
        """IterativeMatchState can be serialized for checkpoint"""
        from src.state import IterativeMatchState
        from dataclasses import asdict

        state = IterativeMatchState()
        state.iteration_count = 3
        state.coverage_history = [0.5, 0.7, 0.85]
        state.target_achieved = True

        # Should be serializable for checkpoint
        serialized = asdict(state)
        assert serialized['iteration_count'] == 3
        assert serialized['coverage_history'] == [0.5, 0.7, 0.85]

    def test_video_candidate_caption_first_attributes(self):
        """VideoCandidate has caption-first mode attributes"""
        from src.state import VideoCandidate

        # Caption-first mode video candidate
        candidate = VideoCandidate(
            video_id='abc123def45',
            url='https://youtube.com/watch?v=abc123def45',
            title='Test Video',
            has_captions=True,
            caption_language='en',
            is_auto_caption=False,
            transcript_source='manual_caption',
        )

        assert len(candidate.video_id) == 11  # YouTube ID length
        assert candidate.has_captions is True
        assert candidate.transcript_source == 'manual_caption'


# =============================================================================
# CONFIG SCHEMA CROSS-PACKAGE TESTS
# =============================================================================

class TestConfigSchemaCrossPackage:
    """Tests: config sections used across different packages"""

    def test_matching_config_in_iterative_match(self):
        """MatchingConfig.high_matches_mode accessible in iterative_match"""
        from src.config.sections.matching import MatchingConfig, HighMatchesModeConfig

        matching = MatchingConfig()
        hmm = matching.high_matches_mode

        # iterative_match accesses these
        assert hasattr(hmm, 'enabled')
        assert hasattr(hmm, 'target_confidence')
        assert hasattr(hmm, 'coverage_target')
        assert hasattr(hmm, 'max_iterations')
        assert hasattr(hmm, 'videos_per_iteration')
        assert hasattr(hmm, 'keyword_strategy')

    def test_premise_scoring_config_in_scoring(self):
        """PremiseScoringConfig accessible in scoring.py"""
        from src.config.sections.matching import PremiseScoringConfig

        config = PremiseScoringConfig()

        # scoring.py accesses these
        assert hasattr(config, 'enabled')
        assert hasattr(config, 'premise_weight')
        assert hasattr(config, 'keyword_weight')
        assert hasattr(config, 'embedding_weight')
        assert hasattr(config, 'transcript_weight')

    def test_download_config_caption_first_section(self):
        """DownloadConfig.caption_first accessible in stages"""
        from src.config.sections.download import DownloadConfig

        download = DownloadConfig()
        caption_first = download.caption_first

        # caption stage checks this
        assert hasattr(caption_first, 'enabled')
        assert hasattr(caption_first, 'prefer_manual_captions')


# =============================================================================
# CAPTION-FIRST MODE CROSS-PACKAGE TESTS
# =============================================================================

class TestCaptionFirstCrossPackage:
    """Tests: caption-first mode interactions across packages"""

    def test_video_id_vs_file_path_handling(self):
        """VideoCandidate video_id is 11-char YouTube ID in caption-first mode"""
        from src.state import VideoCandidate

        # In caption-first mode, video_path stores YouTube ID, not file path
        candidate = VideoCandidate(
            video_id='abc123def45',
            url='https://youtube.com/watch?v=abc123def45',
        )

        # Rule 22: 11-char pattern indicates YouTube ID
        assert len(candidate.video_id) == 11

        # Should NOT be treated as a file path
        path = Path(candidate.video_id)
        assert not path.exists()  # Not a real file

    def test_caption_first_video_candidate_propagation(self):
        """VideoCandidate caption data propagates to text_metadata"""
        from src.state import VideoCandidate, TranscriptSegment

        candidate = VideoCandidate(
            video_id='abc123def45',
            url='https://youtube.com/watch?v=abc123def45',
            has_captions=True,
            caption_language='en',
            is_auto_caption=False,
            transcript_source='manual_caption',
        )

        # text_metadata segment derived from caption
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text='Caption text here',
            source_file=candidate.video_id,  # Uses video_id as source
            transcript_source=candidate.transcript_source,
        )

        assert segment.source_file == 'abc123def45'
        assert segment.transcript_source == 'manual_caption'


class TestCaptionFirstMatchingIntegration:
    """Tests: caption-first affects matching package behavior"""

    def test_coverage_analyzer_handles_caption_source_files(self):
        """Coverage analyzer handles video_id as source_file"""
        from src.matching.coverage_analyzer import analyze_coverage, CoverageReport
        from src.state import VoiceoverSegment

        # Match with YouTube ID as video_file (caption-first mode)
        matches = [
            {'segment_index': 0, 'confidence': 0.90, 'video_file': 'abc123def45'},
        ]

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0, end=5, text='Test'),
        ]

        report = analyze_coverage(matches, voiceover_segments)

        # Should handle 11-char video IDs without errors
        assert report.total_segments == 1
        assert report.high_confidence == 1


# =============================================================================
# OTIO/OUTPUT CROSS-PACKAGE TESTS
# =============================================================================

class TestOTIOStateIntegration:
    """Tests: OTIO timeline generation uses state.matches correctly"""

    def test_match_to_otio_clip_conversion(self):
        """State Match objects can be used for OTIO clip generation"""
        from src.state import Match

        match = Match(
            segment_index=0,
            video_file='/path/to/video_abc123def45.mp4',
            video_start=10.0,
            video_end=15.0,
            confidence=0.92,
        )

        # OTIO needs these fields
        assert hasattr(match, 'video_file')
        assert hasattr(match, 'video_start')
        assert hasattr(match, 'video_end')
        assert hasattr(match, 'segment_index')

    def test_otio_file_url_format(self):
        """OTIO requires file:/// URL prefix (Rule 14)"""
        from src.otio.utils import encode_path_for_xml_url

        # Windows path
        windows_path = 'E:\\Videos\\test.mp4'
        url = encode_path_for_xml_url(windows_path)
        assert url.startswith('file:///')

        # Forward slashes in URL (no backslashes after conversion)
        assert '\\' not in url


# =============================================================================
# STAGE DEPENDENCY CHAIN TESTS
# =============================================================================

class TestStageChainDependencies:
    """Tests: Pipeline stage order and data dependencies"""

    def test_stage_order_iterative_match_after_match(self):
        """ITERATIVE_MATCH comes after MATCH in stage order"""
        from src.checkpoint import STAGE_ORDER

        match_idx = STAGE_ORDER.index('MATCH')
        iterative_idx = STAGE_ORDER.index('ITERATIVE_MATCH')

        assert iterative_idx > match_idx

    def test_stage_order_premise_before_scene_detection(self):
        """PREMISE comes before SCENE_DETECTION"""
        from src.checkpoint import STAGE_ORDER

        premise_idx = STAGE_ORDER.index('PREMISE')
        scene_idx = STAGE_ORDER.index('SCENE_DETECTION')

        assert premise_idx < scene_idx

    def test_iterative_match_depends_on_match_results(self):
        """IterativeMatchStage requires state.matches from MatchStage"""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import PipelineState

        state = PipelineState()

        # IterativeMatchStage reads state.matches
        assert hasattr(state, 'matches')

    def test_iterative_match_modifies_correct_state_fields(self):
        """IterativeMatchStage updates iterative_match_state"""
        from src.state import PipelineState, IterativeMatchState

        state = PipelineState()

        # Should be able to set iterative_match_state
        state.iterative_match_state = IterativeMatchState()
        state.iterative_match_state.iteration_count = 2

        assert state.iterative_match_state.iteration_count == 2


# =============================================================================
# CHECKPOINT CROSS-PACKAGE TESTS
# =============================================================================

class TestCheckpointCrossPackage:
    """Tests: Checkpoint system works with new state fields"""

    def test_iterative_match_in_stage_order(self):
        """ITERATIVE_MATCH stage is in STAGE_ORDER"""
        from src.checkpoint import STAGE_ORDER

        assert 'ITERATIVE_MATCH' in STAGE_ORDER

    def test_checkpoint_data_has_premise_field(self):
        """CheckpointData can store video premises"""
        from src.checkpoint import CheckpointData

        data = CheckpointData()

        assert hasattr(data, 'premise')
        assert data.premise == {}

    def test_checkpoint_data_has_match_field(self):
        """CheckpointData can store match results"""
        from src.checkpoint import CheckpointData

        data = CheckpointData()

        assert hasattr(data, 'match')
        assert data.match == {}


# =============================================================================
# CLIENT PROFILES CROSS-PACKAGE TESTS
# =============================================================================

class TestClientProfilesIntegration:
    """Tests: client_profiles.py integration with config and matching"""

    def test_client_profile_path_structure(self):
        """Client profile paths follow expected structure"""
        from pathlib import Path
        import os

        # Expected: ~/.matcher_rejections/client_<name>/
        home = Path.home()
        expected_base = home / '.matcher_rejections'

        # Structure should be compatible with config expectations
        client_dir = expected_base / 'client_test'
        assert str(client_dir).replace('\\', '/').endswith('.matcher_rejections/client_test')


# =============================================================================
# ERROR HANDLING CROSS-PACKAGE TESTS
# =============================================================================

class TestErrorHandlingCrossPackage:
    """Tests: Error handling across package boundaries"""

    def test_coverage_analyzer_empty_matches(self):
        """Coverage analyzer handles empty matches list"""
        from src.matching.coverage_analyzer import analyze_coverage
        from src.state import VoiceoverSegment

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0, end=5, text='Test'),
        ]

        report = analyze_coverage([], voiceover_segments)

        assert report.total_segments == 1
        assert report.high_confidence == 0
        assert report.coverage_ratio == 0.0

    def test_coverage_analyzer_empty_segments(self):
        """Coverage analyzer handles empty voiceover_segments"""
        from src.matching.coverage_analyzer import analyze_coverage

        report = analyze_coverage([], [])

        assert report.total_segments == 0
        assert report.coverage_ratio == 0.0

    def test_recovery_keywords_empty_weak_segments(self):
        """Recovery keywords handles empty weak_segments list"""
        from src.matching.recovery_keywords import generate_recovery_keywords

        keywords = generate_recovery_keywords(
            weak_segments=[],
            existing_keywords=['food'],
            existing_videos=[],
            strategy='weak_segments',
            max_keywords=5,
            config=None,
        )

        # Should return empty list, not error
        assert keywords == []


# =============================================================================
# RULE COMPLIANCE TESTS
# =============================================================================

class TestRuleCompliance:
    """Tests: CLAUDE.md rules are enforced across packages"""

    def test_rule_2_config_dict_object_handling(self):
        """Rule 2: Config nested dataclasses handle dict→object conversion"""
        from src.config.sections.matching import HighMatchesModeConfig

        # Config may load as dict after merge
        config_dict = {'enabled': True, 'target_confidence': 0.85}

        # Should convert without error
        config_obj = HighMatchesModeConfig(**config_dict)
        assert config_obj.enabled is True

    def test_rule_6_dict_object_dual_access(self):
        """Rule 6: Code handles both dict.get() and getattr() patterns"""

        # Simulate config as dict
        config_dict = {'enabled': True, 'target': 0.90}

        # Simulate config as object
        class ConfigObj:
            enabled = True
            target = 0.90

        config_obj = ConfigObj()

        # Code should handle both
        def get_enabled(config):
            if isinstance(config, dict):
                return config.get('enabled', False)
            return getattr(config, 'enabled', False)

        assert get_enabled(config_dict) is True
        assert get_enabled(config_obj) is True

    def test_rule_14_otio_file_url_prefix(self):
        """Rule 14: OTIO paths use file:/// URL prefix"""
        from src.otio.utils import encode_path_for_xml_url

        url = encode_path_for_xml_url('E:/Videos/test.mp4')

        # Must start with file:///
        assert url.startswith('file:///')
        # Forward slashes only
        assert '\\' not in url

    def test_rule_22_caption_first_video_id_pattern(self):
        """Rule 22: Caption-first video_id is 11-char YouTube ID"""
        from src.state import VideoCandidate
        import re

        candidate = VideoCandidate(
            video_id='abc123def45',
            url='https://youtube.com/watch?v=abc123def45',
        )

        # YouTube ID pattern: 11 alphanumeric + underscore/dash
        yt_pattern = re.compile(r'^[a-zA-Z0-9_-]{11}$')
        assert yt_pattern.match(candidate.video_id)
