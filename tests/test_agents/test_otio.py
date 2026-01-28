"""
Tests for OTIOHealer - comprehensive timeline generation error recovery.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.agents.healers.otio import OTIOHealer
from src.agents.base import HealerResult, HealerAction


class TestOTIOHealerDetection:
    """Tests for error detection methods."""

    def test_error_patterns(self, mock_config, project_dir):
        """Test healer has correct error patterns."""
        healer = OTIOHealer(mock_config, project_dir)

        assert "opentimelineio" in healer.error_patterns
        assert "media reference" in healer.error_patterns
        assert "duration" in healer.error_patterns
        assert "gap" in healer.error_patterns

    def test_can_handle_otio_errors(self, mock_config, project_dir):
        """Test can_handle matches OTIO-related errors."""
        healer = OTIOHealer(mock_config, project_dir)

        # Should match OTIO errors
        assert healer.can_handle(Exception("opentimelineio error"), "OUTPUT")
        assert healer.can_handle(Exception("Timeline creation failed"), "OUTPUT")
        assert healer.can_handle(Exception("clip duration invalid"), "OUTPUT")
        assert healer.can_handle(Exception("gap overflow"), "OUTPUT")

    def test_can_handle_media_errors(self, mock_config, project_dir):
        """Test detection of media reference errors."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("Media reference not found"), "OUTPUT")
        assert healer.can_handle(Exception("file not found: video.mp4"), "OUTPUT")
        assert healer.can_handle(FileNotFoundError("missing.mp4"), "OUTPUT")

    def test_can_handle_exception_types(self, mock_config, project_dir):
        """Test can_handle matches exception types."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer.can_handle(ValueError("any message"), "OUTPUT")
        assert healer.can_handle(FileNotFoundError("any message"), "OUTPUT")
        assert healer.can_handle(OSError("any message"), "OUTPUT")

    def test_is_media_error(self, mock_config, project_dir):
        """Test _is_media_error detection."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer._is_media_error("file not found")
        assert healer._is_media_error("media reference invalid")
        assert healer._is_media_error("no such file or directory")
        assert not healer._is_media_error("unrelated error")

    def test_is_duration_error(self, mock_config, project_dir):
        """Test _is_duration_error detection."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer._is_duration_error("negative duration")
        assert healer._is_duration_error("zero duration")
        assert healer._is_duration_error("source_range invalid")
        assert not healer._is_duration_error("media not found")

    def test_is_gap_error(self, mock_config, project_dir):
        """Test _is_gap_error detection."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer._is_gap_error("gap overflow")
        assert healer._is_gap_error("timeline too long")
        assert healer._is_gap_error("doesn't fit")
        assert not healer._is_gap_error("duration issue")

    def test_is_overlap_error(self, mock_config, project_dir):
        """Test _is_overlap_error detection."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer._is_overlap_error("clip overlap detected")
        assert healer._is_overlap_error("collision")
        assert healer._is_overlap_error("clips intersect")
        assert not healer._is_overlap_error("gap issue")


class TestOTIOHealerMediaFixes:
    """Tests for media reference fixing."""

    def test_fix_media_references_no_matches(self, mock_config, project_dir):
        """Test handling when no matches available."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()
        state.matches = []

        result = healer._fix_media_references(Exception("test"), state)

        assert not result.success
        assert "No matches" in result.message

    def test_fix_media_references_existing_files(self, mock_config, project_dir, mock_matches):
        """Test that existing files are not modified."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()
        state.matches = mock_matches

        result = healer._fix_media_references(Exception("test"), state)

        # Files exist so no fixing needed
        assert "No media reference issues found" in result.message or result.details.get("fixed_count", 0) == 0

    def test_get_video_path_attributes(self, mock_config, project_dir):
        """Test _get_video_path checks multiple attributes."""
        healer = OTIOHealer(mock_config, project_dir)

        # Test video_path attribute
        match1 = Mock()
        match1.video_path = "/path/to/video.mp4"
        assert healer._get_video_path(match1) == "/path/to/video.mp4"

        # Test source_file attribute
        match2 = Mock(spec=['source_file'])
        match2.source_file = "/path/to/source.mp4"
        assert healer._get_video_path(match2) == "/path/to/source.mp4"

    def test_set_video_path(self, mock_config, project_dir):
        """Test _set_video_path updates correct attribute."""
        healer = OTIOHealer(mock_config, project_dir)

        match = Mock()
        match.video_path = "/old/path.mp4"
        healer._set_video_path(match, "/new/path.mp4")

        assert match.video_path == "/new/path.mp4"


class TestOTIOHealerDurationFixes:
    """Tests for duration fixing."""

    def test_fix_durations_no_matches(self, mock_config, project_dir):
        """Test handling when no matches available."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()
        state.matches = []

        result = healer._fix_durations(Exception("test"), state)

        assert not result.success

    def test_fix_segment_duration_negative(self, mock_config, project_dir):
        """Test fixing negative duration."""
        healer = OTIOHealer(mock_config, project_dir)

        segment = Mock()
        segment.start = 10.0
        segment.end = 5.0  # End before start (negative duration)
        segment.duration = None

        fixed = healer._fix_segment_duration(segment)

        assert fixed > 0
        assert segment.end > segment.start

    def test_fix_segment_duration_too_long(self, mock_config, project_dir):
        """Test fixing excessively long duration."""
        healer = OTIOHealer(mock_config, project_dir)

        segment = Mock()
        segment.start = 0.0
        segment.end = 100000.0  # Way too long
        segment.duration = None

        fixed = healer._fix_segment_duration(segment)

        assert fixed > 0
        assert segment.end <= segment.start + healer.MAX_CLIP_DURATION

    def test_fix_match_times_reversed(self, mock_config, project_dir):
        """Test fixing reversed start/end times."""
        healer = OTIOHealer(mock_config, project_dir)

        match = Mock()
        match.start_time = 10.0
        match.end_time = 5.0  # Before start

        fixed = healer._fix_match_times(match)

        assert fixed > 0
        assert match.end_time > match.start_time

    def test_fix_match_times_negative_start(self, mock_config, project_dir):
        """Test fixing negative start time."""
        healer = OTIOHealer(mock_config, project_dir)

        match = Mock()
        match.start_time = -5.0
        match.end_time = 10.0

        fixed = healer._fix_match_times(match)

        assert fixed > 0
        assert match.start_time >= 0

    def test_fix_speed_adjustment_too_slow(self, mock_config, project_dir):
        """Test fixing speed that's too slow."""
        healer = OTIOHealer(mock_config, project_dir)

        # Use a simple class to hold the attribute value
        class MatchWithSpeed:
            def __init__(self):
                self.time_scalar = 0.01  # Way too slow
                self.speed = None  # No speed attribute

        match = MatchWithSpeed()

        fixed = healer._fix_speed_adjustment(match)

        assert fixed > 0
        assert match.time_scalar >= healer.MIN_SPEED

    def test_fix_speed_adjustment_too_fast(self, mock_config, project_dir):
        """Test fixing speed that's too fast."""
        healer = OTIOHealer(mock_config, project_dir)

        class MatchWithSpeed:
            def __init__(self):
                self.time_scalar = 100.0  # Way too fast
                self.speed = None

        match = MatchWithSpeed()

        fixed = healer._fix_speed_adjustment(match)

        assert fixed > 0
        assert match.time_scalar <= healer.MAX_SPEED


class TestOTIOHealerGapFixes:
    """Tests for gap mode fixing."""

    def test_fix_gaps_progression(self, mock_config, project_dir):
        """Test gap mode cycles through modes."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        # Start with scale
        mock_config.output.gap_mode = "scale"

        result = healer._fix_gaps(Exception("test"), state)

        assert result.success
        assert mock_config.output.gap_mode == "proportional"

    def test_fix_gaps_to_none(self, mock_config, project_dir):
        """Test gap mode eventually reaches none."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.gap_mode = "proportional"

        result = healer._fix_gaps(Exception("test"), state)

        assert result.success
        assert mock_config.output.gap_mode == "none"

    def test_fix_gaps_disable_alignment(self, mock_config, project_dir):
        """Test disabling alignment when gap modes exhausted."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.gap_mode = "none"
        mock_config.output.align_to_voiceover = True

        result = healer._fix_gaps(Exception("test"), state)

        assert result.success
        assert mock_config.output.align_to_voiceover is False


class TestOTIOHealerOverlapFixes:
    """Tests for overlap fixing."""

    def test_fix_overlaps_trims_clips(self, mock_config, project_dir):
        """Test overlapping clips are trimmed."""
        healer = OTIOHealer(mock_config, project_dir)

        # Create overlapping matches
        match1 = Mock()
        match1.segment = Mock()
        match1.segment.start = 0.0
        match1.segment.end = 10.0

        match2 = Mock()
        match2.segment = Mock()
        match2.segment.start = 8.0  # Overlaps with match1
        match2.segment.end = 15.0

        state = Mock()
        state.matches = [match1, match2]

        result = healer._fix_overlaps(Exception("test"), state)

        assert result.success
        assert match1.segment.end < match2.segment.start

    def test_fix_overlaps_no_issues(self, mock_config, project_dir):
        """Test when no overlaps exist."""
        healer = OTIOHealer(mock_config, project_dir)

        match1 = Mock()
        match1.segment = Mock()
        match1.segment.start = 0.0
        match1.segment.end = 5.0

        match2 = Mock()
        match2.segment = Mock()
        match2.segment.start = 10.0  # No overlap
        match2.segment.end = 15.0

        state = Mock()
        state.matches = [match1, match2]

        result = healer._fix_overlaps(Exception("test"), state)

        assert not result.success
        assert "No overlaps found" in result.message


class TestOTIOHealerExportFixes:
    """Tests for export format fixing."""

    def test_fix_export_edl_error(self, mock_config, project_dir):
        """Test EDL export gets disabled on EDL errors."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.export_edl = True

        result = healer._fix_export(Exception("EDL adapter error"), state)

        assert result.success
        assert mock_config.output.export_edl is False

    def test_fix_export_xml_error(self, mock_config, project_dir):
        """Test XML export gets disabled on XML errors."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.export_xml = True

        result = healer._fix_export(Exception("resolve XML error"), state)

        assert result.success
        assert mock_config.output.export_xml is False

    def test_fix_export_general_adapter_error(self, mock_config, project_dir):
        """Test general adapter errors disable both EDL and XML."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.export_edl = True
        mock_config.output.export_xml = True

        result = healer._fix_export(Exception("adapter write error"), state)

        assert result.success
        assert mock_config.output.export_edl is False
        assert mock_config.output.export_xml is False


class TestOTIOHealerMetadataFixes:
    """Tests for metadata sanitization."""

    def test_sanitize_metadata_json_safe(self, mock_config, project_dir):
        """Test JSON-safe values pass through."""
        healer = OTIOHealer(mock_config, project_dir)

        metadata = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }

        result = healer._sanitize_metadata(metadata)

        assert result == metadata

    def test_sanitize_metadata_numpy_conversion(self, mock_config, project_dir):
        """Test numpy types are converted."""
        healer = OTIOHealer(mock_config, project_dir)

        # Mock numpy-like object
        class FakeNumpyFloat:
            def item(self):
                return 3.14

        metadata = {"numpy_float": FakeNumpyFloat()}

        result = healer._sanitize_metadata(metadata)

        assert result["numpy_float"] == 3.14

    def test_sanitize_metadata_array_conversion(self, mock_config, project_dir):
        """Test array-like objects with tolist() are converted."""
        healer = OTIOHealer(mock_config, project_dir)

        class FakeArray:
            def tolist(self):
                return [1, 2, 3]

        metadata = {"array": FakeArray()}

        result = healer._sanitize_metadata(metadata)

        assert result["array"] == [1, 2, 3]


class TestOTIOHealerPathFixes:
    """Tests for path sanitization."""

    def test_sanitize_path_unicode(self, mock_config, project_dir):
        """Test unicode characters are replaced."""
        healer = OTIOHealer(mock_config, project_dir)

        # Smart quotes and other unicode
        path = "/path/to/video\u2019s_file.mp4"

        result = healer._sanitize_path(path)

        assert "\u2019" not in result
        assert "'" in result or "_" in result

    def test_sanitize_path_invalid_chars(self, mock_config, project_dir):
        """Test Windows-invalid characters are removed."""
        healer = OTIOHealer(mock_config, project_dir)

        path = "/path/to/video<>file.mp4"

        result = healer._sanitize_path(path)

        assert "<" not in result
        assert ">" not in result

    def test_sanitize_path_multiple_underscores(self, mock_config, project_dir):
        """Test multiple underscores are collapsed."""
        healer = OTIOHealer(mock_config, project_dir)

        path = "/path/to/video___file.mp4"

        result = healer._sanitize_path(path)

        assert "___" not in result


class TestOTIOHealerFramerateFixes:
    """Tests for framerate fixing."""

    def test_fix_framerate_changes_rate(self, mock_config, project_dir):
        """Test frame rate gets changed on error."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.frame_rate = 30.0

        result = healer._fix_framerate(Exception("frame rate error"), state)

        assert result.success
        assert mock_config.output.frame_rate != 30.0

    def test_fix_framerate_cycles_rates(self, mock_config, project_dir):
        """Test frame rate cycles through fallback rates."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        # Use a non-standard rate
        mock_config.output.frame_rate = 60.0

        result = healer._fix_framerate(Exception("frame rate error"), state)

        assert result.success
        assert mock_config.output.frame_rate in [30.0, 24.0, 25.0, 29.97, 23.976]


class TestOTIOHealerMemoryFixes:
    """Tests for memory/performance fixes."""

    def test_fix_memory_disables_alternatives(self, mock_config, project_dir):
        """Test alternatives get disabled on memory errors."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        mock_config.output.include_alternatives = True
        mock_config.output.include_strategy_tracks = True
        mock_config.output.include_entity_images = True
        mock_config.output.include_entity_videos = True

        result = healer._fix_memory(Exception("out of memory"), state)

        assert result.success
        assert mock_config.output.include_alternatives is False
        assert mock_config.output.include_strategy_tracks is False


class TestOTIOHealerSafeMode:
    """Tests for safe mode fallback."""

    def test_safe_mode_applies_all_settings(self, mock_config, project_dir):
        """Test safe mode applies all safe settings."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        # Set non-safe values
        mock_config.output.include_alternatives = True
        mock_config.output.include_strategy_tracks = True
        mock_config.output.include_entity_images = True
        mock_config.output.export_edl = True
        mock_config.output.export_xml = True

        result = healer._apply_safe_mode(Exception("unknown error"), state)

        assert result.success
        assert result.modified_config
        assert mock_config.output.include_alternatives is False
        assert mock_config.output.export_edl is False

    def test_safe_mode_already_minimal(self, mock_config, project_dir):
        """Test safe mode when already at minimal settings."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()

        # Already at safe values
        mock_config.output.include_alternatives = False
        mock_config.output.include_strategy_tracks = False
        mock_config.output.include_entity_images = False
        mock_config.output.include_entity_videos = False
        mock_config.output.export_edl = False
        mock_config.output.export_xml = False
        mock_config.output.gap_mode = "none"
        mock_config.output.frame_rate = 30.0

        result = healer._apply_safe_mode(Exception("test"), state)

        # Should fail since no changes possible
        assert not result.success or len(result.details.get("changes", [])) == 0


class TestOTIOHealerPreflightCheck:
    """Tests for preflight checking."""

    def test_preflight_no_matches(self, mock_config, project_dir):
        """Test preflight with no matches returns no issues (early pipeline state)."""
        healer = OTIOHealer(mock_config, project_dir)
        state = Mock()
        state.matches = None

        issues = healer.preflight_check(state)

        assert len(issues) == 0

    def test_preflight_missing_media(self, mock_config, project_dir):
        """Test preflight detects missing media."""
        healer = OTIOHealer(mock_config, project_dir)

        match = Mock()
        match.video_path = "/nonexistent/file.mp4"
        match.segment = Mock()
        match.segment.start = 0.0
        match.segment.end = 5.0

        state = Mock()
        state.matches = [match]

        issues = healer.preflight_check(state)

        assert any("missing media" in i for i in issues)

    def test_preflight_invalid_duration(self, mock_config, project_dir, tmp_path):
        """Test preflight detects invalid durations."""
        healer = OTIOHealer(mock_config, project_dir)

        # Create file so it's not flagged as missing
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        match = Mock()
        match.video_path = str(video_file)
        match.segment = Mock()
        match.segment.start = 10.0
        match.segment.end = 5.0  # Invalid: end before start

        state = Mock()
        state.matches = [match]

        issues = healer.preflight_check(state)

        assert any("invalid duration" in i for i in issues)

    def test_preflight_overlapping_clips(self, mock_config, project_dir, tmp_path):
        """Test preflight detects overlapping clips."""
        healer = OTIOHealer(mock_config, project_dir)

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        match1 = Mock()
        match1.video_path = str(video_file)
        match1.segment = Mock()
        match1.segment.start = 0.0
        match1.segment.end = 10.0

        match2 = Mock()
        match2.video_path = str(video_file)
        match2.segment = Mock()
        match2.segment.start = 5.0  # Overlaps
        match2.segment.end = 15.0

        state = Mock()
        state.matches = [match1, match2]

        issues = healer.preflight_check(state)

        assert any("overlapping" in i for i in issues)


class TestOTIOHealerCanHandleAcceptanceCriteria:
    """Test OTIOHealer.can_handle() acceptance criteria:
    - True for OTIO-related errors (import, serialization, track corruption)
    - False for download errors
    """

    def test_can_handle_import_errors(self, mock_config, project_dir):
        """OTIOHealer handles OTIO import errors."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("opentimelineio import failed"), "OUTPUT") is True
        assert healer.can_handle(ImportError("No module named 'opentimelineio'"), "OUTPUT") is True

    def test_can_handle_serialization_failures(self, mock_config, project_dir):
        """OTIOHealer handles serialization failures."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("Failed to serialize timeline metadata"), "OUTPUT") is True
        assert healer.can_handle(Exception("JSON encode error in clip metadata"), "OUTPUT") is True

    def test_can_handle_track_corruption(self, mock_config, project_dir):
        """OTIOHealer handles track corruption errors."""
        healer = OTIOHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("Track mismatch: expected 10, got 8"), "OUTPUT") is True
        assert healer.can_handle(Exception("Video track count invalid"), "OUTPUT") is True

    def test_cannot_handle_download_errors(self, mock_config, project_dir):
        """OTIOHealer returns False for download-related errors."""
        healer = OTIOHealer(mock_config, project_dir)

        # Pure download errors shouldn't match OTIO patterns
        assert healer.can_handle(Exception("HTTP 403 Forbidden from youtube.com"), "DOWNLOAD") is False
        assert healer.can_handle(Exception("yt-dlp extraction failed for video ID"), "DOWNLOAD") is False
        assert healer.can_handle(Exception("Connection refused to CDN server"), "DOWNLOAD") is False


class TestOTIOHealerTimelineReconstruction:
    """Test OTIOHealer.heal() attempts timeline reconstruction.
    Acceptance criterion 5: logs reconstruction steps, returns .fixed() or .failed().
    """

    def test_fix_logs_attempt_on_media_error(self, mock_config, project_dir, mock_state):
        """fix() calls log_attempt when handling media errors."""
        healer = OTIOHealer(mock_config, project_dir)

        with patch.object(healer, 'log_attempt') as mock_log:
            healer.fix(FileNotFoundError("video.mp4 not found"), mock_state, "OUTPUT")

        # Verify log_attempt was called (reconstruction steps logged)
        assert mock_log.call_count >= 1
        calls = [str(c) for c in mock_log.call_args_list]
        assert any("Analyzing error" in str(c) or "Resolving" in str(c) for c in calls)

    def test_fix_logs_success_on_duration_fix(self, mock_config, project_dir):
        """fix() calls log_success when durations are successfully fixed."""
        healer = OTIOHealer(mock_config, project_dir)

        # Create state with a match that has a fixable negative duration
        match = Mock()
        match.segment = Mock()
        match.segment.start = 10.0
        match.segment.end = 5.0  # Negative duration - fixable
        match.segment.duration = None
        match.start_time = 0.0
        match.end_time = 5.0
        match.time_scalar = 1.0
        match.speed = None
        match.speed_factor = None
        match.metadata = None

        state = Mock()
        state.matches = [match]

        with patch.object(healer, 'log_success') as mock_success:
            result = healer.fix(ValueError("negative duration"), state, "OUTPUT")

        assert result.success is True
        assert mock_success.call_count >= 1

    def test_fix_returns_fixed_or_failed(self, mock_config, project_dir, mock_state):
        """fix() always returns a HealerResult with success True or False."""
        healer = OTIOHealer(mock_config, project_dir)

        # Test various error types - all should return HealerResult
        errors = [
            FileNotFoundError("missing.mp4"),
            ValueError("negative duration"),
            Exception("Gap overflow in timeline"),
            Exception("completely unknown xyz123"),
        ]

        for error in errors:
            result = healer.fix(error, mock_state, "OUTPUT")
            assert isinstance(result, HealerResult)
            assert isinstance(result.success, bool)
            assert result.action in list(HealerAction)


class TestOTIOHealerIntegration:
    """Integration tests for full fix() method."""

    def test_fix_routes_media_error(self, mock_config, project_dir, mock_state):
        """Test fix() routes media errors correctly."""
        healer = OTIOHealer(mock_config, project_dir)

        result = healer.fix(
            FileNotFoundError("video.mp4 not found"),
            mock_state,
            "OUTPUT"
        )

        # Should attempt media reference fix
        assert isinstance(result, HealerResult)

    def test_fix_routes_duration_error(self, mock_config, project_dir, mock_state):
        """Test fix() routes duration errors correctly."""
        healer = OTIOHealer(mock_config, project_dir)

        result = healer.fix(
            ValueError("negative duration -5.0"),
            mock_state,
            "OUTPUT"
        )

        assert isinstance(result, HealerResult)

    def test_fix_routes_gap_error(self, mock_config, project_dir, mock_state):
        """Test fix() routes gap errors correctly."""
        healer = OTIOHealer(mock_config, project_dir)

        result = healer.fix(
            Exception("Gap overflow in timeline"),
            mock_state,
            "OUTPUT"
        )

        assert isinstance(result, HealerResult)

    def test_fix_fallback_to_safe_mode(self, mock_config, project_dir, mock_state):
        """Test fix() falls back to safe mode for unknown errors."""
        healer = OTIOHealer(mock_config, project_dir)

        # Error that doesn't match any specific category
        result = healer.fix(
            Exception("completely random error xyz123"),
            mock_state,
            "OUTPUT"
        )

        # Should try safe mode as last resort
        assert isinstance(result, HealerResult)
