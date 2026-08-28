"""Integration tests for scripts/imagen_otio.py."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import opentimelineio as otio

from src.state import VoiceoverSegment, GeneratedImageBatch, GeneratedImageResult
from src.config.sections.generated_images import GeneratedImagesConfig, GeneratedImageSizeConfig


class TestDryRunModel:
    """Test 1: dry-run with gemini-2.5-flash-image shows correct model in output."""

    def test_dry_run_shows_gemini_2_5_flash_image_model(self, tmp_path, capsys):
        """Dry-run output should display gemini-2.5-flash-image model."""
        from scripts.imagen_otio import main

        # Create a minimal SRT file
        srt_path = tmp_path / "voiceover" / "test.srt"
        srt_path.parent.mkdir(parents=True)
        srt_path.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nSample voiceover text.\n\n",
            encoding="utf-8",
        )

        config = GeneratedImagesConfig(
            enabled=True,
            image_size=GeneratedImageSizeConfig(width=1408, height=768),
            quality="standard",
            budget_usd=2.0,
            min_segments_per_image=7,
            max_segments_per_image=9,
            batch_target_segments=8,
            provider="gemini-flash",
            model="gemini-2.5-flash-image",
        )

        with patch("scripts.imagen_otio.find_srt", return_value=srt_path):
            with patch("scripts.imagen_otio.parse_srt") as mock_parse:
                mock_parse.return_value = [
                    VoiceoverSegment(index=1, start=1.0, end=4.0, text="Sample voiceover text."),
                ]
                with patch("scripts.imagen_otio.GeneratedImagesConfig", return_value=config):
                    with patch("scripts.imagen_otio.build_generated_image_batches") as mock_batches:
                        batch = GeneratedImageBatch(
                            batch_id="generated_000",
                            segment_start_index=1,
                            segment_end_index=1,
                            segment_count=1,
                            start_time=1.0,
                            end_time=4.0,
                            text="Sample voiceover text.",
                            segment_indices=[1],
                        )
                        mock_batches.return_value = [batch]

                        # Patch sys.argv to include --dry-run
                        with patch.object(sys, "argv", [
                            "imagen_otio.py",
                            str(tmp_path / "voiceover"),
                            "--dry-run",
                            "--size", "1408x768",
                        ]):
                            try:
                                main()
                            except SystemExit:
                                pass

        captured = capsys.readouterr()
        assert "gemini-2.5-flash-image" in captured.out, (
            f"Expected model 'gemini-2.5-flash-image' in dry-run output, got:\n{captured.out}"
        )


class TestDryRunCost:
    """Test 2: dry-run shows correct cost $0.039 per image."""

    def test_dry_run_cost_is_0_039_per_image(self, tmp_path, capsys):
        """Dry-run output should show $0.039 per image for gemini-2.5-flash-image."""
        from scripts.imagen_otio import main

        srt_path = tmp_path / "voiceover" / "test.srt"
        srt_path.parent.mkdir(parents=True)
        srt_path.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nSample voiceover text.\n\n",
            encoding="utf-8",
        )

        config = GeneratedImagesConfig(
            enabled=True,
            image_size=GeneratedImageSizeConfig(width=1408, height=768),
            quality="standard",
            budget_usd=2.0,
            min_segments_per_image=7,
            max_segments_per_image=9,
            batch_target_segments=8,
            provider="gemini-flash",
            model="gemini-2.5-flash-image",
        )

        with patch("scripts.imagen_otio.find_srt", return_value=srt_path):
            with patch("scripts.imagen_otio.parse_srt") as mock_parse:
                mock_parse.return_value = [
                    VoiceoverSegment(index=1, start=1.0, end=4.0, text="Sample voiceover text."),
                ]
                with patch("scripts.imagen_otio.GeneratedImagesConfig", return_value=config):
                    with patch("scripts.imagen_otio.build_generated_image_batches") as mock_batches:
                        batch = GeneratedImageBatch(
                            batch_id="generated_000",
                            segment_start_index=1,
                            segment_end_index=1,
                            segment_count=1,
                            start_time=1.0,
                            end_time=4.0,
                            text="Sample voiceover text.",
                            segment_indices=[1],
                        )
                        mock_batches.return_value = [batch]

                        with patch.object(sys, "argv", [
                            "imagen_otio.py",
                            str(tmp_path / "voiceover"),
                            "--dry-run",
                            "--size", "1408x768",
                        ]):
                            try:
                                main()
                            except SystemExit:
                                pass

        captured = capsys.readouterr()
        # Cost per image for gemini-2.5-flash-image is $0.039
        assert "0.039" in captured.out, (
            f"Expected cost '$0.039' per image in dry-run output, got:\n{captured.out}"
        )


class TestBuildV12Otio:
    """Tests 3-6: build_v12_otio functionality."""

    @pytest.fixture
    def mock_results(self):
        """Produce fake GeneratedImageResult objects."""
        # Create a temp file so the path exists
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fake_path = f.name

        results = [
            GeneratedImageResult(
                batch_id="generated_000",
                file=fake_path,
                prompt="A sample prompt for testing",
                width=1408,
                height=768,
                start_time=1.0,
                end_time=4.0,
                segment_start_index=1,
                segment_end_index=2,
                segment_indices=[1, 2],
                cost_usd=0.039,
            ),
            GeneratedImageResult(
                batch_id="generated_001",
                file=fake_path,
                prompt="Another sample prompt",
                width=1408,
                height=768,
                start_time=4.0,
                end_time=7.0,
                segment_start_index=3,
                segment_end_index=4,
                segment_indices=[3, 4],
                cost_usd=0.039,
            ),
        ]
        return results

    def test_timeline_tracks_name_is_empty_string(self, mock_results):
        """Test 3: timeline.tracks.name should be '' (empty string, DaVinci convention)."""
        from scripts.imagen_otio import build_v12_otio

        timeline = build_v12_otio(mock_results, global_start_frame=0)
        assert timeline.tracks.name == "", (
            f"Expected timeline.tracks.name to be '', got '{timeline.tracks.name}'"
        )

    def test_v12_is_direct_child_of_timeline_tracks_no_inner_stack(self, mock_results):
        """Test 4: V12 track should be direct child of timeline.tracks, no inner Stack."""
        from scripts.imagen_otio import build_v12_otio

        timeline = build_v12_otio(mock_results, global_start_frame=0)

        # timeline.tracks should be a Stack
        assert isinstance(timeline.tracks, otio.schema.Stack), (
            f"Expected timeline.tracks to be Stack, got {type(timeline.tracks)}"
        )

        # Children of timeline.tracks should NOT include any Stack — V12 must be direct
        for child in timeline.tracks:
            assert not isinstance(child, otio.schema.Stack), (
                f"Found inner Stack inside timeline.tracks: {child}. "
                "V12 should be a direct child, not wrapped in another Stack."
            )

        # V12 track should be present as a direct child
        v12_tracks = [c for c in timeline.tracks if isinstance(c, otio.schema.Track) and c.name == "V12 - Generated Images"]
        assert len(v12_tracks) == 1, (
            f"Expected exactly one 'V12 - Generated Images' track as direct child, found {len(v12_tracks)}"
        )

    def test_clip_has_media_reference_set_not_none(self, mock_results):
        """Test 5: clip.media_reference should be set (not None)."""
        from scripts.imagen_otio import build_v12_otio

        timeline = build_v12_otio(mock_results, global_start_frame=0)

        # Find clips in the V12 track
        v12_track = None
        for child in timeline.tracks:
            if isinstance(child, otio.schema.Track) and child.name == "V12 - Generated Images":
                v12_track = child
                break

        assert v12_track is not None, "V12 track not found"

        clips = [item for item in v12_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 1, "No clips found in V12 track"

        for clip in clips:
            assert clip.media_reference is not None, (
                f"Clip '{clip.name}' has media_reference=None, expected ExternalReference"
            )

    def test_clip_has_freeze_frame_effect(self, mock_results):
        """Test 6: clip should have FreezeFrame effect."""
        from scripts.imagen_otio import build_v12_otio

        timeline = build_v12_otio(mock_results, global_start_frame=0)

        # Find the V12 track
        v12_track = None
        for child in timeline.tracks:
            if isinstance(child, otio.schema.Track) and child.name == "V12 - Generated Images":
                v12_track = child
                break

        assert v12_track is not None, "V12 track not found"

        clips = [item for item in v12_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 1, "No clips found in V12 track"

        for clip in clips:
            freeze_frames = [eff for eff in clip.effects if isinstance(eff, otio.schema.FreezeFrame)]
            assert len(freeze_frames) == 1, (
                f"Clip '{clip.name}' expected exactly one FreezeFrame effect, found {len(freeze_frames)}"
            )


class TestBuildV12OtioWithMockedService:
    """Integration test using mocked GeneratedImageService with fake results."""

    def test_build_v12_otio_integration_with_mocked_service(self, tmp_path):
        """Full integration: mock GeneratedImageService, verify all timeline properties."""
        from scripts.imagen_otio import build_v12_otio
        import tempfile

        # Create fake image files
        fake_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fake_files.append(f.name)

        results = [
            GeneratedImageResult(
                batch_id="generated_000",
                file=fake_files[0],
                prompt="First prompt",
                width=1408,
                height=768,
                start_time=1.0,
                end_time=4.0,
                segment_start_index=1,
                segment_end_index=2,
                segment_indices=[1, 2],
                cost_usd=0.039,
            ),
            GeneratedImageResult(
                batch_id="generated_001",
                file=fake_files[1],
                prompt="Second prompt",
                width=1408,
                height=768,
                start_time=5.0,
                end_time=8.0,
                segment_start_index=3,
                segment_end_index=4,
                segment_indices=[3, 4],
                cost_usd=0.039,
            ),
        ]

        timeline = build_v12_otio(results, global_start_frame=0)

        # Test 3: tracks.name = ''
        assert timeline.tracks.name == ""

        # Test 4: V12 direct child, no inner Stack
        for child in timeline.tracks:
            assert not isinstance(child, otio.schema.Stack)
        v12_count = sum(1 for c in timeline.tracks if isinstance(c, otio.schema.Track) and c.name == "V12 - Generated Images")
        assert v12_count == 1

        # Test 5: clip media_reference not None
        v12 = next(c for c in timeline.tracks if isinstance(c, otio.schema.Track))
        for item in v12:
            if isinstance(item, otio.schema.Clip):
                assert item.media_reference is not None

        # Test 6: FreezeFrame effect
        for item in v12:
            if isinstance(item, otio.schema.Clip):
                assert any(isinstance(eff, otio.schema.FreezeFrame) for eff in item.effects)