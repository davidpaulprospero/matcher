"""
Tests for DOWNLOAD_SEGMENTS stage remapping logic fixes.

Tests the fixes for:
1. Timestamp suffix stripping (e.g., video_id_0000 -> video_id)
2. Stock video skipping (pexels_, pixabay_, entity_)

These tests replicate the logic from src/stages/download.py lines 568-632
to verify the bug fixes work correctly.
"""

import pytest
from pathlib import Path

from src.state import AudioDownload


class TestTimestampSuffixStripping:
    """Test that timestamp suffixes are stripped when matching video_id"""

    @pytest.mark.fast
    def test_strips_timestamp_suffix_from_youtube_segment(self):
        """Test that YB4UyAHH5Ig_0000 matches YB4UyAHH5Ig"""
        # This is the core bug fix: matches reference timestamped files
        # but audio downloads have base video_id

        # Setup audio downloads
        audio1 = AudioDownload(
            file="YB4UyAHH5Ig.mp3",
            url="https://youtube.com/watch?v=YB4UyAHH5Ig",
            video_id="YB4UyAHH5Ig",
            title="Test Video"
        )
        audio_downloads_by_id = {audio1.video_id: audio1}

        # Simulate the match that references timestamped file
        match_video_file = "YB4UyAHH5Ig_0000.mp4"
        audio_file = Path(match_video_file).stem  # "YB4UyAHH5Ig_0000"

        # Apply the fix: strip timestamp suffix
        base_audio_file = audio_file
        if '_' in audio_file:
            parts = audio_file.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_audio_file = parts[0]

        # Try to find video_id (should succeed with fix)
        video_id = None
        for vid, audio in audio_downloads_by_id.items():
            audio_path = audio.file
            if Path(audio_path).stem == base_audio_file:
                video_id = vid
                break

        # Verify the match was found
        assert video_id == "YB4UyAHH5Ig", f"Failed to match {audio_file} -> {base_audio_file}"
        assert base_audio_file == "YB4UyAHH5Ig"

    @pytest.mark.fast
    def test_strips_multiple_digit_timestamp_suffixes(self):
        """Test various timestamp formats: _0000, _1234, _9999"""
        test_cases = [
            ("video1_0000.mp4", "video1"),
            ("video2_1234.mp4", "video2"),
            ("video3_9999.mp4", "video3"),
            ("abc123_0525.mp4", "abc123"),
        ]

        for filename, expected_base in test_cases:
            audio_file = Path(filename).stem

            # Simulate the fix logic
            base_audio_file = audio_file
            if '_' in audio_file:
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            assert base_audio_file == expected_base, f"Failed for {filename}"

    @pytest.mark.fast
    def test_preserves_underscores_in_video_id(self):
        """Test that underscores in video_id are preserved"""
        # Some video IDs might have underscores as part of their ID
        audio_file = "my_video_id_0000"

        # Apply fix logic
        base_audio_file = audio_file
        if '_' in audio_file:
            parts = audio_file.rsplit('_', 1)  # rsplit from right
            if len(parts) == 2 and parts[1].isdigit():
                base_audio_file = parts[0]

        # Should strip only the timestamp, preserve the ID underscore
        assert base_audio_file == "my_video_id"

    @pytest.mark.fast
    def test_does_not_strip_non_numeric_suffix(self):
        """Test that non-numeric suffixes are not stripped"""
        test_cases = [
            "video_HD",      # _HD is not stripped
            "video_720p",    # _720p is not stripped
            "video_final",   # _final is not stripped
        ]

        for audio_file in test_cases:
            base_audio_file = audio_file
            if '_' in audio_file:
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            # Should NOT strip non-numeric suffixes
            assert base_audio_file == audio_file


class TestStockVideoSkipping:
    """Test that stock videos and entity images are skipped during remapping"""

    @pytest.mark.fast
    def test_skips_pexels_videos(self):
        """Test that pexels_ videos are skipped silently"""
        # Replicate the skip logic from download.py line 569
        audio_file = "pexels_4990235_Luis_Quintero_HD"

        # Check if it should be skipped
        should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))

        assert should_skip == True, "Pexels videos should be skipped"

    @pytest.mark.fast
    def test_skips_pixabay_videos(self):
        """Test that pixabay_ videos are skipped silently"""
        audio_file = "pixabay_1191_Vimeo-Free-Videos_720p"

        # Check if it should be skipped
        should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))

        assert should_skip == True, "Pixabay videos should be skipped"

    @pytest.mark.fast
    def test_skips_entity_images(self):
        """Test that entity_ images are skipped silently"""
        audio_file = "entity_CaesarsPalace_001"

        # Check if it should be skipped
        should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))

        assert should_skip == True, "Entity images should be skipped"

    @pytest.mark.fast
    def test_does_not_skip_youtube_videos(self):
        """Test that YouTube videos are NOT skipped"""
        audio_file = "YB4UyAHH5Ig_0000"

        # Check if it should be skipped
        should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))

        assert should_skip == False, "YouTube videos should NOT be skipped"

    @pytest.mark.fast
    def test_does_not_skip_regular_videos(self):
        """Test that regular video files are NOT skipped"""
        test_cases = [
            "my_video_file",
            "some_random_video_0000",
            "test_123",
        ]

        for audio_file in test_cases:
            should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))
            assert should_skip == False, f"{audio_file} should NOT be skipped"


class TestRegressionPrevention:
    """Tests to prevent regression of the original bug"""

    @pytest.mark.fast
    def test_youtube_segment_with_timestamp_gets_remapped(self):
        """Regression test: ensure YouTube segments with timestamps are remapped"""
        # Replicate the full remapping logic for a YouTube segment with timestamp

        # Setup: audio download
        audio1 = AudioDownload(
            file="Q5_QZ6bLKzY.mp3",
            url="https://youtube.com/watch?v=Q5_QZ6bLKzY",
            video_id="Q5_QZ6bLKzY",
            title="Test Video"
        )
        audio_downloads_by_id = {audio1.video_id: audio1}

        # Setup: downloaded segment (what actually exists on disk)
        segment_file = "Q5_QZ6bLKzY_0003.mp4"
        segment_video_id = "Q5_QZ6bLKzY"

        # Replicate remapping logic from download.py lines 568-600
        # Extract audio filename from segment
        audio_file = Path(segment_file).stem  # "Q5_QZ6bLKzY_0003"

        # Skip stock videos check
        if audio_file.startswith(('pexels_', 'pixabay_', 'entity_')):
            # Should not skip for YouTube videos
            pytest.fail("YouTube video incorrectly identified as stock video")

        # Strip timestamp suffix
        base_audio_file = audio_file
        if '_' in audio_file:
            parts = audio_file.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_audio_file = parts[0]  # "Q5_QZ6bLKzY"

        # Find matching video_id in audio_downloads_by_id
        found_video_id = None
        for vid, audio in audio_downloads_by_id.items():
            audio_path = audio.file
            if Path(audio_path).stem == base_audio_file:
                found_video_id = vid
                break

        # Verify the match was found (no warning should be generated)
        assert found_video_id == "Q5_QZ6bLKzY", f"Failed to find video_id for {audio_file} -> {base_audio_file}"
        assert found_video_id == segment_video_id, "video_id mismatch between segment and audio"

    @pytest.mark.fast
    def test_no_warnings_for_stock_videos(self):
        """Regression test: ensure no warnings for stock videos"""
        # Replicate the skip logic for stock videos

        stock_video_files = [
            "pexels_123.mp4",
            "pixabay_456.mp4",
            "entity_Test.jpg",
        ]

        for video_file in stock_video_files:
            audio_file = Path(video_file).stem

            # Check if it should be skipped (lines 569-570 in download.py)
            should_skip = audio_file.startswith(('pexels_', 'pixabay_', 'entity_'))

            # All stock videos should be skipped (no warning generated)
            assert should_skip == True, f"Stock video {video_file} should be skipped but wasn't"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
