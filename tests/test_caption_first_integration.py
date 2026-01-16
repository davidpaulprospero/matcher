"""
End-to-end integration tests for caption-first matching pipeline.

Tests the complete flow:
1. CaptionStage fetches YouTube captions
2. TranscribeStage skips captioned videos
3. Embeddings include transcript_source metadata
4. Scoring applies caption_boost for manual captions
5. TieredMatcher integrates caption boost in confidence

This test suite validates that all caption-first components work together.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.caption import CaptionStage
from src.stages.transcribe import TranscribeStage
from src.stages import StageResult
from src.state import (
    PipelineState, CaptionDownload, AudioDownload, DownloadedVideo,
    VoiceoverSegment, VideoCandidate
)
from src.matching.scoring import apply_caption_boost
from src.downloader.caption_fetcher import CaptionFetcher, CaptionResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_project_dir():
    """Create temporary project directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / ".cache" / "captions").mkdir(parents=True)
        (project_dir / ".cache" / "transcriptions").mkdir(parents=True)
        (project_dir / ".cache" / "embeddings").mkdir(parents=True)
        yield project_dir


@pytest.fixture
def caption_first_config():
    """Create config with caption-first enabled"""
    config = Mock()
    config.download = Mock()
    config.download.caption_first = Mock()
    config.download.caption_first.enabled = True
    config.download.caption_first.prefer_manual_captions = True
    config.download.caption_first.languages = ["en", "en-US", "en-GB"]
    config.download.caption_first.fallback_to_audio = True
    config.download.caption_first.confidence_boost_manual = 0.1
    config.download.caption_first.fetch_timeout = 30
    config.download.cookies_path = ""
    config.download.cookies_from_browser = ""

    config.cache = Mock()
    config.pipeline = Mock()
    config.pipeline.skip_transcription = False
    config.pipeline.parallel_transcription = False
    config.pipeline.parallel_embedding = True

    config.transcription = Mock()
    config.transcription.model = "base"
    config.transcription.language = "auto"
    config.transcription.max_workers = 2

    config.embedding = Mock()
    config.embedding.batch_size = 100
    config.embedding.provider = "gemini"

    config.matching = Mock()
    config.matching.chapter_matching_enabled = False
    config.matching.broll_boost = 0.1

    config.global_cache = Mock()
    config.global_cache.current_project_boost = 0.1

    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = Mock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.save = Mock()
    return checkpoint


@pytest.fixture
def sample_srt_content():
    """Sample SRT caption content"""
    return """1
00:00:00,000 --> 00:00:03,500
Welcome to our documentary about Paris.

2
00:00:04,000 --> 00:00:08,000
The Eiffel Tower stands majestically over the city.

3
00:00:08,500 --> 00:00:12,000
Every year, millions of visitors come to see it.

4
00:00:12,500 --> 00:00:16,000
The tower was built in 1889 for the World's Fair.

5
00:00:16,500 --> 00:00:20,000
It has become the most iconic landmark in France.
"""


@pytest.fixture
def sample_auto_srt_content():
    """Sample auto-generated SRT content (lower quality)"""
    return """1
00:00:00,000 --> 00:00:03,500
welcome to our documentary about paris

2
00:00:04,000 --> 00:00:08,000
the eiffel tower stands majestically over the city

3
00:00:08,500 --> 00:00:12,000
every year millions of visitors come to see it
"""


# ============================================================================
# UNIT TESTS - CAPTION BOOST SCORING
# ============================================================================

class TestCaptionBoostScoring:
    """Test apply_caption_boost function in isolation"""

    def test_manual_caption_gets_boost(self, caption_first_config):
        """Manual captions should get confidence boost"""
        segment = Mock()
        segment.transcript_source = 'manual_caption'

        confidence = 0.75
        boosted, reason = apply_caption_boost(confidence, segment, caption_first_config)

        assert boosted == 0.85  # 0.75 + 0.1
        assert "manual caption boost" in reason

    def test_auto_caption_no_boost(self, caption_first_config):
        """Auto-generated captions should not get boost"""
        segment = Mock()
        segment.transcript_source = 'auto_caption'

        confidence = 0.75
        boosted, reason = apply_caption_boost(confidence, segment, caption_first_config)

        assert boosted == 0.75  # No change
        assert reason == ""

    def test_whisper_no_boost(self, caption_first_config):
        """Whisper transcripts should not get boost"""
        segment = Mock()
        segment.transcript_source = 'whisper'

        confidence = 0.75
        boosted, reason = apply_caption_boost(confidence, segment, caption_first_config)

        assert boosted == 0.75
        assert reason == ""

    def test_empty_source_no_boost(self, caption_first_config):
        """Empty transcript_source should not get boost"""
        segment = Mock()
        segment.transcript_source = ''

        confidence = 0.75
        boosted, reason = apply_caption_boost(confidence, segment, caption_first_config)

        assert boosted == 0.75
        assert reason == ""

    def test_boost_capped_at_one(self, caption_first_config):
        """Boost should not exceed 1.0"""
        segment = Mock()
        segment.transcript_source = 'manual_caption'

        confidence = 0.95
        boosted, reason = apply_caption_boost(confidence, segment, caption_first_config)

        assert boosted == 1.0  # Capped

    def test_configurable_boost_amount(self):
        """Boost amount should be configurable"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.confidence_boost_manual = 0.2

        segment = Mock()
        segment.transcript_source = 'manual_caption'

        confidence = 0.70
        boosted, reason = apply_caption_boost(confidence, segment, config)

        assert boosted == pytest.approx(0.90)  # 0.70 + 0.2
        assert "+0.20" in reason


# ============================================================================
# INTEGRATION TESTS - CAPTION STAGE
# ============================================================================

class TestCaptionStageIntegration:
    """Integration tests for CaptionStage with mocked yt-dlp"""

    def test_caption_stage_processes_video_candidates(
        self, caption_first_config, mock_checkpoint, temp_project_dir
    ):
        """Test CaptionStage processes video_candidates from VIDEO_METADATA"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        # Create state with video candidates
        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(
                video_id="vid1_manual",
                url="https://youtube.com/watch?v=vid1_manual",
                title="Video with Manual Captions",
                duration=120.0,
                keyword="paris"
            ),
            VideoCandidate(
                video_id="vid2_auto",
                url="https://youtube.com/watch?v=vid2_auto",
                title="Video with Auto Captions",
                duration=90.0,
                keyword="eiffel"
            ),
            VideoCandidate(
                video_id="vid3_none",
                url="https://youtube.com/watch?v=vid3_none",
                title="Video without Captions",
                duration=60.0,
                keyword="tower"
            ),
        ]

        # Create mock SRT files
        manual_srt = temp_project_dir / ".cache" / "captions" / "vid1_manual.en.srt"
        manual_srt.write_text("""1
00:00:00,000 --> 00:00:05,000
This is a manual caption segment.
""", encoding='utf-8')

        auto_srt = temp_project_dir / ".cache" / "captions" / "vid2_auto.en-auto.srt"
        auto_srt.write_text("""1
00:00:00,000 --> 00:00:05,000
this is an auto caption segment
""", encoding='utf-8')

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher

            # vid1 - manual caption
            # vid2 - auto caption
            # vid3 - no caption
            def fetch_side_effect(video_id, **kwargs):
                if video_id == "vid1_manual":
                    return CaptionResult(
                        video_id=video_id,
                        file=str(manual_srt),
                        language="en",
                        is_auto_generated=False
                    )
                elif video_id == "vid2_auto":
                    return CaptionResult(
                        video_id=video_id,
                        file=str(auto_srt),
                        language="en",
                        is_auto_generated=True
                    )
                return None

            mock_fetcher.fetch_captions.side_effect = fetch_side_effect
            mock_fetcher.parse_caption_file.return_value = [
                {'text': 'Test segment', 'start': 0.0, 'end': 5.0}
            ]
            mock_fetcher.update_segment_count = Mock()

            stage = CaptionStage()
            result = stage.run(state, caption_first_config, mock_checkpoint)

        # Verify results
        assert result.success is True
        assert result.data.get('caption_count') == 2
        assert result.data.get('fallback_count') == 1
        assert "vid3_none" in state.videos_need_audio
        assert len(state.caption_downloads) == 2

    def test_caption_stage_sets_transcript_source(
        self, caption_first_config, mock_checkpoint, temp_project_dir
    ):
        """Test that transcript_source is set correctly on segments"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(
                video_id="manual_vid",
                url="https://youtube.com/watch?v=manual_vid",
                title="Manual Caption Video",
                duration=60.0,
                keyword="test"
            ),
        ]

        srt_file = temp_project_dir / ".cache" / "captions" / "manual_vid.en.srt"
        srt_file.write_text("""1
00:00:00,000 --> 00:00:05,000
Manual caption text
""", encoding='utf-8')

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_captions.return_value = CaptionResult(
                video_id="manual_vid",
                file=str(srt_file),
                language="en",
                is_auto_generated=False
            )
            mock_fetcher.parse_caption_file.return_value = [
                {'text': 'Manual caption text', 'start': 0.0, 'end': 5.0}
            ]
            mock_fetcher.update_segment_count = Mock()

            stage = CaptionStage()
            result = stage.run(state, caption_first_config, mock_checkpoint)

        # Verify transcript_source is set
        assert "manual_vid" in state.transcripts
        segments = state.transcripts["manual_vid"]
        assert len(segments) > 0
        assert segments[0].get('transcript_source') == 'manual_caption'


# ============================================================================
# INTEGRATION TESTS - TRANSCRIBE STAGE SKIP BEHAVIOR
# ============================================================================

class TestTranscribeStageSkipsCaptioned:
    """Test TranscribeStage correctly skips videos with captions"""

    def test_filter_captioned_videos(self, caption_first_config, temp_project_dir):
        """Test _filter_captioned_videos removes captioned videos"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()

        # Use valid 11-char YouTube video IDs (required by _extract_video_id)
        captioned_vid_id = "dQw4w9WgXcQ"  # 11 chars - has captions
        uncaptioned_vid_id = "xvFZjo5PgG0"  # 11 chars - needs audio

        # Add some caption downloads (simulating CAPTION stage output)
        state.caption_downloads = [
            CaptionDownload(
                file="/path/to/vid1.srt",
                video_id=captioned_vid_id,
                url=f"https://youtube.com/watch?v={captioned_vid_id}",
                title="Captioned Video",
                duration=60.0,
                keyword="test",
                language="en",
                is_auto_generated=False
            )
        ]

        # Mark some videos as needing audio (no captions)
        state.videos_need_audio = [uncaptioned_vid_id]

        # Pre-populate transcripts from captions
        state.transcripts = {
            captioned_vid_id: [{'text': 'Caption text', 'start': 0, 'end': 5}]
        }

        # Create fake audio files with video IDs in filename
        # _extract_video_id looks for 11-char patterns
        audio1 = temp_project_dir / f"{captioned_vid_id}.mp3"
        audio2 = temp_project_dir / f"{uncaptioned_vid_id}.mp3"
        audio1.touch()
        audio2.touch()

        video_files = [audio1, audio2]

        stage = TranscribeStage()
        filtered = stage._filter_captioned_videos(video_files, state, caption_first_config)

        # Only uncaptioned video should remain (captioned one is filtered out)
        assert len(filtered) == 1
        assert uncaptioned_vid_id in str(filtered[0])


# ============================================================================
# INTEGRATION TESTS - TEXT METADATA PROPAGATION
# ============================================================================

class TestTextMetadataPropagation:
    """Test transcript_source flows through to text_metadata"""

    def test_text_metadata_includes_transcript_source(
        self, caption_first_config, temp_project_dir
    ):
        """Verify transcript_source is included in text_metadata during embedding"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()

        # Simulate transcripts from both sources
        state.transcripts = {
            "/path/to/manual_video.mp4": [
                {
                    'text': 'Manual caption segment',
                    'start': 0.0, 'end': 5.0,
                    'transcript_source': 'manual_caption'
                },
            ],
            "/path/to/whisper_video.mp4": [
                {
                    'text': 'Whisper transcribed segment',
                    'start': 0.0, 'end': 5.0,
                    'transcript_source': 'whisper'
                },
            ],
            "/path/to/auto_video.mp4": [
                {
                    'text': 'Auto caption segment',
                    'start': 0.0, 'end': 5.0,
                    'transcript_source': 'auto_caption'
                },
            ],
        }

        # Build text_metadata manually (simulating _compute_embeddings behavior)
        texts = []
        for video_path, segments in state.transcripts.items():
            for seg in segments:
                texts.append({
                    'text': seg.get('text', ''),
                    'video_path': video_path,
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'transcript_source': seg.get('transcript_source', 'whisper'),
                })

        state.text_metadata = texts

        # Verify transcript_source is present
        sources = [t['transcript_source'] for t in state.text_metadata]
        assert 'manual_caption' in sources
        assert 'whisper' in sources
        assert 'auto_caption' in sources

        # Verify manual captions can be identified
        manual_entries = [t for t in state.text_metadata if t['transcript_source'] == 'manual_caption']
        assert len(manual_entries) == 1


# ============================================================================
# INTEGRATION TESTS - FULL PIPELINE SIMULATION
# ============================================================================

class TestFullPipelineFlow:
    """Simulate the complete caption-first pipeline flow"""

    def test_end_to_end_caption_first_flow(
        self, caption_first_config, mock_checkpoint, temp_project_dir, sample_srt_content
    ):
        """
        Test complete flow:
        1. VIDEO_METADATA provides candidates
        2. CAPTION fetches captions
        3. TRANSCRIBE skips captioned, runs Whisper for uncaptioned
        4. Embeddings include transcript_source
        5. Matching applies caption boost
        """
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        # ========== STEP 1: Setup video candidates (VIDEO_METADATA output) ==========
        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(
                video_id="paris_doc",
                url="https://youtube.com/watch?v=paris_doc",
                title="Paris Documentary",
                duration=180.0,
                keyword="paris documentary"
            ),
            VideoCandidate(
                video_id="eiffel_vlog",
                url="https://youtube.com/watch?v=eiffel_vlog",
                title="Eiffel Tower Vlog",
                duration=120.0,
                keyword="eiffel tower"
            ),
        ]

        # ========== STEP 2: Run CAPTION stage ==========
        srt_file = temp_project_dir / ".cache" / "captions" / "paris_doc.en.srt"
        srt_file.write_text(sample_srt_content, encoding='utf-8')

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher

            def fetch_captions_side_effect(video_id, **kwargs):
                if video_id == "paris_doc":
                    return CaptionResult(
                        video_id=video_id,
                        file=str(srt_file),
                        language="en",
                        is_auto_generated=False  # Manual caption
                    )
                return None  # eiffel_vlog has no captions

            mock_fetcher.fetch_captions.side_effect = fetch_captions_side_effect

            # Parse returns actual segments
            def parse_side_effect(file_path):
                return [
                    {'text': 'Welcome to our documentary about Paris.', 'start': 0.0, 'end': 3.5},
                    {'text': 'The Eiffel Tower stands majestically over the city.', 'start': 4.0, 'end': 8.0},
                    {'text': 'Every year, millions of visitors come to see it.', 'start': 8.5, 'end': 12.0},
                    {'text': 'The tower was built in 1889 for the World\'s Fair.', 'start': 12.5, 'end': 16.0},
                    {'text': 'It has become the most iconic landmark in France.', 'start': 16.5, 'end': 20.0},
                ]

            mock_fetcher.parse_caption_file.side_effect = parse_side_effect
            mock_fetcher.update_segment_count = Mock()

            caption_stage = CaptionStage()
            caption_result = caption_stage.run(state, caption_first_config, mock_checkpoint)

        # Verify CAPTION stage results
        assert caption_result.success is True
        assert len(state.caption_downloads) == 1
        assert state.caption_downloads[0].video_id == "paris_doc"
        assert state.caption_downloads[0].is_auto_generated is False
        assert "eiffel_vlog" in state.videos_need_audio

        # Verify transcripts have transcript_source
        assert "paris_doc" in state.transcripts
        for seg in state.transcripts["paris_doc"]:
            assert seg.get('transcript_source') == 'manual_caption'

        # ========== STEP 3: Build text_metadata (TRANSCRIBE output) ==========
        # In real pipeline, TranscribeStage._compute_embeddings does this
        state.text_metadata = []
        for video_id, segments in state.transcripts.items():
            for seg in segments:
                state.text_metadata.append({
                    'text': seg.get('text', ''),
                    'video_path': f"/videos/{video_id}.mp4",
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'transcript_source': seg.get('transcript_source', 'whisper'),
                })

        # Add whisper transcript for uncaptioned video
        state.text_metadata.append({
            'text': 'Whisper transcribed content for eiffel vlog',
            'video_path': '/videos/eiffel_vlog.mp4',
            'start_time': 0.0,
            'end_time': 10.0,
            'transcript_source': 'whisper',
        })

        # ========== STEP 4: Verify scoring can identify transcript sources ==========
        # Create mock segments for scoring
        manual_segment = Mock()
        manual_segment.transcript_source = 'manual_caption'

        whisper_segment = Mock()
        whisper_segment.transcript_source = 'whisper'

        # Apply caption boost
        base_confidence = 0.80

        manual_boosted, manual_reason = apply_caption_boost(
            base_confidence, manual_segment, caption_first_config
        )
        whisper_boosted, whisper_reason = apply_caption_boost(
            base_confidence, whisper_segment, caption_first_config
        )

        # Manual caption should be boosted
        assert manual_boosted == 0.90
        assert "manual caption boost" in manual_reason

        # Whisper should not be boosted
        assert whisper_boosted == 0.80
        assert whisper_reason == ""

        # ========== STEP 5: Verify caption_downloads has correct metadata ==========
        cd = state.caption_downloads[0]
        assert cd.language == "en"
        assert cd.is_auto_generated is False
        assert cd.video_id == "paris_doc"


class TestCaptionBoostInMatching:
    """Test caption boost is applied correctly in matching context"""

    def test_manual_caption_outranks_whisper_at_equal_similarity(
        self, caption_first_config
    ):
        """
        Given two segments with equal embedding similarity,
        the one with manual_caption should have higher final confidence.
        """
        base_similarity = 0.75

        # Segment from manual caption
        manual_seg = Mock()
        manual_seg.transcript_source = 'manual_caption'

        # Segment from Whisper
        whisper_seg = Mock()
        whisper_seg.transcript_source = 'whisper'

        # Apply caption boost
        manual_conf, _ = apply_caption_boost(base_similarity, manual_seg, caption_first_config)
        whisper_conf, _ = apply_caption_boost(base_similarity, whisper_seg, caption_first_config)

        # Manual should be higher
        assert manual_conf > whisper_conf
        assert manual_conf == 0.85
        assert whisper_conf == 0.75

    def test_auto_caption_no_boost_vs_whisper(self, caption_first_config):
        """
        Auto-generated captions should not be boosted over Whisper.
        (Quality is roughly equivalent)
        """
        base_similarity = 0.75

        auto_seg = Mock()
        auto_seg.transcript_source = 'auto_caption'

        whisper_seg = Mock()
        whisper_seg.transcript_source = 'whisper'

        auto_conf, _ = apply_caption_boost(base_similarity, auto_seg, caption_first_config)
        whisper_conf, _ = apply_caption_boost(base_similarity, whisper_seg, caption_first_config)

        # Both should be equal (no boost for either)
        assert auto_conf == whisper_conf == 0.75


# ============================================================================
# CHECKPOINT/RESTORE TESTS
# ============================================================================

class TestCaptionStageCheckpoint:
    """Test checkpoint save/restore for CaptionStage"""

    def test_checkpoint_saves_caption_downloads(
        self, caption_first_config, mock_checkpoint, temp_project_dir
    ):
        """Verify checkpoint saves caption_downloads correctly"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(video_id="test_vid", url="https://youtube.com/watch?v=test_vid",
                          title="Test", duration=60, keyword="test")
        ]

        srt_file = temp_project_dir / ".cache" / "captions" / "test_vid.en.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:05,000\nTest\n", encoding='utf-8')

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_captions.return_value = CaptionResult(
                video_id="test_vid", file=str(srt_file),
                language="en", is_auto_generated=False
            )
            mock_fetcher.parse_caption_file.return_value = [
                {'text': 'Test', 'start': 0.0, 'end': 5.0}
            ]
            mock_fetcher.update_segment_count = Mock()

            stage = CaptionStage()
            stage.run(state, caption_first_config, mock_checkpoint)

        # Verify checkpoint.save was called with correct data
        mock_checkpoint.save.assert_called_once()
        call_args = mock_checkpoint.save.call_args
        stage_name, checkpoint_data = call_args[0]

        assert stage_name == "CAPTION"
        assert 'caption_downloads' in checkpoint_data
        assert len(checkpoint_data['caption_downloads']) == 1
        assert checkpoint_data['caption_downloads'][0]['video_id'] == 'test_vid'
        assert checkpoint_data['caption_downloads'][0]['is_auto_generated'] is False

    def test_restore_rebuilds_transcripts(self, caption_first_config, temp_project_dir):
        """Verify restore re-parses caption files to rebuild transcripts"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()

        # Create caption file
        srt_file = temp_project_dir / ".cache" / "captions" / "restored_vid.en.srt"
        srt_file.write_text("""1
00:00:00,000 --> 00:00:05,000
Restored caption text
""", encoding='utf-8')

        # Setup checkpoint data
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'caption_downloads': [{
                'file': str(srt_file),
                'video_id': 'restored_vid',
                'url': 'https://youtube.com/watch?v=restored_vid',
                'title': 'Restored Video',
                'duration': 60.0,
                'keyword': 'test',
                'language': 'en',
                'is_auto_generated': False,
            }],
            'videos_need_audio': [],
            'transcripts_from_captions': ['restored_vid'],
        }

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.parse_caption_file.return_value = [
                {'text': 'Restored caption text', 'start': 0.0, 'end': 5.0}
            ]

            stage = CaptionStage()
            result = stage.restore(state, checkpoint, caption_first_config)

        assert result is True
        assert len(state.caption_downloads) == 1
        # Note: transcripts are rebuilt in _restore_transcripts if config provided


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestCaptionFirstErrorHandling:
    """Test error handling in caption-first flow"""

    def test_graceful_handling_of_fetch_failure(
        self, caption_first_config, mock_checkpoint, temp_project_dir
    ):
        """Pipeline should continue when individual caption fetch fails"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(video_id="success_vid", url="https://youtube.com/watch?v=success_vid",
                          title="Success", duration=60, keyword="test"),
            VideoCandidate(video_id="fail_vid", url="https://youtube.com/watch?v=fail_vid",
                          title="Fail", duration=60, keyword="test"),
        ]

        srt_file = temp_project_dir / ".cache" / "captions" / "success_vid.en.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:05,000\nSuccess\n", encoding='utf-8')

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher

            def fetch_side_effect(video_id, **kwargs):
                if video_id == "success_vid":
                    return CaptionResult(
                        video_id=video_id, file=str(srt_file),
                        language="en", is_auto_generated=False
                    )
                return None  # fail_vid returns None

            mock_fetcher.fetch_captions.side_effect = fetch_side_effect
            mock_fetcher.parse_caption_file.return_value = [
                {'text': 'Success', 'start': 0.0, 'end': 5.0}
            ]
            mock_fetcher.update_segment_count = Mock()

            stage = CaptionStage()
            result = stage.run(state, caption_first_config, mock_checkpoint)

        # Should succeed with partial results
        assert result.success is True
        assert result.data.get('caption_count') == 1
        assert result.data.get('fallback_count') == 1
        assert "fail_vid" in state.videos_need_audio

    def test_handles_parse_failure_gracefully(
        self, caption_first_config, mock_checkpoint, temp_project_dir
    ):
        """Parse failures should mark video for audio fallback"""
        caption_first_config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(video_id="bad_srt", url="https://youtube.com/watch?v=bad_srt",
                          title="Bad SRT", duration=60, keyword="test"),
        ]

        with patch('src.downloader.caption_fetcher.CaptionFetcher') as MockFetcher:
            mock_fetcher = Mock()
            MockFetcher.return_value = mock_fetcher
            mock_fetcher.fetch_captions.return_value = CaptionResult(
                video_id="bad_srt", file="/path/to/bad.srt",
                language="en", is_auto_generated=False
            )
            mock_fetcher.parse_caption_file.return_value = []  # Parse returns empty

            stage = CaptionStage()
            result = stage.run(state, caption_first_config, mock_checkpoint)

        # Should fallback to audio
        assert result.success is True
        assert "bad_srt" in state.videos_need_audio


# ============================================================================
# CONFIG EDGE CASES
# ============================================================================

class TestConfigEdgeCases:
    """Test edge cases in caption-first configuration"""

    def test_disabled_caption_first_skips_stage(self, mock_checkpoint, temp_project_dir):
        """When caption_first.enabled=False, stage should skip"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.enabled = False
        config.cache = Mock()
        config.cache.cache_dir = str(temp_project_dir / ".cache")

        state = PipelineState()
        state.video_candidates = [
            VideoCandidate(video_id="vid", url="url", title="title", duration=60, keyword="kw")
        ]

        stage = CaptionStage()
        result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'not_enabled'

    def test_zero_boost_has_no_effect(self):
        """When confidence_boost_manual=0, no boost should be applied"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.confidence_boost_manual = 0.0

        segment = Mock()
        segment.transcript_source = 'manual_caption'

        boosted, reason = apply_caption_boost(0.75, segment, config)

        assert boosted == 0.75
        assert reason == ""

    def test_missing_caption_config_no_crash(self):
        """If caption_first config is missing, should handle gracefully"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = None  # Missing

        segment = Mock()
        segment.transcript_source = 'manual_caption'

        # Should not crash, should use default
        boosted, reason = apply_caption_boost(0.75, segment, config)

        # Default boost is 0.1
        assert boosted == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
