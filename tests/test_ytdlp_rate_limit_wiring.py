from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.downloader.audio_first import AudioFirstPipeline
from src.downloader.orchestrator import DownloadOrchestrator
from src.downloader.title_filter import SearchResult
from src.downloader.types import MergedSegment
from src.rate_limit.coordinator import (
    GlobalRateLimitCoordinator,
    RateLimitConfig,
    build_coordinator_rate_limit_config,
)


@pytest.fixture(autouse=True)
def reset_coordinator():
    GlobalRateLimitCoordinator.reset_instance()
    yield
    GlobalRateLimitCoordinator.reset_instance()


def _make_audio_pipeline_config() -> SimpleNamespace:
    audio_first = SimpleNamespace(
        enabled=True,
        audio_quality=5,
        fallback_full_video=True,
        buffer_seconds=30.0,
        merge_gap_seconds=15.0,
        min_segment_duration=5.0,
        max_segment_duration=600.0,
    )
    download = SimpleNamespace(
        audio_first=audio_first,
        download_timeouts={'short': 60, 'long': 600},
        max_keyword_len=50,
        max_retries=1,
        retry_delay=1.0,
        quality='1080',
        stall_timeout=0,
        llm_title_filter=None,
        download_resume=None,
        ffmpeg_location='',
        cookies_from_browser='',
        cookies_file=None,
    )
    return SimpleNamespace(download=download)


def _make_downloader_config(tmp_path, slots_per_second: float = 7.0, burst_size: int = 2) -> SimpleNamespace:
    download = SimpleNamespace(
        cookies_from_browser='',
        cookie_rotation=None,
        impersonation=None,
        vpn=None,
        speed_tracking=None,
        circuit_breaker=None,
        batch_retry=None,
        rate_limit=SimpleNamespace(per_tier_isolation=False, share_budget_across_keywords=False),
        rate_limit_budget=None,
        retry_budget=SimpleNamespace(enabled=False, max_attempts=5, max_backoff_time_seconds=300.0),
        llm_title_filter=None,
        audio_first=SimpleNamespace(
            enabled=True,
            audio_quality=5,
            fallback_full_video=True,
            buffer_seconds=30.0,
            merge_gap_seconds=15.0,
            min_segment_duration=5.0,
            max_segment_duration=600.0,
        ),
        max_filename_len=10,
        max_keyword_len=50,
        download_timeouts={'short': 60, 'long': 600},
        download_timeout=60,
        max_retries=1,
        retry_delay=1.0,
        retry_backoff=2.0,
        quality='1080',
        stall_timeout=0,
        metrics_exporter=None,
        adaptive_timeout_enabled=False,
        adaptive_timeout_size_estimates=None,
        download_resume=None,
        extractor_args=None,
        mullvad=None,
        tier_slot_management=None,
    )
    return SimpleNamespace(
        cache_dir=str(tmp_path / "cache"),
        downloaded_videos_dir=str(tmp_path / "downloads"),
        download=download,
        global_cache=SimpleNamespace(enabled=False),
        rate_limit=SimpleNamespace(
            slots_per_second=slots_per_second,
            burst_size=burst_size,
        ),
    )


def test_global_rate_limit_singleton_reconfigures_with_pipeline_values():
    coordinator = GlobalRateLimitCoordinator(
        RateLimitConfig(enabled=True, slots_per_second=2.0, burst_size=5)
    )

    updated = GlobalRateLimitCoordinator(
        build_coordinator_rate_limit_config({'slots_per_second': 0.5, 'burst_size': 3})
    )

    assert updated is coordinator
    assert updated._config.slots_per_second == 0.5
    assert updated._config.burst_size == 3


@patch('src.downloader.core.CookieMethodFallback')
@patch('src.downloader.core.AudioFirstPipeline')
@patch('src.downloader.core.SearchOptimizer')
@patch('src.downloader.core.SpeechScreener')
@patch('src.downloader.core.TitleFilter')
@patch('src.downloader.core.TranscodingManager')
@patch('src.downloader.core.CheckpointManager')
def test_video_downloader_uses_top_level_rate_limit_config(
    mock_checkpoint,
    mock_transcode,
    mock_title,
    mock_speech,
    mock_search,
    mock_audio,
    mock_method_fallback,
    tmp_path,
):
    from src.downloader.core import VideoDownloader

    mock_checkpoint.return_value.load_sources.return_value = []
    mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}

    config = _make_downloader_config(tmp_path, slots_per_second=7.0, burst_size=2)
    downloader = VideoDownloader(config)

    assert downloader.rate_limit_coordinator._config.slots_per_second == 7.0
    assert downloader.rate_limit_coordinator._config.burst_size == 2
    assert downloader.audio_first.rate_limit_coordinator is downloader.rate_limit_coordinator


def test_download_orchestrator_wires_tier_slot_management(tmp_path):
    mock_downloader = MagicMock()
    mock_downloader.download_config = SimpleNamespace(
        max_concurrent=4,
        delay_between_keywords=0,
        tier_slot_management=SimpleNamespace(
            enabled=True,
            max_concurrent_per_tier={'short': 2, 'medium': 1, 'long': 1, 'longer': 1},
            allow_borrowing=False,
            max_total_concurrent=3,
        ),
    )
    mock_downloader.checkpoint = SimpleNamespace(failed_keywords=[])
    mock_downloader.rate_limit_budget = None
    mock_downloader.log_source_diversity_report = Mock()

    orchestrator = DownloadOrchestrator(mock_downloader)

    with patch('src.downloader.orchestrator.DownloadCoordinator') as mock_coordinator_cls, \
         patch.object(orchestrator, '_setup_checkpoint', return_value=[]), \
         patch.object(orchestrator, '_log_download_config'), \
         patch.object(orchestrator, '_init_progress_tracking'), \
         patch.object(orchestrator, '_log_budget_summary'), \
         patch.object(orchestrator, '_process_retry_queue', return_value=([], [])):
        mock_coordinator = MagicMock()
        mock_coordinator.get_status.return_value = {}
        mock_coordinator_cls.return_value = mock_coordinator

        downloaded, failed = orchestrator.download_all([], tmp_path)

    assert downloaded == []
    assert failed == []
    mock_coordinator_cls.assert_called_once_with(
        max_concurrent=4,
        tier_config={'short': 2, 'medium': 1, 'long': 1, 'longer': 1},
        enable_tier_slots=True,
        allow_borrowing=False,
        max_total_concurrent=3,
    )
    assert mock_downloader.download_coordinator is mock_coordinator


def test_audio_first_audio_download_acquires_global_slot(tmp_path):
    config = _make_audio_pipeline_config()

    with patch('src.downloader.audio_first.FormatFallbackHandler', return_value=SimpleNamespace(is_enabled=False)):
        pipeline = AudioFirstPipeline(
            config=config,
            get_tier_value_func=lambda tier, key, default: {'min': 20, 'max': 120, 'per_keyword': 1, 'max_total': 0}.get(key, default),
            search_metadata_func=Mock(return_value=SearchResult(videos=[{
                'id': 'vid1',
                'title': 'Travel',
                'duration': 90,
                'webpage_url': 'https://www.youtube.com/watch?v=vid1',
            }])),
            filter_titles_func=Mock(return_value=[]),
            cleanup_partial_func=Mock(),
            tier_download_counts={},
            lock=threading.Lock(),
        )

    pipeline._get_cookie_args = Mock(return_value=[])
    pipeline.rate_limit_coordinator = Mock()
    pipeline.rate_limit_coordinator.is_enabled.return_value = True

    audio_dir = tmp_path / "travel_s_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def _run_side_effect(*args, **kwargs):
        (audio_dir / "vid1.mp3").write_bytes(b"audio")
        result = Mock(returncode=0)
        result.stderr = ""
        return result

    with patch('subprocess.run', side_effect=_run_side_effect):
        downloads = pipeline.download_audio_for_keyword("travel", tmp_path, "short")

    assert len(downloads) == 1
    pipeline.rate_limit_coordinator.acquire_slot.assert_called_once_with('download', timeout=30.0)
    pipeline.rate_limit_coordinator.release_slot.assert_called_once_with('download')


def test_audio_first_segment_download_acquires_global_slot(tmp_path):
    config = _make_audio_pipeline_config()

    with patch('src.downloader.audio_first.FormatFallbackHandler', return_value=SimpleNamespace(is_enabled=False)):
        pipeline = AudioFirstPipeline(
            config=config,
            get_tier_value_func=lambda tier, key, default: default,
            search_metadata_func=Mock(),
            filter_titles_func=Mock(),
            cleanup_partial_func=Mock(),
            tier_download_counts={},
            lock=threading.Lock(),
        )

    pipeline._get_cookie_args = Mock(return_value=[])
    pipeline._check_existing_segments = Mock(return_value=[None])
    pipeline.rate_limit_coordinator = Mock()
    pipeline.rate_limit_coordinator.is_enabled.return_value = True

    segment_file = tmp_path / "travel_segments" / "vid1_0010.mp4"
    segment_file.parent.mkdir(parents=True, exist_ok=True)
    segment_file.write_bytes(b"segment")

    process = Mock()
    process.returncode = 0

    merged_segments = [
        MergedSegment(
            video_id="vid1",
            video_url="https://www.youtube.com/watch?v=vid1",
            start_time=10.0,
            end_time=20.0,
            original_matches=[],
            keyword="travel",
        )
    ]

    with patch('subprocess.Popen', return_value=process), \
         patch.object(pipeline, '_wait_for_process_with_progress', return_value=("", "", None)), \
         patch('src.downloader.audio_first.segment_utils.rename_segments_with_timing', return_value=[str(segment_file)]):
        downloads = pipeline.download_video_segments(merged_segments, tmp_path)

    assert len(downloads) == 1
    pipeline.rate_limit_coordinator.acquire_slot.assert_called_once_with('download', timeout=30.0)
    pipeline.rate_limit_coordinator.release_slot.assert_called_once_with('download')
