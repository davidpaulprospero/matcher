"""
Tests for CaptionBatchCheckpoint in src/caption_fetcher.py (US-005 Sprint 8)

Tests checkpoint creation, serialization, abort handling, and resume functionality.
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.caption_fetcher import (
    CaptionBatchCheckpoint,
    CaptionResult,
    CaptionSegment,
    CaptionFetcher,
    ErrorPatternResult,
    ErrorPatternAbortError,
)


class TestCaptionBatchCheckpoint:
    """Test CaptionBatchCheckpoint dataclass creation and basic operations."""

    @pytest.mark.fast
    def test_checkpoint_creation_empty(self):
        """Test creating an empty checkpoint."""
        checkpoint = CaptionBatchCheckpoint()

        assert checkpoint.results == {}
        assert checkpoint.total_requested == 0
        assert checkpoint.success_count == 0
        assert checkpoint.error_count == 0
        assert checkpoint.aborted is False
        assert checkpoint.abort_reason == ""
        assert checkpoint.abort_pattern_info is None
        assert checkpoint.remaining_video_ids == []
        assert checkpoint.created_at > 0
        assert checkpoint.updated_at > 0

    @pytest.mark.fast
    def test_checkpoint_creation_with_values(self):
        """Test creating a checkpoint with initial values."""
        checkpoint = CaptionBatchCheckpoint(
            total_requested=100,
            remaining_video_ids=['vid1', 'vid2', 'vid3']
        )

        assert checkpoint.total_requested == 100
        assert checkpoint.remaining_video_ids == ['vid1', 'vid2', 'vid3']

    @pytest.mark.fast
    def test_update_with_caption_result(self):
        """Test updating checkpoint with a successful CaptionResult."""
        checkpoint = CaptionBatchCheckpoint(
            remaining_video_ids=['vid1', 'vid2']
        )

        # Create a CaptionResult
        segment = CaptionSegment(
            index=0, start_time=0.0, end_time=2.0,
            text='Hello', source_file='vid1'
        )
        result = CaptionResult(
            video_id='vid1',
            segments=[segment],
            language='en',
            is_auto_generated=False,
            format_source='vtt'
        )

        checkpoint.update('vid1', result)

        assert 'vid1' in checkpoint.results
        assert checkpoint.success_count == 1
        assert checkpoint.error_count == 0
        assert 'vid1' not in checkpoint.remaining_video_ids
        assert checkpoint.results['vid1']['language'] == 'en'
        assert checkpoint.results['vid1']['segment_count'] == 1

    @pytest.mark.fast
    def test_update_with_error_dict(self):
        """Test updating checkpoint with an error result."""
        checkpoint = CaptionBatchCheckpoint(
            remaining_video_ids=['vid1', 'vid2']
        )

        error_result = {
            'video_id': 'vid1',
            'error': True,
            'reason': 'Network timeout',
            'caption_quality': 'low'
        }

        checkpoint.update('vid1', error_result)

        assert 'vid1' in checkpoint.results
        assert checkpoint.success_count == 0
        assert checkpoint.error_count == 1
        assert 'vid1' not in checkpoint.remaining_video_ids

    @pytest.mark.fast
    def test_update_with_unavailable_dict(self):
        """Test updating checkpoint with an unavailable result."""
        checkpoint = CaptionBatchCheckpoint(
            remaining_video_ids=['vid1']
        )

        unavailable_result = {
            'video_id': 'vid1',
            'unavailable': True,
            'reason': 'No captions available',
            'caption_quality': 'low'
        }

        checkpoint.update('vid1', unavailable_result)

        assert checkpoint.error_count == 1  # unavailable counts as error
        assert checkpoint.success_count == 0


class TestCaptionBatchCheckpointAbort:
    """Test checkpoint abort functionality."""

    @pytest.mark.fast
    def test_mark_aborted_basic(self):
        """Test marking a checkpoint as aborted."""
        checkpoint = CaptionBatchCheckpoint()

        checkpoint.mark_aborted(
            reason='Error pattern detected: 403 Forbidden',
            remaining_ids=['vid5', 'vid6', 'vid7']
        )

        assert checkpoint.aborted is True
        assert checkpoint.abort_reason == 'Error pattern detected: 403 Forbidden'
        assert checkpoint.remaining_video_ids == ['vid5', 'vid6', 'vid7']

    @pytest.mark.fast
    def test_mark_aborted_with_pattern_result(self):
        """Test marking aborted with ErrorPatternResult details."""
        checkpoint = CaptionBatchCheckpoint()

        pattern_result = ErrorPatternResult(
            detected=True,
            error_signature='403 Forbidden',
            affected_video_ids=['vid1', 'vid2', 'vid3'],
            sample_size=10,
            ratio=0.3,
            likely_cause='possible geoblocking'
        )

        checkpoint.mark_aborted(
            reason=str(pattern_result),
            pattern_result=pattern_result,
            remaining_ids=['vid4', 'vid5']
        )

        assert checkpoint.aborted is True
        assert checkpoint.abort_pattern_info is not None
        assert checkpoint.abort_pattern_info['error_signature'] == '403 Forbidden'
        assert checkpoint.abort_pattern_info['affected_video_ids'] == ['vid1', 'vid2', 'vid3']
        assert checkpoint.abort_pattern_info['ratio'] == 0.3


class TestCaptionBatchCheckpointSerialization:
    """Test checkpoint serialization and deserialization."""

    @pytest.mark.fast
    def test_to_dict(self):
        """Test converting checkpoint to dict."""
        checkpoint = CaptionBatchCheckpoint(
            total_requested=100,
            success_count=50,
            error_count=5,
            remaining_video_ids=['vid1', 'vid2']
        )
        checkpoint.results = {'vid3': {'video_id': 'vid3', 'segments': []}}

        data = checkpoint.to_dict()

        assert data['total_requested'] == 100
        assert data['success_count'] == 50
        assert data['error_count'] == 5
        assert data['results'] == {'vid3': {'video_id': 'vid3', 'segments': []}}
        assert data['remaining_video_ids'] == ['vid1', 'vid2']

    @pytest.mark.fast
    def test_from_dict(self):
        """Test creating checkpoint from dict."""
        data = {
            'results': {'vid1': {'video_id': 'vid1', 'segments': []}},
            'total_requested': 200,
            'success_count': 100,
            'error_count': 10,
            'aborted': True,
            'abort_reason': 'Test abort',
            'abort_pattern_info': {'error_signature': '429'},
            'remaining_video_ids': ['vid2', 'vid3']
        }

        checkpoint = CaptionBatchCheckpoint.from_dict(data)

        assert checkpoint.total_requested == 200
        assert checkpoint.success_count == 100
        assert checkpoint.error_count == 10
        assert checkpoint.aborted is True
        assert checkpoint.abort_reason == 'Test abort'
        assert checkpoint.abort_pattern_info['error_signature'] == '429'
        assert 'vid1' in checkpoint.results


class TestCaptionBatchCheckpointPersistence:
    """Test checkpoint file save/load operations."""

    @pytest.mark.fast
    def test_save_and_load(self, tmp_path):
        """Test saving and loading a checkpoint."""
        checkpoint_path = tmp_path / '.cache' / 'caption_checkpoint.json'

        # Create and save checkpoint
        checkpoint = CaptionBatchCheckpoint(
            total_requested=50,
            remaining_video_ids=['vid1', 'vid2', 'vid3']
        )

        # Add some results
        segment = CaptionSegment(
            index=0, start_time=0.0, end_time=2.0,
            text='Test', source_file='vid4'
        )
        result = CaptionResult(
            video_id='vid4',
            segments=[segment],
            language='en',
            is_auto_generated=False,
            format_source='vtt'
        )
        checkpoint.update('vid4', result)

        # Save
        assert checkpoint.save(checkpoint_path) is True
        assert checkpoint_path.exists()

        # Load
        loaded = CaptionBatchCheckpoint.load(checkpoint_path)

        assert loaded is not None
        assert loaded.total_requested == 50
        assert loaded.success_count == 1
        assert 'vid4' in loaded.results
        assert loaded.results['vid4']['language'] == 'en'

    @pytest.mark.fast
    def test_load_nonexistent(self, tmp_path):
        """Test loading from non-existent file returns None."""
        checkpoint_path = tmp_path / 'nonexistent.json'

        loaded = CaptionBatchCheckpoint.load(checkpoint_path)

        assert loaded is None

    @pytest.mark.fast
    def test_load_corrupt_json(self, tmp_path):
        """Test loading corrupt JSON returns None gracefully."""
        checkpoint_path = tmp_path / 'corrupt.json'
        checkpoint_path.write_text('{ invalid json }')

        loaded = CaptionBatchCheckpoint.load(checkpoint_path)

        assert loaded is None

    @pytest.mark.fast
    def test_get_checkpoint_path(self, tmp_path):
        """Test get_checkpoint_path returns correct path."""
        path = CaptionBatchCheckpoint.get_checkpoint_path(tmp_path)

        assert path == tmp_path / '.cache' / 'caption_checkpoint.json'


class TestCaptionBatchCheckpointHelpers:
    """Test checkpoint helper methods."""

    @pytest.mark.fast
    def test_get_remaining_ids(self):
        """Test getting remaining IDs from checkpoint."""
        checkpoint = CaptionBatchCheckpoint()
        checkpoint.results = {
            'vid1': {'video_id': 'vid1'},
            'vid2': {'video_id': 'vid2'}
        }

        all_ids = ['vid1', 'vid2', 'vid3', 'vid4', 'vid5']
        remaining = checkpoint.get_remaining_ids(all_ids)

        assert remaining == ['vid3', 'vid4', 'vid5']

    @pytest.mark.fast
    def test_has_result(self):
        """Test checking if video has result."""
        checkpoint = CaptionBatchCheckpoint()
        checkpoint.results = {'vid1': {'video_id': 'vid1'}}

        assert checkpoint.has_result('vid1') is True
        assert checkpoint.has_result('vid2') is False

    @pytest.mark.fast
    def test_get_successful_results(self):
        """Test getting only successful results."""
        checkpoint = CaptionBatchCheckpoint()
        checkpoint.results = {
            'vid1': {'video_id': 'vid1', 'segments': []},
            'vid2': {'video_id': 'vid2', 'error': True, 'reason': 'failed'},
            'vid3': {'video_id': 'vid3', 'unavailable': True},
            'vid4': {'video_id': 'vid4', 'segments': []}
        }

        successful = checkpoint.get_successful_results()

        assert 'vid1' in successful
        assert 'vid4' in successful
        assert 'vid2' not in successful
        assert 'vid3' not in successful


class TestBatchFetchWithCheckpoint:
    """Test batch fetch integration with checkpoint."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock fetcher that simulates batch operations."""
        config = Mock()
        config.download.caption_first.max_parallel_fetches = 2
        config.download.caption_first.abort_on_error_pattern = 'warn'
        config.download.caption_first.error_pattern_threshold = 0.3
        config.download.caption_first.error_pattern_sample_size = 10
        config.download.caption_first.prioritize_by_channel = False
        config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=config)
        return fetcher

    @pytest.mark.fast
    def test_batch_fetch_updates_checkpoint(self, mock_fetcher):
        """Test that batch fetch updates checkpoint as results arrive."""
        checkpoint = CaptionBatchCheckpoint(
            total_requested=3,
            remaining_video_ids=['vid1', 'vid2', 'vid3']
        )

        # Mock the fetch method to return controlled results
        def mock_fetch(*args, **kwargs):
            return CaptionResult(
                video_id=args[0],
                segments=[CaptionSegment(0, 0.0, 1.0, 'Test', args[0])],
                language='en',
                is_auto_generated=False,
                format_source='vtt'
            )

        with patch.object(
            mock_fetcher, 'fetch_captions_auto_language_with_retry',
            side_effect=mock_fetch
        ):
            results = mock_fetcher.fetch_captions_batch(
                video_ids=['vid1', 'vid2', 'vid3'],
                batch_checkpoint=checkpoint,
                checkpoint_save_interval=2
            )

        # All videos should be processed
        assert len(results) == 3
        assert len(checkpoint.results) == 3
        assert checkpoint.success_count == 3


class TestAbortSavesCheckpoint:
    """Test that ErrorPatternAbortError properly saves checkpoint."""

    @pytest.mark.fast
    def test_abort_marks_checkpoint(self):
        """Test abort at video 50/100 saves 50 results and marks remaining."""
        # Create a checkpoint with partial results
        checkpoint = CaptionBatchCheckpoint(
            total_requested=100,
            remaining_video_ids=[f'vid{i}' for i in range(100)]
        )

        # Simulate 50 successful fetches
        for i in range(50):
            result = {
                'video_id': f'vid{i}',
                'segments': [],
                'language': 'en',
                'is_auto_generated': False,
                'segment_count': 10,
                'caption_quality': 'high'
            }
            checkpoint.update(f'vid{i}', result)

        # Simulate abort with pattern
        pattern_result = ErrorPatternResult(
            detected=True,
            error_signature='403 Forbidden',
            affected_video_ids=[f'vid{i}' for i in range(50, 65)],
            sample_size=15,
            ratio=1.0,
            likely_cause='geoblocking'
        )

        remaining_ids = [f'vid{i}' for i in range(50, 100)]
        checkpoint.mark_aborted(
            reason=str(pattern_result),
            pattern_result=pattern_result,
            remaining_ids=remaining_ids
        )

        # Verify checkpoint state
        assert checkpoint.success_count == 50
        assert checkpoint.aborted is True
        assert len(checkpoint.remaining_video_ids) == 50
        assert 'vid0' in checkpoint.results
        assert 'vid49' in checkpoint.results
        assert 'vid50' not in checkpoint.results


class TestResumeFromCheckpoint:
    """Test resume_from_checkpoint functionality."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock fetcher."""
        config = Mock()
        config.download.caption_first.max_parallel_fetches = 2
        config.download.caption_first.abort_on_error_pattern = 'warn'
        config.download.caption_first.error_pattern_threshold = 0.3
        config.download.caption_first.error_pattern_sample_size = 10
        config.download.caption_first.prioritize_by_channel = False
        config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=config)
        return fetcher

    @pytest.mark.fast
    def test_resume_skips_already_fetched(self, mock_fetcher):
        """Test that resume skips videos already in checkpoint."""
        # Create checkpoint with some results
        checkpoint = CaptionBatchCheckpoint(
            total_requested=5,
            aborted=True,
            abort_reason='Test abort'
        )
        # Add results for vid1 and vid2
        checkpoint.results = {
            'vid1': {'video_id': 'vid1', 'segments': [], 'language': 'en'},
            'vid2': {'video_id': 'vid2', 'segments': [], 'language': 'en'}
        }
        checkpoint.remaining_video_ids = ['vid3', 'vid4', 'vid5']

        # Track which videos were actually fetched
        fetched_videos = []

        def mock_fetch(*args, **kwargs):
            vid = args[0]
            fetched_videos.append(vid)
            return CaptionResult(
                video_id=vid,
                segments=[CaptionSegment(0, 0.0, 1.0, 'Test', vid)],
                language='en',
                is_auto_generated=False,
                format_source='vtt'
            )

        with patch.object(
            mock_fetcher, 'fetch_captions_auto_language_with_retry',
            side_effect=mock_fetch
        ):
            results = mock_fetcher.resume_from_checkpoint(
                checkpoint=checkpoint,
                video_ids=['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
            )

        # Should have results for all 5
        assert len(results) == 5

        # But only vid3, vid4, vid5 should have been fetched
        assert 'vid1' not in fetched_videos
        assert 'vid2' not in fetched_videos
        assert 'vid3' in fetched_videos
        assert 'vid4' in fetched_videos
        assert 'vid5' in fetched_videos

        # Checkpoint should no longer be marked aborted
        assert checkpoint.aborted is False

    @pytest.mark.fast
    def test_resume_all_already_fetched(self, mock_fetcher):
        """Test resume when all videos already in checkpoint."""
        checkpoint = CaptionBatchCheckpoint(
            total_requested=3
        )
        checkpoint.results = {
            'vid1': {'video_id': 'vid1'},
            'vid2': {'video_id': 'vid2'},
            'vid3': {'video_id': 'vid3'}
        }

        results = mock_fetcher.resume_from_checkpoint(
            checkpoint=checkpoint,
            video_ids=['vid1', 'vid2', 'vid3']
        )

        # Should return cached results without fetching
        assert len(results) == 3
        assert 'vid1' in results


class TestEndToEndAbortResume:
    """End-to-end test: abort at 50/100, save, resume, fetch remaining 50."""

    @pytest.mark.fast
    def test_abort_resume_scenario(self, tmp_path):
        """Test complete abort and resume scenario."""
        checkpoint_path = tmp_path / '.cache' / 'caption_checkpoint.json'

        # Phase 1: Initial batch that gets aborted
        checkpoint = CaptionBatchCheckpoint(
            total_requested=100,
            remaining_video_ids=[f'vid{i}' for i in range(100)]
        )

        # Simulate 50 successful fetches before abort
        for i in range(50):
            result = CaptionResult(
                video_id=f'vid{i}',
                segments=[CaptionSegment(0, 0.0, 1.0, f'Text {i}', f'vid{i}')],
                language='en',
                is_auto_generated=False,
                format_source='vtt'
            )
            checkpoint.update(f'vid{i}', result)

        # Mark aborted
        remaining = [f'vid{i}' for i in range(50, 100)]
        checkpoint.mark_aborted(
            reason='Network errors',
            remaining_ids=remaining
        )

        # Save checkpoint
        assert checkpoint.save(checkpoint_path) is True
        assert checkpoint_path.exists()

        # Phase 2: Load and resume
        loaded = CaptionBatchCheckpoint.load(checkpoint_path)
        assert loaded is not None
        assert loaded.success_count == 50
        assert loaded.aborted is True
        assert len(loaded.remaining_video_ids) == 50

        # Simulate resuming and fetching remaining 50
        for i in range(50, 100):
            result = CaptionResult(
                video_id=f'vid{i}',
                segments=[CaptionSegment(0, 0.0, 1.0, f'Text {i}', f'vid{i}')],
                language='en',
                is_auto_generated=False,
                format_source='vtt'
            )
            loaded.update(f'vid{i}', result)

        # Verify final state
        assert loaded.success_count == 100  # 50 original + 50 new
        assert len(loaded.results) == 100

        # Verify all videos present
        for i in range(100):
            assert f'vid{i}' in loaded.results
