"""Test for AudioDownload obsolete field removal (channel, duration_tier)"""
import pytest
from unittest.mock import Mock
from src.state import PipelineState, AudioDownload
from src.stages.download import DownloadStage


def test_restore_audio_downloads_with_obsolete_fields():
    """Test that obsolete fields (channel, duration_tier) are removed during restore"""
    stage = DownloadStage()
    state = PipelineState()

    # Mock checkpoint with OLD data structure including obsolete fields
    mock_checkpoint = Mock()
    mock_checkpoint.get_stage_data.return_value = {
        'audio_downloads': [
            {
                'audio_file': 'E:\\v\\test.mp3',  # OLD FIELD NAME
                'video_id': 'TEST123',
                'video_url': 'https://youtube.com/watch?v=TEST123',  # OLD FIELD NAME
                'title': 'Test Video',
                'channel': 'Test Channel',  # OBSOLETE FIELD
                'duration': 336.0,
                'keyword': 'test keyword',
                'duration_tier': 'medium',  # OBSOLETE FIELD
                'upload_date': '',
                'license': 'Unknown'
            },
            {
                'audio_file': 'E:\\v\\test2.mp3',
                'video_id': 'TEST456',
                'video_url': 'https://youtube.com/watch?v=TEST456',
                'title': 'Test Video 2',
                'channel': 'Another Channel',  # OBSOLETE FIELD
                'duration': 180.0,
                'keyword': 'another keyword',
                'duration_tier': 'short'  # OBSOLETE FIELD
            }
        ],
        'downloaded_videos': [],
        'failed_keywords': []
    }

    # Test restore
    result = stage.restore(state, mock_checkpoint)

    # Verify restore succeeded
    assert result is True
    assert len(state.downloaded_audio) == 2

    # Verify first audio download
    audio1 = state.downloaded_audio[0]
    assert audio1.file == 'E:\\v\\test.mp3'  # Mapped from 'audio_file'
    assert audio1.url == 'https://youtube.com/watch?v=TEST123'  # Mapped from 'video_url'
    assert audio1.video_id == 'TEST123'
    assert audio1.title == 'Test Video'
    assert audio1.duration == 336.0
    assert audio1.keyword == 'test keyword'

    # Verify obsolete fields are NOT in the AudioDownload object
    audio1_dict = vars(audio1)
    assert 'channel' not in audio1_dict, "Obsolete field 'channel' should be removed"
    assert 'duration_tier' not in audio1_dict, "Obsolete field 'duration_tier' should be removed"
    assert 'upload_date' not in audio1_dict, "Extra field 'upload_date' should not be present"
    assert 'license' not in audio1_dict, "Extra field 'license' should not be present"

    # Verify second audio download
    audio2 = state.downloaded_audio[1]
    assert audio2.file == 'E:\\v\\test2.mp3'
    assert audio2.url == 'https://youtube.com/watch?v=TEST456'
    assert audio2.video_id == 'TEST456'

    # Verify no obsolete fields in second entry either
    audio2_dict = vars(audio2)
    assert 'channel' not in audio2_dict
    assert 'duration_tier' not in audio2_dict


def test_audio_download_dataclass_fields():
    """Verify AudioDownload only has expected fields"""
    audio = AudioDownload(
        file='test.mp3',
        video_id='TEST',
        url='https://youtube.com',
        title='Title',
        duration=100.0,
        keyword='keyword'
    )

    # Get all instance attributes
    attrs = vars(audio)
    expected_fields = {'file', 'video_id', 'url', 'title', 'duration', 'keyword'}

    assert set(attrs.keys()) == expected_fields, \
        f"AudioDownload has unexpected fields: {set(attrs.keys()) - expected_fields}"

    # Verify obsolete fields don't exist
    assert not hasattr(audio, 'channel')
    assert not hasattr(audio, 'duration_tier')
    assert not hasattr(audio, 'audio_file')
    assert not hasattr(audio, 'video_url')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
