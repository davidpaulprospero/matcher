"""
Comprehensive test suite for src/downloader/checkpoint.py - CheckpointManager class.

Tests coverage for:
- DownloadCheckpoint dataclass (to_dict, from_dict)
- CheckpointManager initialization
- Duration tier loading and configuration
- Tier value retrieval (dict and dataclass formats)
- Checkpoint loading, saving, and clearing
- Sources.json loading and saving

Created: January 10, 2026
Session: 13 Phase 3
Target Coverage: 55% → 75%
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.checkpoint import DownloadCheckpoint, CheckpointManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Create mock Config object with duration_tiers"""
    config = Mock()

    # Duration tiers config (dataclass format)
    config.duration_tiers = Mock()
    config.duration_tiers.short = Mock()
    config.duration_tiers.short.min_seconds = 20
    config.duration_tiers.short.max_seconds = 120
    config.duration_tiers.short.videos_per_keyword = 8
    config.duration_tiers.short.max_total = 0

    config.duration_tiers.medium = Mock()
    config.duration_tiers.medium.min_seconds = 120
    config.duration_tiers.medium.max_seconds = 600
    config.duration_tiers.medium.videos_per_keyword = 8
    config.duration_tiers.medium.max_total = 0

    config.duration_tiers.long = Mock()
    config.duration_tiers.long.min_seconds = 600
    config.duration_tiers.long.max_seconds = 1500
    config.duration_tiers.long.videos_per_keyword = 5
    config.duration_tiers.long.max_total = 0

    config.duration_tiers.longer = Mock()
    config.duration_tiers.longer.min_seconds = 1500
    config.duration_tiers.longer.max_seconds = 3000
    config.duration_tiers.longer.videos_per_keyword = 1
    config.duration_tiers.longer.max_total = 1

    return config


@pytest.fixture
def checkpoint_file(temp_dir):
    """Create checkpoint file path"""
    return temp_dir / "checkpoint.json"


@pytest.fixture
def sources_file(temp_dir):
    """Create sources file path"""
    return temp_dir / "sources.json"


@pytest.fixture
def manager(mock_config, checkpoint_file, sources_file):
    """Create CheckpointManager instance"""
    return CheckpointManager(
        config=mock_config,
        checkpoint_file=checkpoint_file,
        sources_file=sources_file
    )


# ============================================================================
# Test DownloadCheckpoint Dataclass
# ============================================================================

class TestDownloadCheckpoint:
    """Test DownloadCheckpoint dataclass"""

    @pytest.mark.fast
    def test_checkpoint_creation(self):
        """Test basic checkpoint creation"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=["travel", "vacation"],
            completed_videos=["vid1", "vid2"],
            failed_keywords=["failed"],
            current_keyword="beach",
            current_tier="short",
            timestamp="2026-01-10T10:00:00"
        )

        assert checkpoint.completed_keywords == ["travel", "vacation"]
        assert checkpoint.completed_videos == ["vid1", "vid2"]
        assert checkpoint.failed_keywords == ["failed"]
        assert checkpoint.current_keyword == "beach"
        assert checkpoint.current_tier == "short"
        assert checkpoint.timestamp == "2026-01-10T10:00:00"

    @pytest.mark.fast
    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization to dict"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=["travel"],
            completed_videos=["vid1"],
            failed_keywords=[],
            current_keyword="beach",
            current_tier="medium",
            timestamp="2026-01-10T10:00:00"
        )

        data = checkpoint.to_dict()

        assert isinstance(data, dict)
        assert data['completed_keywords'] == ["travel"]
        assert data['completed_videos'] == ["vid1"]
        assert data['failed_keywords'] == []
        assert data['current_keyword'] == "beach"
        assert data['current_tier'] == "medium"
        assert data['timestamp'] == "2026-01-10T10:00:00"

    @pytest.mark.fast
    def test_checkpoint_from_dict(self):
        """Test checkpoint deserialization from dict"""
        data = {
            'completed_keywords': ["travel", "vacation"],
            'completed_videos': ["vid1", "vid2"],
            'failed_keywords': ["bad"],
            'current_keyword': "sunset",
            'current_tier': "long",
            'timestamp': "2026-01-10T12:00:00"
        }

        checkpoint = DownloadCheckpoint.from_dict(data)

        assert isinstance(checkpoint, DownloadCheckpoint)
        assert checkpoint.completed_keywords == ["travel", "vacation"]
        assert checkpoint.completed_videos == ["vid1", "vid2"]
        assert checkpoint.failed_keywords == ["bad"]
        assert checkpoint.current_keyword == "sunset"
        assert checkpoint.current_tier == "long"
        assert checkpoint.timestamp == "2026-01-10T12:00:00"

    @pytest.mark.fast
    def test_checkpoint_schema_version_default(self):
        """Test that schema_version defaults to 1 (US-58-002)"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=["travel"],
            completed_videos=["vid1"],
            failed_keywords=[],
            current_keyword="beach",
            current_tier="short",
            timestamp="2026-02-05T10:00:00"
        )
        assert checkpoint.schema_version == 1

    @pytest.mark.fast
    def test_checkpoint_to_dict_includes_schema_version(self):
        """Test that to_dict includes schema_version in output (US-58-002)"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2026-02-05T10:00:00"
        )
        data = checkpoint.to_dict()
        assert 'schema_version' in data
        assert data['schema_version'] == 1

    @pytest.mark.fast
    def test_checkpoint_from_dict_missing_schema_version_defaults_to_1(self):
        """Test that loading checkpoint without schema_version defaults to 1 (US-58-002)"""
        data = {
            'completed_keywords': ["travel"],
            'completed_videos': ["vid1"],
            'failed_keywords': [],
            'current_keyword': "beach",
            'current_tier': "short",
            'timestamp': "2026-02-05T10:00:00"
        }
        # No schema_version key in data
        assert 'schema_version' not in data

        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.schema_version == 1

    @pytest.mark.fast
    def test_checkpoint_from_dict_mismatched_schema_version_warns(self, caplog):
        """Test that loading checkpoint with mismatched schema_version logs warning (US-58-002)"""
        import logging
        caplog.set_level(logging.WARNING)

        data = {
            'completed_keywords': [],
            'completed_videos': [],
            'failed_keywords': [],
            'current_keyword': None,
            'current_tier': None,
            'timestamp': "2026-02-05T10:00:00",
            'schema_version': 99  # Future/mismatched version
        }

        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.schema_version == 99
        assert "schema_version mismatch" in caplog.text
        assert "loaded v99" in caplog.text
        assert "current v1" in caplog.text

    @pytest.mark.fast
    def test_checkpoint_from_dict_matching_schema_version_no_warning(self, caplog):
        """Test that loading checkpoint with matching schema_version does not warn (US-58-002)"""
        import logging
        caplog.set_level(logging.WARNING)

        data = {
            'completed_keywords': [],
            'completed_videos': [],
            'failed_keywords': [],
            'current_keyword': None,
            'current_tier': None,
            'timestamp': "2026-02-05T10:00:00",
            'schema_version': 1  # Matches current
        }

        checkpoint = DownloadCheckpoint.from_dict(data)
        assert checkpoint.schema_version == 1
        assert "schema_version mismatch" not in caplog.text

    @pytest.mark.fast
    def test_checkpoint_with_none_values(self):
        """Test checkpoint with None for optional fields"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2026-01-10T10:00:00"
        )

        assert checkpoint.current_keyword is None
        assert checkpoint.current_tier is None

        # Test serialization preserves None
        data = checkpoint.to_dict()
        restored = DownloadCheckpoint.from_dict(data)
        assert restored.current_keyword is None
        assert restored.current_tier is None


# ============================================================================
# Test CheckpointManager Initialization
# ============================================================================

class TestCheckpointManagerInit:
    """Test CheckpointManager initialization"""

    @pytest.mark.fast
    def test_init_basic(self, mock_config, checkpoint_file, sources_file):
        """Test basic initialization"""
        manager = CheckpointManager(
            config=mock_config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        assert manager.config is mock_config
        assert manager.checkpoint_file == checkpoint_file
        assert manager.sources_file == sources_file
        assert isinstance(manager.duration_tiers, dict)

    @pytest.mark.fast
    def test_init_loads_tiers(self, manager):
        """Test that initialization loads duration tiers"""
        assert 'short' in manager.duration_tiers
        assert 'medium' in manager.duration_tiers
        assert 'long' in manager.duration_tiers
        assert 'longer' in manager.duration_tiers

    @pytest.mark.fast
    def test_init_with_custom_paths(self, mock_config):
        """Test initialization with custom paths"""
        custom_checkpoint = Path("/custom/checkpoint.json")
        custom_sources = Path("/custom/sources.json")

        manager = CheckpointManager(
            config=mock_config,
            checkpoint_file=custom_checkpoint,
            sources_file=custom_sources
        )

        assert manager.checkpoint_file == custom_checkpoint
        assert manager.sources_file == custom_sources


# ============================================================================
# Test Duration Tier Loading
# ============================================================================

class TestDurationTierLoading:
    """Test duration tier configuration loading"""

    @pytest.mark.fast
    def test_load_duration_tiers_from_config(self, manager):
        """Test loading tiers from config"""
        tiers = manager.duration_tiers

        # Check short tier
        assert tiers['short']['min'] == 20
        assert tiers['short']['max'] == 120
        assert tiers['short']['per_keyword'] == 8
        assert tiers['short']['max_total'] == 0

        # Check medium tier
        assert tiers['medium']['min'] == 120
        assert tiers['medium']['max'] == 600

        # Check long tier
        assert tiers['long']['min'] == 600
        assert tiers['long']['max'] == 1500

        # Check longer tier
        assert tiers['longer']['min'] == 1500
        assert tiers['longer']['max'] == 3000

    @pytest.mark.fast
    def test_load_duration_tiers_no_config(self, checkpoint_file, sources_file):
        """Test default tiers when config missing"""
        config = Mock()
        config.duration_tiers = None  # No tiers config

        manager = CheckpointManager(
            config=config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        tiers = manager.duration_tiers

        # Should use defaults
        assert tiers['short']['min'] == 20
        assert tiers['short']['max'] == 120
        assert tiers['short']['per_keyword'] == 8
        assert tiers['medium']['per_keyword'] == 8
        assert tiers['long']['per_keyword'] == 5
        assert tiers['longer']['per_keyword'] == 1

    @pytest.mark.fast
    def test_load_duration_tiers_missing_hasattr(self, checkpoint_file, sources_file):
        """Test when config doesn't have duration_tiers attribute"""
        config = Mock(spec=[])  # No duration_tiers attribute

        manager = CheckpointManager(
            config=config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        # Should use defaults
        assert 'short' in manager.duration_tiers
        assert manager.duration_tiers['short']['min'] == 20

    @pytest.mark.fast
    def test_load_duration_tiers_empty_dict(self, checkpoint_file, sources_file):
        """Test when duration_tiers config is empty"""
        config = Mock()
        config.duration_tiers = Mock()
        # No tier attributes

        manager = CheckpointManager(
            config=config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        # Should use defaults (tiers dict will be empty, so fallback)
        assert 'short' in manager.duration_tiers


# ============================================================================
# Test Tier Value Retrieval
# ============================================================================

class TestTierValueRetrieval:
    """Test getting tier config values"""

    @pytest.mark.fast
    def test_get_tier_value_dict_format(self, manager):
        """Test retrieving values from dict format"""
        # Manager has dict-formatted tiers
        assert manager.get_tier_value('short', 'min') == 20
        assert manager.get_tier_value('short', 'max') == 120
        assert manager.get_tier_value('short', 'per_keyword') == 8
        assert manager.get_tier_value('short', 'max_total') == 0

    @pytest.mark.fast
    def test_get_tier_value_with_default(self, manager):
        """Test default value when key missing"""
        result = manager.get_tier_value('short', 'nonexistent', default=999)
        assert result == 999

    @pytest.mark.fast
    def test_get_tier_value_missing_tier(self, manager):
        """Test retrieving from non-existent tier"""
        result = manager.get_tier_value('invalid_tier', 'min', default=100)
        assert result == 100

    @pytest.mark.fast
    def test_get_tier_value_dataclass_format(self, mock_config, checkpoint_file, sources_file):
        """Test retrieving values from dataclass format (not dict)"""
        # Manually set a dataclass-formatted tier
        class TierConfig:
            def __init__(self):
                self.min_seconds = 30
                self.max_seconds = 90
                self.videos_per_keyword = 10
                self.max_total = 5

        manager = CheckpointManager(mock_config, checkpoint_file, sources_file)
        manager.duration_tiers['test_tier'] = TierConfig()

        # Test different key mappings
        assert manager.get_tier_value('test_tier', 'min') == 30
        assert manager.get_tier_value('test_tier', 'max') == 90
        assert manager.get_tier_value('test_tier', 'per_keyword') == 10
        assert manager.get_tier_value('test_tier', 'max_total') == 5

    @pytest.mark.fast
    def test_get_tier_value_dataclass_fallback_names(self, mock_config, checkpoint_file, sources_file):
        """Test fallback attribute names for dataclass"""
        # Create tier with fallback attribute names (e.g., 'min' instead of 'min_seconds')
        class TierConfig:
            def __init__(self):
                self.min = 40  # Fallback name
                self.max = 80
                self.per_keyword = 12

        manager = CheckpointManager(mock_config, checkpoint_file, sources_file)
        manager.duration_tiers['test_tier'] = TierConfig()

        assert manager.get_tier_value('test_tier', 'min') == 40
        assert manager.get_tier_value('test_tier', 'max') == 80
        assert manager.get_tier_value('test_tier', 'per_keyword') == 12

    @pytest.mark.fast
    def test_get_tier_value_none_value(self, manager):
        """Test handling None values"""
        # Set a tier value to None
        manager.duration_tiers['test'] = {'min': None}

        result = manager.get_tier_value('test', 'min', default=50)
        assert result == 50  # Should use default when value is None


# ============================================================================
# Test Checkpoint Loading
# ============================================================================

class TestCheckpointLoading:
    """Test loading checkpoints from file"""

    @pytest.mark.fast
    def test_load_checkpoint_success(self, manager, checkpoint_file):
        """Test successfully loading checkpoint"""
        # Create checkpoint file
        checkpoint_data = {
            'completed_keywords': ["travel", "vacation"],
            'completed_videos': ["vid1", "vid2"],
            'failed_keywords': [],
            'current_keyword': "beach",
            'current_tier': "medium",
            'timestamp': "2026-01-10T10:00:00"
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        checkpoint = manager.load_checkpoint()

        assert isinstance(checkpoint, DownloadCheckpoint)
        assert checkpoint.completed_keywords == ["travel", "vacation"]
        assert checkpoint.completed_videos == ["vid1", "vid2"]
        assert checkpoint.current_keyword == "beach"
        assert checkpoint.current_tier == "medium"

    @pytest.mark.fast
    def test_load_checkpoint_not_found(self, manager, checkpoint_file):
        """Test loading when checkpoint doesn't exist"""
        # No file created
        checkpoint = manager.load_checkpoint()
        assert checkpoint is None

    @pytest.mark.fast
    def test_load_checkpoint_invalid_json(self, manager, checkpoint_file):
        """Test loading when JSON is corrupted"""
        checkpoint_file.write_text("{ invalid json }")

        checkpoint = manager.load_checkpoint()
        assert checkpoint is None  # Should return None on error

    @pytest.mark.fast
    def test_load_checkpoint_missing_fields(self, manager, checkpoint_file, caplog):
        """Test loading with missing required fields"""
        import logging
        caplog.set_level(logging.WARNING)

        # Missing 'failed_keywords' field
        incomplete_data = {
            'completed_keywords': ["travel"],
            'completed_videos': []
        }
        checkpoint_file.write_text(json.dumps(incomplete_data))

        checkpoint = manager.load_checkpoint()
        assert checkpoint is None
        assert "Could not load checkpoint" in caplog.text


# ============================================================================
# Test Checkpoint Saving
# ============================================================================

class TestCheckpointSaving:
    """Test saving checkpoints to file"""

    @pytest.mark.fast
    def test_save_checkpoint_basic(self, manager, checkpoint_file):
        """Test basic checkpoint saving"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=["travel"],
            completed_videos=["vid1"],
            failed_keywords=[],
            current_keyword="beach",
            current_tier="short",
            timestamp=""  # Will be set by save
        )

        manager.save_checkpoint(checkpoint)

        # Verify file was created
        assert checkpoint_file.exists()

        # Verify contents
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        assert data['completed_keywords'] == ["travel"]
        assert data['current_keyword'] == "beach"
        assert data['current_tier'] == "short"

    @pytest.mark.fast
    def test_save_checkpoint_creates_directory(self, manager, temp_dir):
        """Test that save creates parent directory if missing"""
        nested_checkpoint = temp_dir / "nested" / "dir" / "checkpoint.json"
        manager.checkpoint_file = nested_checkpoint

        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=""
        )

        manager.save_checkpoint(checkpoint)

        # Verify directory and file created
        assert nested_checkpoint.exists()
        assert nested_checkpoint.parent.exists()

    @pytest.mark.fast
    def test_save_checkpoint_updates_timestamp(self, manager, checkpoint_file):
        """Test that save updates timestamp"""
        checkpoint = DownloadCheckpoint(
            completed_keywords=["travel"],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="old_timestamp"
        )

        manager.save_checkpoint(checkpoint)

        # Load and verify timestamp was updated
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        assert data['timestamp'] != "old_timestamp"
        # Verify it's a valid ISO format
        datetime.fromisoformat(data['timestamp'])

    @pytest.mark.fast
    def test_save_checkpoint_overwrites_existing(self, manager, checkpoint_file):
        """Test that save overwrites existing checkpoint"""
        # Create initial checkpoint
        checkpoint1 = DownloadCheckpoint(
            completed_keywords=["travel"],
            completed_videos=["vid1"],
            failed_keywords=[],
            current_keyword="beach",
            current_tier="short",
            timestamp=""
        )
        manager.save_checkpoint(checkpoint1)

        # Save new checkpoint
        checkpoint2 = DownloadCheckpoint(
            completed_keywords=["travel", "vacation"],
            completed_videos=["vid1", "vid2"],
            failed_keywords=[],
            current_keyword="sunset",
            current_tier="medium",
            timestamp=""
        )
        manager.save_checkpoint(checkpoint2)

        # Verify latest data
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        assert data['completed_keywords'] == ["travel", "vacation"]
        assert data['current_keyword'] == "sunset"


# ============================================================================
# Test Checkpoint Clearing
# ============================================================================

class TestCheckpointClearing:
    """Test clearing checkpoints"""

    @pytest.mark.fast
    def test_clear_checkpoint_existing(self, manager, checkpoint_file):
        """Test clearing existing checkpoint"""
        # Create checkpoint
        checkpoint_data = {'completed_keywords': ["travel"]}
        checkpoint_file.write_text(json.dumps(checkpoint_data))
        assert checkpoint_file.exists()

        # Clear it
        manager.clear_checkpoint()

        # Verify deleted
        assert not checkpoint_file.exists()

    @pytest.mark.fast
    def test_clear_checkpoint_not_existing(self, manager, checkpoint_file):
        """Test clearing when checkpoint doesn't exist (no error)"""
        # No checkpoint file
        assert not checkpoint_file.exists()

        # Should not raise error
        manager.clear_checkpoint()

        assert not checkpoint_file.exists()


# ============================================================================
# Test Sources Loading
# ============================================================================

class TestSourcesLoading:
    """Test loading sources.json"""

    @pytest.mark.fast
    def test_load_sources_success(self, manager, sources_file):
        """Test successfully loading sources"""
        sources_data = [
            {
                'file': '/videos/vid1.mp4',
                'url': 'https://youtube.com/watch?v=vid1',
                'title': 'Travel Video',
                'duration': 120.5,
                'duration_tier': 'medium',
                'keyword': 'travel'
            },
            {
                'file': '/videos/vid2.mp4',
                'url': 'https://youtube.com/watch?v=vid2',
                'title': 'Beach Video',
                'duration': 60.0,
                'duration_tier': 'short',
                'keyword': 'beach'
            }
        ]
        sources_file.write_text(json.dumps(sources_data))

        sources = manager.load_sources()

        assert len(sources) == 2
        assert sources[0].file == '/videos/vid1.mp4'
        assert sources[0].title == 'Travel Video'
        assert sources[1].file == '/videos/vid2.mp4'
        assert sources[1].title == 'Beach Video'

    @pytest.mark.fast
    def test_load_sources_not_found(self, manager, sources_file):
        """Test loading when sources.json doesn't exist"""
        # No file created
        sources = manager.load_sources()
        assert sources == []

    @pytest.mark.fast
    def test_load_sources_invalid_json(self, manager, sources_file):
        """Test loading when JSON is corrupted"""
        sources_file.write_text("{ invalid json }")

        sources = manager.load_sources()
        assert sources == []  # Should return empty list on error

    @pytest.mark.fast
    def test_load_sources_empty_list(self, manager, sources_file):
        """Test loading empty sources list"""
        sources_file.write_text(json.dumps([]))

        sources = manager.load_sources()
        assert sources == []

    @pytest.mark.fast
    def test_load_sources_logs_count(self, manager, sources_file, caplog):
        """Test that load logs the count of loaded sources"""
        import logging
        caplog.set_level(logging.INFO)

        sources_data = [
            {'file': 'vid1.mp4', 'url': 'url1', 'title': 'Title1',
             'duration': 120.0, 'duration_tier': 'medium', 'keyword': 'travel'},
            {'file': 'vid2.mp4', 'url': 'url2', 'title': 'Title2',
             'duration': 60.0, 'duration_tier': 'short', 'keyword': 'beach'}
        ]
        sources_file.write_text(json.dumps(sources_data))

        sources = manager.load_sources()

        assert "Loaded 2 existing source records" in caplog.text


# ============================================================================
# Test Sources Saving
# ============================================================================

class TestSourcesSaving:
    """Test saving sources.json"""

    @pytest.mark.fast
    def test_save_sources_basic(self, manager, sources_file):
        """Test basic sources saving"""
        from src.state import DownloadedVideo

        sources = [
            DownloadedVideo(
                file='/videos/vid1.mp4',
                url='https://youtube.com/watch?v=vid1',
                title='Travel Video',
                duration=120.5,
                duration_tier='medium',
                keyword='travel'
            )
        ]

        manager.save_sources(sources)

        # Verify file created
        assert sources_file.exists()

        # Verify contents
        with open(sources_file, 'r') as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]['file'] == '/videos/vid1.mp4'
        assert data[0]['title'] == 'Travel Video'

    @pytest.mark.fast
    def test_save_sources_creates_directory(self, manager, temp_dir):
        """Test that save creates parent directory if missing"""
        from src.state import DownloadedVideo

        nested_sources = temp_dir / "nested" / "sources.json"
        manager.sources_file = nested_sources

        sources = [
            DownloadedVideo(
                file='vid1.mp4',
                url='url1',
                title='Title1',
                duration=60.0,
                duration_tier='short',
                keyword='test'
            )
        ]

        manager.save_sources(sources)

        # Verify directory and file created
        assert nested_sources.exists()
        assert nested_sources.parent.exists()

    @pytest.mark.fast
    def test_save_sources_empty_list(self, manager, sources_file):
        """Test saving empty sources list"""
        manager.save_sources([])

        # Verify file created with empty array
        assert sources_file.exists()
        with open(sources_file, 'r') as f:
            data = json.load(f)
        assert data == []

    @pytest.mark.fast
    def test_save_sources_overwrites_existing(self, manager, sources_file):
        """Test that save overwrites existing sources"""
        from src.state import DownloadedVideo

        # Save initial sources
        sources1 = [
            DownloadedVideo(
                file='vid1.mp4', url='url1', title='Title1',
                duration=60.0, duration_tier='short', keyword='test'
            )
        ]
        manager.save_sources(sources1)

        # Save new sources
        sources2 = [
            DownloadedVideo(
                file='vid2.mp4', url='url2', title='Title2',
                duration=120.0, duration_tier='medium', keyword='test2'
            ),
            DownloadedVideo(
                file='vid3.mp4', url='url3', title='Title3',
                duration=180.0, duration_tier='long', keyword='test3'
            )
        ]
        manager.save_sources(sources2)

        # Verify latest data
        with open(sources_file, 'r') as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]['file'] == 'vid2.mp4'
        assert data[1]['file'] == 'vid3.mp4'
