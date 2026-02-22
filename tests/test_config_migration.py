"""
Tests for Config Migration — US-112-010

Verifies migration from v3.0.0 to v4.0.0 preserves all critical config data
and adds new v4 sections.
"""

import os
import pytest
import tempfile
import shutil
from src.config.utils import ConfigMigration, ConfigMigrationError, CURRENT_CONFIG_VERSION


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def migrator():
    return ConfigMigration()


class TestDetectVersion:
    """Test version detection from config data."""

    def test_detect_v3_version(self, migrator):
        """Detect version 3.0.0 from config."""
        config = {"project": {"version": "3.0.0", "name": "test"}}
        assert migrator.detect_version(config) == "3.0.0"

    def test_detect_v4_version(self, migrator):
        """Detect version 4.0.0 from config."""
        config = {"project": {"version": "4.0.0", "name": "test"}}
        assert migrator.detect_version(config) == "4.0.0"

    def test_default_to_v3_for_missing_version(self, migrator):
        """Default to v3.0.0 when version is missing."""
        config = {"project": {"name": "test"}}
        assert migrator.detect_version(config) == "3.0.0"

    def test_default_to_v3_for_missing_project(self, migrator):
        """Default to v3.0.0 when project section is missing."""
        config = {"matching": {"min_confidence": 0.6}}
        assert migrator.detect_version(config) == "3.0.0"


class TestNeedsMigration:
    """Test migration need detection."""

    def test_v3_needs_migration(self, migrator, temp_config_dir):
        """v3 config should need migration."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  version: "3.0.0"\n  name: test\n')

        assert migrator.needs_migration(config_path) is True

    def test_v4_no_migration_needed(self, migrator, temp_config_dir):
        """v4 config should not need migration."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  version: "4.0.0"\n  name: test\n')

        assert migrator.needs_migration(config_path) is False


class TestMigrateV3ToV4:
    """Test migration from v3.0.0 to v4.0.0."""

    def test_migration_adds_region_backoff(self, migrator):
        """Migration adds region_backoff section."""
        config = {
            "project": {"version": "3.0.0", "name": "test"},
            "video_search": {"results_per_keyword": 20},
        }

        result = migrator._migrate_v3_to_v4(config)

        assert "region_backoff" in result
        assert result["region_backoff"]["enabled"] is False
        assert result["region_backoff"]["us_multiplier"] == 1.0

    def test_migration_adds_search_budget(self, migrator):
        """Migration adds search_budget section."""
        config = {
            "project": {"version": "3.0.0", "name": "test"},
        }

        result = migrator._migrate_v3_to_v4(config)

        assert "search_budget" in result
        assert result["search_budget"]["max_total_results"] == 200
        assert result["search_budget"]["results_per_keyword"] == 20

    def test_migration_adds_video_search_fields(self, migrator):
        """Migration adds search_budget_aware and auto_distribute_budget."""
        config = {
            "project": {"version": "3.0.0", "name": "test"},
            "video_search": {"results_per_keyword": 20},
        }

        result = migrator._migrate_v3_to_v4(config)

        assert result["video_search"]["search_budget_aware"] is True
        assert result["video_search"]["auto_distribute_budget"] is True

    def test_migration_updates_version(self, migrator):
        """Migration updates project version to 4.0.0."""
        config = {
            "project": {"version": "3.0.0", "name": "test"},
        }

        result = migrator._migrate_v3_to_v4(config)

        assert result["project"]["version"] == "4.0.0"

    def test_migration_preserves_existing_fields(self, migrator):
        """Migration preserves existing config fields."""
        config = {
            "project": {"version": "3.0.0", "name": "my-project"},
            "matching": {"min_confidence": 0.7, "enabled": True},
            "video_search": {"results_per_keyword": 30},
        }

        result = migrator._migrate_v3_to_v4(config)

        # Project fields preserved
        assert result["project"]["name"] == "my-project"
        # Matching fields preserved
        assert result["matching"]["min_confidence"] == 0.7
        assert result["matching"]["enabled"] is True
        # Video search fields preserved + new fields added
        assert result["video_search"]["results_per_keyword"] == 30

    def test_migration_does_not_overwrite_existing_v4_sections(self, migrator):
        """Migration doesn't overwrite existing v4 sections."""
        config = {
            "project": {"version": "3.0.0", "name": "test"},
            "region_backoff": {"enabled": True, "us_multiplier": 2.0},
            "search_budget": {"max_total_results": 500},
        }

        result = migrator._migrate_v3_to_v4(config)

        # Should preserve existing v4 sections
        assert result["region_backoff"]["enabled"] is True
        assert result["region_backoff"]["us_multiplier"] == 2.0
        assert result["search_budget"]["max_total_results"] == 500


class TestMigrateConfig:
    """Test full migrate_config method."""

    def test_migrate_file_v3_to_v4(self, migrator, temp_config_dir):
        """Migrate a v3 config file to v4."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  version: "3.0.0"\n  name: test\n')

        result = migrator.migrate_config(config_path, "4.0.0")

        assert result is True

        # Verify file was migrated
        import yaml
        with open(config_path, 'r') as f:
            migrated = yaml.safe_load(f)

        assert migrated["project"]["version"] == "4.0.0"
        assert "region_backoff" in migrated
        assert "search_budget" in migrated

    def test_migrate_same_version_no_op(self, migrator, temp_config_dir):
        """Migrating to same version is a no-op."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  version: "4.0.0"\n  name: test\n')

        result = migrator.migrate_config(config_path, "4.0.0")

        assert result is False

    def test_migrate_invalid_path_raises(self, migrator):
        """Invalid config path raises error."""
        with pytest.raises(ConfigMigrationError, match="Failed to load config"):
            migrator.migrate_config("/nonexistent/path.yaml", "4.0.0")

    def test_migrate_file_v4_to_v5(self, migrator, temp_config_dir):
        """Migrate a v4 config file to v5 through the full pipeline."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  version: "4.0.0"\n  name: test\n')

        result = migrator.migrate_config(config_path, "5.0.0")

        assert result is True

        # Verify file was migrated through the full pipeline
        import yaml
        with open(config_path, 'r') as f:
            migrated = yaml.safe_load(f)

        assert migrated["project"]["version"] == "5.0.0"


class TestBackupAndRollback:
    """Test backup and rollback functionality."""

    def test_backup_creates_file(self, migrator, temp_config_dir):
        """Backup creates a backup file."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        with open(config_path, 'w') as f:
            f.write('project:\n  name: test\n')

        backup_path = migrator.backup_config(config_path)

        assert os.path.exists(backup_path)
        with open(backup_path, 'r') as f:
            assert f.read() == 'project:\n  name: test\n'

    def test_rollback_restores_config(self, migrator, temp_config_dir):
        """Rollback restores config from backup."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        backup_path = os.path.join(temp_config_dir, "backup.yaml")

        # Create original and backup
        with open(config_path, 'w') as f:
            f.write('project:\n  name: original\n')
        with open(backup_path, 'w') as f:
            f.write('project:\n  name: backup\n')

        # Rollback
        result = migrator.rollback(config_path, backup_path)

        assert result is True
        with open(config_path, 'r') as f:
            content = f.read()
            assert "name: backup" in content

    def test_backup_creates_backup_directory(self, migrator, temp_config_dir):
        """Backup creates backup directory if it doesn't exist."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        backup_dir = os.path.join(temp_config_dir, ".config_backups")

        with open(config_path, 'w') as f:
            f.write('project:\n  name: test\n')

        migrator.backup_config(config_path)

        assert os.path.exists(backup_dir)


class TestIntegration:
    """Integration tests for auto-migration."""

    def test_current_version_is_4_0_0(self):
        """Verify current config version is 4.0.0."""
        assert CURRENT_CONFIG_VERSION == "4.0.0"


class TestMigrateV4ToV5:
    """Test migration from v4.0.0 to v5.0.0 (skeleton for documentation)."""

    def test_migration_updates_version(self, migrator):
        """Migration updates project version to 5.0.0."""
        config = {
            "project": {"version": "4.0.0", "name": "test"},
        }

        result = migrator._migrate_v4_to_v5(config)

        assert result["project"]["version"] == "5.0.0"

    def test_migration_preserves_existing_fields(self, migrator):
        """Migration preserves existing config fields."""
        config = {
            "project": {"version": "4.0.0", "name": "my-project"},
            "matching": {"min_confidence": 0.7},
            "video_search": {"results_per_keyword": 30},
            "region_backoff": {"enabled": True},
        }

        result = migrator._migrate_v4_to_v5(config)

        # Project fields preserved
        assert result["project"]["name"] == "my-project"
        # Matching fields preserved
        assert result["matching"]["min_confidence"] == 0.7
        # Video search fields preserved
        assert result["video_search"]["results_per_keyword"] == 30
        # Region backoff preserved
        assert result["region_backoff"]["enabled"] is True

    def test_migration_does_not_mutate_input(self, migrator):
        """Migration should not mutate the input config."""
        config = {
            "project": {"version": "4.0.0", "name": "test"},
        }
        original_version = config["project"]["version"]

        migrator._migrate_v4_to_v5(config)

        # Original should be unchanged
        assert config["project"]["version"] == original_version

    def test_migration_skeleton_example_pattern(self, migrator):
        """Migration skeleton follows documented pattern for adding sections."""
        config = {
            "project": {"version": "4.0.0", "name": "test"},
            "matching": {"min_confidence": 0.6},
        }

        result = migrator._migrate_v4_to_v5(config)

        # Version updated
        assert result["project"]["version"] == "5.0.0"
        # Existing sections preserved
        assert "matching" in result
        assert result["matching"]["min_confidence"] == 0.6


class TestSectionVersions:
    """Test section-level schema versioning."""

    def test_get_section_version_from_config_data(self, migrator):
        """Get section version from config data."""
        config_data = {
            "_section_versions": {
                "matching": "1.0.0",
                "video_search": "2.0.0",
            }
        }

        assert migrator.get_section_version(config_data, "matching") == "1.0.0"
        assert migrator.get_section_version(config_data, "video_search") == "2.0.0"
        assert migrator.get_section_version(config_data, "nonexistent") is None

    def test_needs_section_migration_true(self, migrator):
        """Section needs migration when versions differ."""
        config_data = {
            "_section_versions": {
                "matching": "1.0.0",
            }
        }

        assert migrator.needs_section_migration(config_data, "matching", "2.0.0") is True

    def test_needs_section_migration_false(self, migrator):
        """Section does not need migration when versions match."""
        config_data = {
            "_section_versions": {
                "matching": "2.0.0",
            }
        }

        assert migrator.needs_section_migration(config_data, "matching", "2.0.0") is False

    def test_migrate_section_updates_version(self, migrator, temp_config_dir):
        """Migrating a section updates its version in metadata."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        config_content = """project:
  version: "4.0.0"
  name: test
matching:
  min_confidence: 0.6
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        # Register a section migration
        def migrate_matching_v1_to_v2(section_data, from_version):
            section_data['new_field'] = 'default'
            return section_data

        migrator.SECTION_MIGRATIONS['matching'] = {
            '2.0.0': migrate_matching_v1_to_v2
        }

        # Migrate section
        result = migrator.migrate_section(config_path, "matching", "2.0.0")

        assert result is True

        # Verify version was updated in file
        import yaml
        with open(config_path, 'r') as f:
            migrated_config = yaml.safe_load(f)

        assert migrated_config['_section_versions']['matching'] == "2.0.0"
        assert migrated_config['matching']['new_field'] == 'default'

    def test_migrate_section_already_at_version(self, migrator, temp_config_dir):
        """Migrating a section that is already at target version returns False."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        config_content = """project:
  version: "4.0.0"
  name: test
_section_versions:
  matching: "2.0.0"
matching:
  min_confidence: 0.6
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        result = migrator.migrate_section(config_path, "matching", "2.0.0")

        assert result is False

    def test_migrate_section_missing_section(self, migrator, temp_config_dir):
        """Migrating a missing section returns False."""
        config_path = os.path.join(temp_config_dir, "config.yaml")
        config_content = """project:
  version: "4.0.0"
  name: test
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        result = migrator.migrate_section(config_path, "nonexistent", "1.0.0")

        assert result is False


class TestConfigSectionVersions:
    """Test section versioning in Config class."""

    def test_get_section_version(self):
        """Config.get_section_version returns correct version."""
        from src.config.base import Config

        config = Config()
        config.set_section_version("matching", "1.0.0")

        assert config.get_section_version("matching") == "1.0.0"
        assert config.get_section_version("nonexistent") is None

    def test_get_all_section_versions(self):
        """Config.get_all_section_versions returns all versions."""
        from src.config.base import Config

        config = Config()
        config.set_section_version("matching", "1.0.0")
        config.set_section_version("video_search", "2.0.0")

        versions = config.get_all_section_versions()

        assert versions["matching"] == "1.0.0"
        assert versions["video_search"] == "2.0.0"

    def test_set_section_version(self):
        """Config.set_section_version stores version correctly."""
        from src.config.base import Config

        config = Config()
        config.set_section_version("matching", "1.0.0")

        assert config._section_versions["matching"] == "1.0.0"

    def test_section_versions_in_json_export(self):
        """Section versions included in JSON export metadata."""
        from src.config.base import Config

        config = Config()
        config.set_section_version("matching", "1.0.0")
        config.set_section_version("video_search", "2.0.0")

        json_output = config.to_json()
        import json
        data = json.loads(json_output)

        assert data['metadata']['section_versions']['matching'] == "1.0.0"
        assert data['metadata']['section_versions']['video_search'] == "2.0.0"
