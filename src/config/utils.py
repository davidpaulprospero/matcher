"""
Configuration utility functions.

Provides safe_get_config_value() to eliminate scattered dict/object
dual-access patterns throughout the codebase (CLAUDE.md Rule 6).

Also provides ConfigMigration for major version upgrades (v3 -> v4).
"""

import copy
import logging
import os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Config migration log prefix
CONFIG_MIGRATION_PREFIX = "[CONFIG_MIGRATION]"

# Current config version
CURRENT_CONFIG_VERSION = "4.0.0"

# Example v5 migration skeleton (for documentation purposes)
# Uncomment and customize when creating v5:
# MIGRATIONS: Dict[Tuple[str, str], callable] = {
#     ("3.0.0", "4.0.0"): _migrate_v3_to_v4,
#     ("4.0.0", "5.0.0"): _migrate_v4_to_v5,  # Example skeleton
# }


def safe_get_config_value(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a config value from either a dict or a dataclass/object.

    Implements CLAUDE.md Rule 6: Handle both dict .get() and getattr() access.

    Args:
        obj: A dict, dataclass instance, or any object
        key: The attribute/key name to look up
        default: Default value if key is missing

    Returns:
        The value for the key, or default if not found

    Examples:
        >>> safe_get_config_value({'timeout': 30}, 'timeout', 10)
        30
        >>> safe_get_config_value(some_dataclass, 'timeout', 10)
        10  # if timeout not set
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def config_diff(config_a: Any, config_b: Any, _prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Compare two Config instances and return a dict of differences.

    Recursively compares dataclass fields. Fields with equal values are excluded.

    Args:
        config_a: First config instance
        config_b: Second config instance
        _prefix: Internal use - dot-delimited field path prefix

    Returns:
        Dict mapping field paths to {'old': value_a, 'new': value_b}.
        Empty dict if configs are identical.

    Example:
        >>> diff = config_diff(default_config, modified_config)
        >>> # {'matching.min_confidence': {'old': 0.6, 'new': 0.8}}
    """
    diffs: Dict[str, Dict[str, Any]] = {}

    if not is_dataclass(config_a) or not is_dataclass(config_b):
        # Leaf comparison for non-dataclass values
        if config_a != config_b:
            diffs[_prefix] = {"old": config_a, "new": config_b}
        return diffs

    for f in fields(config_a):
        # Skip private/metadata fields (e.g., _loaded_at, _config_hash)
        if f.name.startswith("_"):
            continue
        path = f"{_prefix}.{f.name}" if _prefix else f.name
        val_a = getattr(config_a, f.name)
        val_b = getattr(config_b, f.name)

        if is_dataclass(val_a) and is_dataclass(val_b):
            diffs.update(config_diff(val_a, val_b, _prefix=path))
        elif val_a != val_b:
            diffs[path] = {"old": val_a, "new": val_b}

    return diffs


class ConfigMigrationError(Exception):
    """Raised when config migration fails."""
    pass


class ConfigMigration:
    """
    Handles configuration file migrations between major versions.

    Supports automatic migration from v3.x to v4.0.0 with rollback capability.

    Usage:
        migrator = ConfigMigration()
        # Check if migration needed
        if migrator.needs_migration("/path/to/config.yaml"):
            # Create backup before migration
            migrator.backup_config("/path/to/config.yaml")
            # Migrate
            migrator.migrate_config("/path/to/config.yaml")

    Migration changes (v3 -> v4):
        - Add region_backoff section with defaults
        - Add search_budget section
        - Add search_budget_aware field to video_search
        - Add auto_distribute_budget field to video_search
    """

    # Migration registry: (from_version, to_version) -> migration function
    MIGRATIONS: Dict[Tuple[str, str], callable] = {}

    def __init__(self, backup_dir: Optional[str] = None):
        """
        Initialize migration handler.

        Args:
            backup_dir: Directory to store backups. Defaults to .config_backups
                       in the same directory as the config file.
        """
        self.backup_dir = backup_dir
        self._register_migrations()
        self._register_section_migrations()

    def _register_migrations(self):
        """Register all available migrations."""
        self.MIGRATIONS = {
            ("3.0.0", "4.0.0"): self._migrate_v3_to_v4,
            # v4 to v5 migration (skeleton for documentation)
            ("4.0.0", "5.0.0"): self._migrate_v4_to_v5,
        }

    def detect_version(self, config_data: Dict[str, Any]) -> str:
        """
        Detect config version from raw config dict.

        Args:
            config_data: Raw config dict loaded from YAML

        Returns:
            Version string (e.g., "3.0.0" or "4.0.0")
        """
        if "project" in config_data and isinstance(config_data["project"], dict):
            return config_data["project"].get("version", "3.0.0")
        return "3.0.0"  # Default to v3 for old configs without version

    def needs_migration(self, config_path: str) -> bool:
        """
        Check if a config file needs migration.

        Args:
            config_path: Path to config.yaml

        Returns:
            True if migration is needed
        """
        import yaml

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            current_version = self.detect_version(config_data)
            needs_mig = current_version != CURRENT_CONFIG_VERSION
            logger.info(f"{CONFIG_MIGRATION_PREFIX} Config version check: current={current_version}, target={CURRENT_CONFIG_VERSION}, migration_needed={needs_mig}")
            return needs_mig
        except Exception as e:
            logger.warning(f"{CONFIG_MIGRATION_PREFIX} Could not check config version: {e}")
            return False

    def migrate_config(self, config_path: str, target_version: str = "4.0.0") -> bool:
        """
        Migrate a config file to target version.

        Args:
            config_path: Path to config.yaml
            target_version: Target version to migrate to

        Returns:
            True if migration was performed

        Raises:
            ConfigMigrationError: If migration fails
        """
        import yaml

        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[CFG-002] Failed to load config for migration: {e}")
            raise ConfigMigrationError(f"Failed to load config: {e}")

        current_version = self.detect_version(config_data)

        if current_version == target_version:
            logger.info(f"{CONFIG_MIGRATION_PREFIX} Config already at version {target_version}")
            return False

        # Check if migration path exists
        migration_key = (current_version, target_version)
        if migration_key not in self.MIGRATIONS:
            logger.error(f"[CFG-002] No migration path from {current_version} to {target_version}")
            raise ConfigMigrationError(
                f"No migration path from {current_version} to {target_version}"
            )

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Starting migration from {current_version} to {target_version}")

        # Perform migration
        migration_func = self.MIGRATIONS[migration_key]
        try:
            migrated_data = migration_func(config_data)
            logger.info(f"{CONFIG_MIGRATION_PREFIX} Migration function completed for {current_version} -> {target_version}")
        except Exception as e:
            logger.error(f"[CFG-002] Migration failed: {e}")
            raise ConfigMigrationError(f"Migration failed: {e}")

        # Write migrated config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(migrated_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"[CFG-002] Failed to write migrated config: {e}")
            raise ConfigMigrationError(f"Failed to write migrated config: {e}")

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Config migrated successfully to {target_version}")
        return True

    def backup_config(self, config_path: str) -> str:
        """
        Create a backup of config file before migration.

        Args:
            config_path: Path to config.yaml

        Returns:
            Path to backup file
        """
        import yaml
        from datetime import datetime

        config_dir = os.path.dirname(os.path.abspath(config_path))
        backup_base = self.backup_dir or os.path.join(config_dir, ".config_backups")
        os.makedirs(backup_base, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_base, f"config.yaml.backup_{timestamp}")

        with open(config_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Config backed up to {backup_path}")
        return backup_path

    def rollback(self, config_path: str, backup_path: str) -> bool:
        """
        Rollback config to a backup.

        Args:
            config_path: Path to config.yaml
            backup_path: Path to backup file

        Returns:
            True if rollback was successful
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as src:
                with open(config_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            logger.info(f"{CONFIG_MIGRATION_PREFIX} Config rolled back to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"[CFG-002] Rollback failed: {e}")
            return False

    def _migrate_v3_to_v4(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate config from v3.0.0 to v4.0.0.

        Changes:
        - Add region_backoff section with defaults
        - Add search_budget section
        - Add search_budget_aware and auto_distribute_budget to video_search
        - Update project.version to 4.0.0
        - Add version_history tracking

        Args:
            config: Raw config dict

        Returns:
            Migrated config dict
        """
        from datetime import datetime

        config = copy.deepcopy(config)

        # Update project version
        if "project" not in config:
            config["project"] = {}
        config["project"]["version"] = "4.0.0"

        # Add version_history tracking
        config["_version_history"] = [
            {
                "version": "4.0.0",
                "date": datetime.now().isoformat(),
                "notes": "Migrated from v3.0.0: Added region_backoff, search_budget, search_budget_aware, auto_distribute_budget"
            }
        ]
        logger.info("Added version_history for v4.0.0 migration")

        # Add region_backoff section (new in v4)
        if "region_backoff" not in config:
            config["region_backoff"] = {
                "enabled": False,
                "us_multiplier": 1.0,
                "eu_multiplier": 1.2,
                "asia_multiplier": 1.5,
                "other_multiplier": 2.0,
                "track_per_region": False,
            }
            logger.info("Added region_backoff section with defaults")

        # Add search_budget section (new in v4)
        if "search_budget" not in config:
            config["search_budget"] = {
                "max_total_results": 200,
                "results_per_keyword": 20,
            }
            logger.info("Added search_budget section with defaults")

        # Add search_budget_aware and auto_distribute_budget to video_search (new in v4)
        if "video_search" in config:
            if "search_budget_aware" not in config["video_search"]:
                config["video_search"]["search_budget_aware"] = True
                logger.info("Added search_budget_aware to video_search")

            if "auto_distribute_budget" not in config["video_search"]:
                config["video_search"]["auto_distribute_budget"] = True
                logger.info("Added auto_distribute_budget to video_search")

        logger.info("Config migration v3->v4 completed")
        return config

    def _migrate_v4_to_v5(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate config from v4.0.0 to v5.0.0.

        This is a SKELETON migration for documentation purposes.
        Edit this function to add actual v5 changes.

        Example changes for v5:
        - Add new required sections
        - Rename or restructure existing fields
        - Add new configuration options

        Args:
            config: Raw config dict (v4.0.0 format)

        Returns:
            Migrated config dict (v5.0.0 format)
        """
        from datetime import datetime

        config = copy.deepcopy(config)

        # Update project version
        if "project" not in config:
            config["project"] = {}
        config["project"]["version"] = "5.0.0"

        # Preserve existing version_history and add new entry
        existing_history = config.get("_version_history", [])
        existing_history.append({
            "version": "5.0.0",
            "date": datetime.now().isoformat(),
            "notes": "Migrated from v4.0.0"
        })
        config["_version_history"] = existing_history
        logger.info("Added version_history for v5.0.0 migration")

        # EXAMPLE: Add new section (uncomment and customize)
        # if "new_feature" not in config:
        #     config["new_feature"] = {
        #         "enabled": False,
        #         "option1": "default_value",
        #         "option2": 100,
        #     }
        #     logger.info("Added new_feature section with defaults")

        # EXAMPLE: Add new field to existing section
        # if "video_search" in config:
        #     if "new_option" not in config["video_search"]:
        #         config["video_search"]["new_option"] = True
        #         logger.info("Added new_option to video_search")

        # EXAMPLE: Migrate deprecated field to new field
        # if "old_field" in config.get("matching", {}):
        #     if "new_field" not in config.get("matching", {}):
        #         config["matching"]["new_field"] = config["matching"]["old_field"]
        #     del config["matching"]["old_field"]
        #     logger.info("Migrated old_field to new_field in matching")

        logger.info("Config migration v4->v5 completed (skeleton)")
        return config

    # Section-specific migration registry: section_name -> {version: migration_func}
    SECTION_MIGRATIONS: Dict[str, Dict[str, callable]] = {}

    def _register_section_migrations(self):
        """Register section-specific migrations."""
        self.SECTION_MIGRATIONS = {
            # Example: 'matching': {'1.0.0': self._migrate_matching_v1_to_v2}
        }

    def get_section_version(self, config_data: Dict[str, Any], section_name: str) -> Optional[str]:
        """
        Get the version of a specific section from config data.

        Args:
            config_data: Raw config dict
            section_name: Name of the section

        Returns:
            Version string if section has a version, None otherwise
        """
        section_versions = config_data.get('_section_versions', {})
        return section_versions.get(section_name)

    def needs_section_migration(self, config_data: Dict[str, Any], section_name: str, target_version: str) -> bool:
        """
        Check if a specific section needs migration.

        Args:
            config_data: Raw config dict
            section_name: Name of the section
            target_version: Target version for the section

        Returns:
            True if migration is needed
        """
        current_version = self.get_section_version(config_data, section_name)
        return current_version != target_version

    def migrate_section(self, config_path: str, section_name: str, target_version: str) -> bool:
        """
        Migrate a specific section of a config file.

        Args:
            config_path: Path to config.yaml
            section_name: Name of the section to migrate
            target_version: Target version to migrate to

        Returns:
            True if migration was performed

        Raises:
            ConfigMigrationError: If migration fails
        """
        import yaml

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Starting section migration: {section_name} -> {target_version}")

        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[CFG-002] Failed to load config for section migration: {e}")
            raise ConfigMigrationError(f"Failed to load config: {e}")

        # Check if section exists
        if section_name not in config_data:
            logger.warning(f"{CONFIG_MIGRATION_PREFIX} Section '{section_name}' not found in config")
            return False

        # Get current section version
        current_version = self.get_section_version(config_data, section_name)

        if current_version == target_version:
            logger.info(f"{CONFIG_MIGRATION_PREFIX} Section '{section_name}' already at version {target_version}")
            return False

        # Check if migration path exists
        section_migrations = self.SECTION_MIGRATIONS.get(section_name, {})
        migration_func = section_migrations.get(target_version)

        if migration_func is None:
            # No specific migration, but we still update the version metadata
            logger.info(f"{CONFIG_MIGRATION_PREFIX} No migration function for section '{section_name}' to {target_version}, updating version only")
        else:
            # Apply migration
            try:
                config_data[section_name] = migration_func(config_data[section_name], current_version)
                logger.info(f"{CONFIG_MIGRATION_PREFIX} Section migration function completed for {section_name}")
            except Exception as e:
                logger.error(f"[CFG-002] Section migration failed for '{section_name}': {e}")
                raise ConfigMigrationError(f"Section migration failed for '{section_name}': {e}")

        # Update section version in metadata
        if '_section_versions' not in config_data:
            config_data['_section_versions'] = {}
        config_data['_section_versions'][section_name] = target_version

        # Write migrated config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"[CFG-002] Failed to write migrated config: {e}")
            raise ConfigMigrationError(f"Failed to write migrated config: {e}")

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Section '{section_name}' migrated to version {target_version}")
        return True

    def migrate_all_sections(self, config_path: str) -> Dict[str, bool]:
        """
        Migrate all sections that need migration.

        Args:
            config_path: Path to config.yaml

        Returns:
            Dict mapping section names to migration status (True if migrated)
        """
        logger.info(f"{CONFIG_MIGRATION_PREFIX} Starting migration of all sections")
        results = {}

        # Get all registered sections
        for section_name in self.SECTION_MIGRATIONS.keys():
            section_migrations = self.SECTION_MIGRATIONS[section_name]
            if section_migrations:
                # Get the latest version for this section
                latest_version = max(section_migrations.keys())
                try:
                    migrated = self.migrate_section(config_path, section_name, latest_version)
                    results[section_name] = migrated
                except ConfigMigrationError as e:
                    logger.error(f"[CFG-002] Failed to migrate section '{section_name}': {e}")
                    results[section_name] = False

        logger.info(f"{CONFIG_MIGRATION_PREFIX} Section migration complete: {sum(results.values())}/{len(results)} migrated")
        return results
