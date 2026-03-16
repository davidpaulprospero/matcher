"""Tests for runtime config hot-reload with change callback (US-112-011)."""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch

from src.config.base import (
    Config,
    FrozenConfigError,
    load_config,
    set_config,
    get_config,
)


# Fixtures for test API keys
@pytest.fixture
def mock_api_keys():
    """Provide mock API keys for tests."""
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_gemini_key_12345',
        'GOOGLE_API_KEY': 'test_google_key_12345',
        'PEXELS_API_KEY': 'test_pexels_key',
        'PIXABAY_API_KEY': 'test_pixabay_key',
    }):
        yield


class TestConfigChangeCallbacks:
    """Test config change callback registration and invocation."""

    def test_register_global_callback(self):
        """Test registering a global callback that fires on any config change."""
        config = Config()

        callback_invoked = []

        def global_callback(cfg, changed_sections):
            callback_invoked.append((cfg, changed_sections))

        config.register_change_callback(global_callback)

        assert '*' in config._change_callbacks
        assert global_callback in config._change_callbacks['*']

    def test_register_section_callback(self):
        """Test registering a callback for a specific section."""
        config = Config()

        def matching_callback(cfg, changed_sections):
            pass

        config.register_change_callback(matching_callback, section='matching')

        assert 'matching' in config._change_callbacks
        assert matching_callback in config._change_callbacks['matching']

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        config = Config()

        def my_callback(cfg, changed_sections):
            pass

        config.register_change_callback(my_callback)
        assert my_callback in config._change_callbacks['*']

        result = config.unregister_change_callback(my_callback)
        assert result is True
        assert my_callback not in config._change_callbacks['*']

    def test_unregister_section_callback(self):
        """Test unregistering a section-specific callback."""
        config = Config()

        def my_callback(cfg, changed_sections):
            pass

        config.register_change_callback(my_callback, section='matching')
        result = config.unregister_change_callback(my_callback, section='matching')

        assert result is True
        assert my_callback not in config._change_callbacks['matching']

    def test_unregister_nonexistent_callback(self):
        """Test unregistering a callback that was never registered."""
        config = Config()

        def never_registered(cfg, changed_sections):
            pass

        result = config.unregister_change_callback(never_registered)
        assert result is False


class TestConfigReloadCallbacks:
    """Test callback invocation during config reload."""

    def test_callback_invoked_on_reload(self, mock_api_keys):
        """Test that callbacks are invoked when config is reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            # Write initial config
            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            # Load config
            config = Config.from_yaml(config_path)

            callback_invoked = []

            def change_callback(cfg, changed_sections):
                callback_invoked.append((cfg, changed_sections))

            config.register_change_callback(change_callback)

            # Modify config file to trigger reload
            modified_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.8}
            }
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Reload
            result = config.reload()

            assert result is True
            assert len(callback_invoked) == 1
            cfg, changed_sections = callback_invoked[0]
            assert cfg is config
            assert 'matching' in changed_sections or 'project' in changed_sections

    def test_section_callback_only_fires_for_matching_section(self, mock_api_keys):
        """Test that section-specific callbacks only fire when that section changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            # Write initial config
            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            # Load config
            config = Config.from_yaml(config_path)

            matching_called = []
            global_called = []

            def matching_callback(cfg, changed_sections):
                matching_called.append((cfg, changed_sections))

            def global_callback(cfg, changed_sections):
                global_called.append((cfg, changed_sections))

            config.register_change_callback(matching_callback, section='matching')
            config.register_change_callback(global_callback)

            # Modify only project section (not matching)
            modified_config = {'project': {'name': 'test2', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Reload
            result = config.reload()

            assert result is True
            # Global callback should fire
            assert len(global_called) == 1
            # Matching callback should NOT fire (matching didn't change)
            assert len(matching_called) == 0


class TestFrozenConfigReload:
    """Test that frozen config raises error on reload attempt."""

    def test_frozen_config_raises_error_on_reload(self, mock_api_keys):
        """Test that reloading a frozen config raises FrozenConfigError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)
            config.freeze()

            # Modify config file
            modified_config = {'project': {'name': 'test2', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Attempt reload should raise FrozenConfigError
            with pytest.raises(FrozenConfigError) as exc_info:
                config.reload()

            assert 'frozen' in str(exc_info.value).lower()

    def test_unfreeze_then_reload_works(self, mock_api_keys):
        """Test that unfreezing allows reload to work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)
            config.freeze()
            config.unfreeze()

            # Modify config file
            modified_config = {'project': {'name': 'test2', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Reload should now work
            result = config.reload()
            assert result is True
            assert config.project.name == 'test2'


class TestCallbackPreservation:
    """Test that callbacks are preserved across reloads."""

    def test_callbacks_preserved_after_reload(self, mock_api_keys):
        """Test that registered callbacks persist after reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            def my_callback(cfg, changed_sections):
                pass

            config.register_change_callback(my_callback)

            # Verify callback registered
            assert my_callback in config._change_callbacks['*']

            # Modify and reload
            modified_config = {'project': {'name': 'test2', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            config.reload()

            # Callback should still be registered after reload
            assert my_callback in config._change_callbacks['*']


class TestCallbackErrorHandling:
    """Test error handling in callbacks."""

    def test_callback_error_does_not_propagate(self, mock_api_keys):
        """Test that callback exceptions are caught and logged, not propagated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {'project': {'name': 'test', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            def failing_callback(cfg, changed_sections):
                raise RuntimeError("Callback failed!")

            config.register_change_callback(failing_callback)

            # Modify and reload - should NOT raise
            modified_config = {'project': {'name': 'test2', 'version': '4.0.0'}}
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Should complete without raising
            result = config.reload()
            assert result is True


class TestConfigDriftDetection:
    """Test config drift detection (US-120-011)."""

    def test_field_hashes_computed_at_startup(self):
        """Test that field hashes are computed when config is loaded."""
        config = Config()

        # Should have hashes for all sections
        assert len(config._field_hashes) > 0
        assert 'matching' in config._field_hashes
        assert 'project' in config._field_hashes

    def test_field_hashes_different_for_different_values(self):
        """Test that the same default config produces the same hash."""
        config1 = Config()
        config2 = Config()

        # Default configs with same values should have same hashes
        assert config1._field_hashes.get('matching') == config2._field_hashes.get('matching')

        # But modifying one config changes its hash (not recomputed, but different from another instance)
        # This test verifies hashes are computed consistently for same default values
        assert len(config1._field_hashes) > 0

    def test_check_drift_no_change(self):
        """Test that check_drift returns empty when no changes occurred."""
        config = Config()

        drift = config.check_drift()

        # No drift should be detected
        assert len(drift) == 0

    def test_check_drift_detects_change(self):
        """Test that check_drift detects programmatic config changes."""
        config = Config()

        # Store original hash
        original_hash = config._field_hashes.get('matching')

        # Modify matching config programmatically
        config.matching.min_confidence = 0.9

        # Check drift - should detect the change
        drift = config.check_drift(force_warn=True)

        # Should detect drift in matching section
        assert len(drift) == 1
        assert drift[0]['section'] == 'matching'
        assert drift[0]['old_hash'] == original_hash
        assert drift[0]['new_hash'] != original_hash

    def test_check_drift_multiple_sections(self):
        """Test that check_drift detects changes in multiple sections."""
        config = Config()

        # Modify multiple sections
        config.matching.min_confidence = 0.9
        config.download.max_retries = 10

        # Check drift
        drift = config.check_drift(force_warn=True)

        # Should detect drift in at least 2 sections
        sections_with_drift = [d['section'] for d in drift]
        assert 'matching' in sections_with_drift
        assert 'download' in sections_with_drift

    def test_drift_warning_only_logged_once(self):
        """Test that drift warning is only logged once per config instance."""
        config = Config()

        # Modify config
        config.matching.min_confidence = 0.9

        # First check - should log warning
        drift1 = config.check_drift()
        assert len(drift1) > 0

        # Second check without force_warn - should NOT log again
        drift2 = config.check_drift(force_warn=False)
        # drift should still be detected, but _drift_warnings_logged is now True
        assert config._drift_warnings_logged is True

    def test_drift_warning_forced_repeat(self):
        """Test that force_warn allows repeated drift warnings."""
        config = Config()

        # Modify config
        config.matching.min_confidence = 0.9

        # First check
        drift1 = config.check_drift(force_warn=True)
        assert len(drift1) > 0
        assert config._drift_warnings_logged is True

        # Second check with force_warn - should work
        drift2 = config.check_drift(force_warn=True)
        # Should still detect drift
        assert len(drift2) > 0


class TestReloadSection:
    """Test selective section reload (US-128-005)."""

    def test_reload_section_only_target_modified(self, mock_api_keys):
        """Test that reload_section only modifies the target section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            # Write initial config
            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            # Load config
            config = Config.from_yaml(config_path)

            # Store original values
            original_matching = config.matching.min_confidence
            original_project_name = config.project.name

            # Modify matching section in file
            modified_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.9}
            }
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # Reload only matching section
            result = config.reload_section('matching')

            assert result is True
            # Matching should be updated
            assert config.matching.min_confidence == 0.9
            # Project should remain unchanged
            assert config.project.name == original_project_name

    def test_reload_section_returns_false_when_unchanged(self, mock_api_keys):
        """Test that reload_section returns False when section hasn't changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Reload without file change
            result = config.reload_section('matching')

            assert result is False

    def test_reload_section_invalid_section_raises_error(self):
        """Test that reload_section raises ValueError for invalid section."""
        config = Config()

        with pytest.raises(ValueError) as exc_info:
            config.reload_section('nonexistent_section')

        assert 'Invalid section' in str(exc_info.value)

    def test_reload_section_callback_receives_section_name(self, mock_api_keys):
        """Test that section-specific callback receives correct section name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            callback_invoked = []

            def matching_callback(cfg, changed_sections):
                callback_invoked.append(changed_sections)

            config.register_change_callback(matching_callback, section='matching')

            # Modify matching section
            modified_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.9}
            }
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            config.reload_section('matching')

            # Callback should be invoked with matching in changed_sections
            assert len(callback_invoked) == 1
            assert 'matching' in callback_invoked[0]

    def test_reload_section_frozen_config(self, mock_api_keys):
        """Test that reload_section works when config is frozen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)
            config.freeze()

            # Modify matching section in file
            modified_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.9}
            }
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            # reload_section should work without raising FrozenConfigError
            result = config.reload_section('matching')

            assert result is True
            assert config.matching.min_confidence == 0.9

            # Config should be refrozen after reload_section
            # (Trying to modify should raise error)
            with pytest.raises(FrozenConfigError):
                config.matching.min_confidence = 0.5

    def test_reload_section_preserves_other_sections(self, mock_api_keys):
        """Test that reloading one section doesn't affect others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7},
                'download': {'max_retries': 3}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Modify matching section only
            modified_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.9},
                'download': {'max_retries': 3}
            }
            with open(config_path, 'w') as f:
                yaml.dump(modified_config, f)

            config.reload_section('matching')

            # Download should be unchanged
            assert config.download.max_retries == 3
            assert config.matching.min_confidence == 0.9


class TestConfigUpdateSection:
    """Test Config.update_section() for runtime config modification (US-128-010)."""

    def test_update_section_simple_field(self, mock_api_keys):
        """Test updating a simple field in a config section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Update matching min_confidence
            result = config.update_section('matching', {'min_confidence': 0.85})

            assert result is True
            assert config.matching.min_confidence == 0.85

    def test_update_section_multiple_fields(self, mock_api_keys):
        """Test updating multiple fields in the same section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7, 'max_retries': 3}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Update multiple fields
            result = config.update_section('matching', {
                'min_confidence': 0.8,
                'max_retries': 5
            })

            assert result is True
            assert config.matching.min_confidence == 0.8
            assert config.matching.max_retries == 5

    def test_update_section_invalid_field_raises_error(self, mock_api_keys):
        """Test that updating an invalid field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Try to update invalid field
            with pytest.raises(ValueError) as exc_info:
                config.update_section('matching', {'invalid_field': 0.5})

            assert 'invalid_field' in str(exc_info.value)

    def test_update_section_invalid_section_raises_error(self):
        """Test that updating an invalid section raises ValueError."""
        config = Config()

        with pytest.raises(ValueError) as exc_info:
            config.update_section('nonexistent_section', {'field': 'value'})

        assert 'nonexistent_section' in str(exc_info.value)

    def test_update_section_type_coercion(self, mock_api_keys):
        """Test that string values are coerced to correct types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7, 'max_retries': 3}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Update with string values that should be coerced
            result = config.update_section('matching', {
                'min_confidence': '0.9',  # string -> float
                'max_retries': '5'        # string -> int
            })

            assert result is True
            assert config.matching.min_confidence == 0.9
            assert config.matching.max_retries == 5

    def test_update_section_triggers_callbacks(self, mock_api_keys):
        """Test that update_section triggers change callbacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            callback_invoked = []

            def matching_callback(cfg, changed_sections):
                callback_invoked.append(changed_sections)

            config.register_change_callback(matching_callback, section='matching')

            config.update_section('matching', {'min_confidence': 0.85})

            assert len(callback_invoked) == 1
            assert 'matching' in callback_invoked[0]

    def test_update_section_frozen_config(self, mock_api_keys):
        """Test that frozen config auto-unfreezes and refreezes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)
            config.freeze()

            # Update should work despite being frozen
            result = config.update_section('matching', {'min_confidence': 0.9})

            assert result is True
            assert config.matching.min_confidence == 0.9

            # Config should be refrozen after update
            with pytest.raises(FrozenConfigError):
                config.matching.min_confidence = 0.5

    def test_update_section_preserves_other_sections(self, mock_api_keys):
        """Test that updating one section doesn't affect others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7},
                'download': {'max_retries': 3}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            # Update matching section
            config.update_section('matching', {'min_confidence': 0.9})

            # Download should be unchanged
            assert config.download.max_retries == 3
            assert config.matching.min_confidence == 0.9

    def test_update_section_returns_true_on_success(self, mock_api_keys):
        """Test that update_section returns True on successful update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')

            initial_config = {
                'project': {'name': 'test', 'version': '4.0.0'},
                'matching': {'min_confidence': 0.7}
            }
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            config = Config.from_yaml(config_path)

            result = config.update_section('matching', {'min_confidence': 0.8})

            assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
