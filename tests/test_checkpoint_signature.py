"""US-130-008: Tests for checkpoint cryptographic signature/verification."""

import gzip
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.checkpoint import CheckpointManager, CheckpointData


class TestCheckpointSignature:
    """Test checkpoint signature and verification functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def checkpoint_with_key(self, temp_project_dir, monkeypatch):
        """Create checkpoint manager with signature key configured."""
        # Set environment variable for signature key
        monkeypatch.setenv("MATCHER_CHECKPOINT_SECRET", "test-secret-key-12345")
        # Create checkpoint manager
        return CheckpointManager(temp_project_dir, config_hash="testhash")

    @pytest.fixture
    def checkpoint_without_key(self, temp_project_dir, monkeypatch):
        """Create checkpoint manager without signature key."""
        # Ensure no env var
        monkeypatch.delenv("MATCHER_CHECKPOINT_SECRET", raising=False)
        # Create checkpoint manager
        return CheckpointManager(temp_project_dir, config_hash="testhash")

    def test_signature_computation(self, checkpoint_with_key):
        """Test that _compute_signature returns a valid HMAC-SHA256."""
        data = {
            "version": "3.0",
            "created_at": "2026-02-18T00:00:00",
            "updated_at": "2026-02-18T01:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
        }

        signature = checkpoint_with_key._compute_signature(data)

        # Should return a 64-character hex string (SHA256)
        assert signature is not None
        assert len(signature) == 64
        assert all(c in "0123456789abcdef" for c in signature)

    def test_signature_deterministic(self, checkpoint_with_key):
        """Test that signature is deterministic for same input."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
        }

        sig1 = checkpoint_with_key._compute_signature(data)
        sig2 = checkpoint_with_key._compute_signature(data)

        assert sig1 == sig2

    def test_signature_changes_with_data(self, checkpoint_with_key):
        """Test that signature changes when data changes."""
        data1 = {"version": "3.0", "last_completed_stage": "MATCH"}
        data2 = {"version": "3.0", "last_completed_stage": "DOWNLOAD"}

        sig1 = checkpoint_with_key._compute_signature(data1)
        sig2 = checkpoint_with_key._compute_signature(data2)

        assert sig1 != sig2

    def test_verify_signature_valid(self, checkpoint_with_key):
        """Test signature verification with valid signature."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
        }

        # Compute signature and add to data
        data["signature"] = checkpoint_with_key._compute_signature(data)

        # Verify
        result = checkpoint_with_key.verify_signature(data)

        assert result["is_valid"] is True
        assert result["signature_present"] is True

    def test_verify_signature_invalid(self, checkpoint_with_key):
        """Test signature verification with invalid/tampered signature."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
            "signature": "invalid_signature_that_is_not_valid",
        }

        result = checkpoint_with_key.verify_signature(data)

        assert result["is_valid"] is False
        assert result["signature_present"] is True
        assert "mismatch" in result["issue"].lower()

    def test_verify_signature_missing_with_key(self, checkpoint_with_key):
        """Test verification when signature missing but key is configured."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
            # No signature field
        }

        result = checkpoint_with_key.verify_signature(data)

        # Should fail because key is configured but no signature
        assert result["signature_present"] is False

    def test_verify_signature_missing_without_key(self, checkpoint_without_key):
        """Test verification when no signature and no key configured."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
        }

        result = checkpoint_without_key.verify_signature(data)

        # Should pass because legacy checkpoint
        assert result["is_valid"] is True

    def test_signature_excluded_from_signing(self, checkpoint_with_key):
        """Test that signature field is excluded when computing signature."""
        data = {
            "version": "3.0",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "signature": "old_signature",
        }

        # Compute new signature
        new_sig = checkpoint_with_key._compute_signature(data)

        # The computed signature should match what we'd get without the signature field
        data_without_sig = {k: v for k, v in data.items() if k != "signature"}
        expected_sig = checkpoint_with_key._compute_signature(data_without_sig)

        assert new_sig == expected_sig

    def test_save_includes_signature(self, temp_project_dir, monkeypatch):
        """Test that save() includes signature in saved checkpoint."""
        monkeypatch.setenv("MATCHER_CHECKPOINT_SECRET", "test-secret-key-12345")
        checkpoint = CheckpointManager(temp_project_dir, config_hash="testhash")

        # Initialize checkpoint data properly
        checkpoint.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            config_hash="testhash"
        )
        checkpoint.data.last_completed_stage = "MATCH"
        checkpoint._atomic_save(force_rotate=True, force_full=True)

        # Read saved checkpoint (may be compressed)
        with open(checkpoint.checkpoint_path, 'rb') as f:
            content = f.read()

        # Handle compression (magic bytes 0x1f 0x8b)
        if len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b:
            content = gzip.decompress(content)

        saved_data = json.loads(content.decode('utf-8'))

        # Signature should be present
        assert "signature" in saved_data
        assert saved_data["signature"] != ""
        assert len(saved_data["signature"]) == 64

    def test_signature_stats_tracking(self, checkpoint_with_key):
        """Test that signature statistics are tracked."""
        # Initially zero
        stats = checkpoint_with_key._signature_stats
        assert stats.get("signatures_computed", 0) == 0
        assert stats.get("signatures_verified", 0) == 0

        # Compute signature
        data = {"version": "3.0"}
        checkpoint_with_key._compute_signature(data)

        stats = checkpoint_with_key._signature_stats
        assert stats.get("signatures_computed", 0) == 1

        # Verify valid signature
        data["signature"] = checkpoint_with_key._compute_signature(data)
        checkpoint_with_key.verify_signature(data)

        stats = checkpoint_with_key._signature_stats
        assert stats.get("signatures_verified", 0) == 1

    def test_get_checkpoint_stats_includes_signature(self, checkpoint_with_key):
        """Test that get_checkpoint_stats includes signature stats."""
        stats = checkpoint_with_key.get_checkpoint_stats()

        assert "signature_stats" in stats
        assert stats["signature_stats"]["enabled"] is True
        assert stats["signature_stats"]["key_configured"] is True
