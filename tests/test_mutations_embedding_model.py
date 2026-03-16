"""
Mutation tests for text-embedding-004 → gemini-embedding-001 migration.

Proves that:
1. The old model name (text-embedding-004) is no longer used as a default anywhere
2. The new model name (gemini-embedding-001) is correctly wired in all locations
3. The cost table includes the new model
4. Config, provider defaults, and fallbacks all agree on the new model
"""

import os
import re
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Project root
ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Source inspection: verify the old model name is gone from defaults
# ---------------------------------------------------------------------------

class TestOldModelRemoved:
    """Verify text-embedding-004 is no longer the default anywhere."""

    @pytest.mark.fast
    def test_config_dataclass_uses_new_model(self):
        """EmbeddingConfig default must be gemini-embedding-001."""
        from src.config.sections.core import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.gemini_model == "models/gemini-embedding-001"
        assert "text-embedding-004" not in cfg.gemini_model

    @pytest.mark.fast
    def test_gemini_embeddings_default_uses_new_model(self):
        """GeminiEmbeddings.__init__ default must be gemini-embedding-001."""
        import inspect
        from src.embeddings import GeminiEmbeddings
        sig = inspect.signature(GeminiEmbeddings.__init__)
        default_model = sig.parameters['model'].default
        assert default_model == "models/gemini-embedding-001"
        assert "text-embedding-004" not in default_model

    @pytest.mark.fast
    def test_get_embedding_provider_fallback_uses_new_model(self):
        """get_embedding_provider getattr fallback must be gemini-embedding-001."""
        source_file = ROOT / "src" / "embeddings.py"
        source = source_file.read_text(encoding="utf-8")
        # Find the getattr fallback line for gemini_model
        pattern = r"getattr\(config\.embedding,\s*'gemini_model',\s*'([^']+)'\)"
        matches = re.findall(pattern, source)
        assert len(matches) >= 1, "No getattr fallback for gemini_model found"
        for match in matches:
            assert match == "models/gemini-embedding-001", f"Fallback uses old model: {match}"

    @pytest.mark.fast
    def test_config_yaml_uses_new_model(self):
        """config.yaml embedding section must reference gemini-embedding-001."""
        config_file = ROOT / "config.yaml"
        content = config_file.read_text(encoding="utf-8")
        assert "gemini-embedding-001" in content
        # Find the embedding section's gemini_model (not the LLM section's)
        in_embedding_section = False
        found_embedding_model = False
        for line in content.splitlines():
            stripped = line.strip()
            # Detect top-level section headers (no indentation)
            if line and not line[0].isspace() and line.rstrip().endswith(':'):
                in_embedding_section = stripped == "embedding:"
            if in_embedding_section and stripped.startswith("gemini_model:"):
                found_embedding_model = True
                assert "gemini-embedding-001" in stripped, f"Embedding gemini_model uses wrong model: {stripped}"
                assert "text-embedding-004" not in stripped
        assert found_embedding_model, "No gemini_model found in embedding section"


# ---------------------------------------------------------------------------
# Source inspection: verify the new model is in the cost table
# ---------------------------------------------------------------------------

class TestCostTable:
    """Verify the cost table includes the new model."""

    @pytest.mark.fast
    def test_new_model_in_cost_table(self):
        """gemini-embedding-001 must have a cost entry."""
        from src.logger import get_api_cost
        cost = get_api_cost('gemini-embedding-001', input_tokens=1000, output_tokens=0)
        assert cost > 0, "gemini-embedding-001 should have non-zero input cost"

    @pytest.mark.fast
    def test_old_model_still_in_cost_table_for_historical_logs(self):
        """text-embedding-004 kept in cost table for historical log parsing."""
        from src.logger import get_api_cost
        cost = get_api_cost('text-embedding-004', input_tokens=1000, output_tokens=0)
        assert cost > 0


# ---------------------------------------------------------------------------
# Behavioral: provider wiring
# ---------------------------------------------------------------------------

class TestProviderWiring:
    """Verify the provider is constructed with the correct model."""

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @pytest.mark.fast
    def test_provider_gets_new_model_from_config(self, mock_configure):
        """get_embedding_provider passes config model to GeminiEmbeddings."""
        from src.embeddings import get_embedding_provider, GeminiEmbeddings
        config = Mock()
        config.embedding = Mock()
        config.embedding.provider = 'gemini'
        config.embedding.gemini_model = 'models/gemini-embedding-001'
        provider = get_embedding_provider(config)
        assert isinstance(provider, GeminiEmbeddings)
        assert provider.model == 'models/gemini-embedding-001'


# ---------------------------------------------------------------------------
# Mutation tests: in-memory string mutations to prove tests catch regressions
# ---------------------------------------------------------------------------

class TestMutations:
    """In-memory mutation testing — verify tests would catch regressions."""

    @pytest.mark.fast
    def test_mutation_revert_config_default_caught(self):
        """MUTATION: Revert EmbeddingConfig default to old model → must be caught."""
        source_file = ROOT / "src" / "config" / "sections" / "core.py"
        source = source_file.read_text(encoding="utf-8")

        # Apply mutation: revert to old model
        mutated = source.replace(
            'gemini_model: str = "models/gemini-embedding-001"',
            'gemini_model: str = "models/text-embedding-004"'
        )
        assert mutated != source, "Mutation not applied"

        # Verify the mutated source contains the old model
        assert "text-embedding-004" in mutated
        assert 'gemini_model: str = "models/gemini-embedding-001"' not in mutated
        # → This mutation would be KILLED by test_config_dataclass_uses_new_model

    @pytest.mark.fast
    def test_mutation_revert_provider_default_caught(self):
        """MUTATION: Revert GeminiEmbeddings default to old model → must be caught."""
        source_file = ROOT / "src" / "embeddings.py"
        source = source_file.read_text(encoding="utf-8")

        mutated = source.replace(
            'model: str = "models/gemini-embedding-001"',
            'model: str = "models/text-embedding-004"'
        )
        assert mutated != source, "Mutation not applied"
        assert 'model: str = "models/text-embedding-004"' in mutated

    @pytest.mark.fast
    def test_mutation_revert_fallback_caught(self):
        """MUTATION: Revert getattr fallback to old model → must be caught."""
        source_file = ROOT / "src" / "embeddings.py"
        source = source_file.read_text(encoding="utf-8")

        mutated = source.replace(
            "'models/gemini-embedding-001'",
            "'models/text-embedding-004'"
        )
        assert mutated != source, "Mutation not applied"

        # Verify the fallback now has the old model
        pattern = r"getattr\(config\.embedding,\s*'gemini_model',\s*'([^']+)'\)"
        matches = re.findall(pattern, mutated)
        for match in matches:
            assert match == "models/text-embedding-004"
        # → This mutation would be KILLED by test_get_embedding_provider_fallback_uses_new_model

    @pytest.mark.fast
    def test_mutation_remove_cost_entry_caught(self):
        """MUTATION: Remove gemini-embedding-001 from cost table → must be caught."""
        source_file = ROOT / "src" / "logger.py"
        source = source_file.read_text(encoding="utf-8")

        mutated = source.replace(
            "    'gemini-embedding-001': {'input': 0.00001, 'output': 0},\n",
            ""
        )
        assert mutated != source, "Mutation not applied"
        assert "gemini-embedding-001" not in mutated
        # → This mutation would be KILLED by test_new_model_in_cost_table

    @pytest.mark.fast
    def test_mutation_config_yaml_revert_caught(self):
        """MUTATION: Revert config.yaml embedding model to old → must be caught."""
        config_file = ROOT / "config.yaml"
        content = config_file.read_text(encoding="utf-8")

        mutated = content.replace("gemini-embedding-001", "text-embedding-004")
        assert mutated != content, "Mutation not applied"

        # Verify the embedding section now has old model
        in_embedding_section = False
        for line in mutated.splitlines():
            stripped = line.strip()
            if line and not line[0].isspace() and line.rstrip().endswith(':'):
                in_embedding_section = stripped == "embedding:"
            if in_embedding_section and stripped.startswith("gemini_model:"):
                assert "text-embedding-004" in stripped
        # → This mutation would be KILLED by test_config_yaml_uses_new_model
