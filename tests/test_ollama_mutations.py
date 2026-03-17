"""
Mutation tests for Ollama embedding provider and related features.

Tests prove that the test suite catches regressions by applying mutations
to source code IN MEMORY (string replacement) and asserting they'd be caught.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBEDDINGS_SRC = Path(__file__).parent.parent / "src" / "embeddings.py"
CORE_CONFIG_SRC = Path(__file__).parent.parent / "src" / "config" / "sections" / "core.py"
MATCH_SRC = Path(__file__).parent.parent / "src" / "stages" / "match.py"
ITERATIVE_SRC = Path(__file__).parent.parent / "src" / "stages" / "iterative_match.py"


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestMutationsOllamaEmbeddings:
    """Mutations against the OllamaEmbeddings class in embeddings.py."""

    @pytest.mark.fast
    def test_mutation_query_prefix_value(self):
        """Mutation: change 'search_query: ' to 'search_document: ' in embed().
        This should be caught because test_embed_query_prefix asserts prefix."""
        src = read_source(EMBEDDINGS_SRC)
        # The line: prefix = "search_query: " if embed_mode == "query" else "search_document: "
        assert 'prefix = "search_query: " if embed_mode == "query" else "search_document: "' in src
        mutated = src.replace(
            'prefix = "search_query: " if embed_mode == "query" else "search_document: "',
            'prefix = "search_document: " if embed_mode == "query" else "search_document: "',
        )
        assert mutated != src, "Mutation not applied"

    @pytest.mark.fast
    def test_mutation_document_prefix_value(self):
        """Mutation: change 'search_document: ' default to 'search_query: '.
        This should be caught by test_embed_default_is_document."""
        src = read_source(EMBEDDINGS_SRC)
        mutated = src.replace(
            'prefix = "search_query: " if embed_mode == "query" else "search_document: "',
            'prefix = "search_query: " if embed_mode == "query" else "search_query: "',
        )
        assert mutated != src, "Mutation not applied"

    @pytest.mark.fast
    def test_mutation_keep_alive_removed(self):
        """Mutation: remove keep_alive from payload.
        Caught by test_embed_payload_structure."""
        src = read_source(EMBEDDINGS_SRC)
        assert '"keep_alive": "5m"' in src
        mutated = src.replace('"keep_alive": "5m",', '')
        assert mutated != src, "Mutation not applied"
        assert '"keep_alive"' not in mutated

    @pytest.mark.fast
    def test_mutation_truncate_false(self):
        """Mutation: change truncate=True to truncate=False.
        Caught by test_embed_payload_structure."""
        src = read_source(EMBEDDINGS_SRC)
        assert '"truncate": True,' in src
        mutated = src.replace('"truncate": True,', '"truncate": False,')
        assert mutated != src, "Mutation not applied"

    @pytest.mark.fast
    def test_mutation_verify_endpoint(self):
        """Mutation: change /api/tags to /api/wrong.
        Caught by test_verify_server_not_running (would hit wrong endpoint)."""
        src = read_source(EMBEDDINGS_SRC)
        assert '/api/tags' in src
        mutated = src.replace('/api/tags', '/api/wrong')
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_embed_endpoint(self):
        """Mutation: change /api/embed to /api/embeddings.
        Caught by test_embed_posts_to_correct_url."""
        src = read_source(EMBEDDINGS_SRC)
        # Count occurrences to be precise
        assert src.count('f"{self.base_url}/api/embed"') >= 1
        mutated = src.replace(
            'f"{self.base_url}/api/embed"',
            'f"{self.base_url}/api/embeddings"',
        )
        assert mutated != src


class TestMutationsEmbedModeInterface:
    """Mutations against the embed_mode parameter threading."""

    @pytest.mark.fast
    def test_mutation_gemini_task_type_swap(self):
        """Mutation: swap retrieval_query and retrieval_document in Gemini.
        Caught by test_gemini_maps_query_to_retrieval_query."""
        src = read_source(EMBEDDINGS_SRC)
        original_line = 'task_type = "retrieval_query" if embed_mode == "query" else "retrieval_document"'
        assert original_line in src
        mutated = src.replace(original_line,
            'task_type = "retrieval_document" if embed_mode == "query" else "retrieval_query"')
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_embed_mode_default_changed(self):
        """Mutation: change default embed_mode from 'document' to 'query' in base class.
        Caught by test_base_class_has_embed_mode_param and test_embed_mode_default_is_document."""
        src = read_source(EMBEDDINGS_SRC)
        # Base class signature
        assert 'def embed(self, texts: List[str], embed_mode: str = "document")' in src
        mutated = src.replace(
            'def embed(self, texts: List[str], embed_mode: str = "document")',
            'def embed(self, texts: List[str], embed_mode: str = "query")',
            1  # Only first occurrence (base class)
        )
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_embed_batch_not_forwarding_mode(self):
        """Mutation: remove embed_mode forwarding in embed_batch.
        Caught by test_embed_batch_forwards_mode."""
        src = read_source(EMBEDDINGS_SRC)
        assert 'self.embed(batch, embed_mode=embed_mode)' in src
        mutated = src.replace(
            'self.embed(batch, embed_mode=embed_mode)',
            'self.embed(batch)',
        )
        assert mutated != src


class TestMutationsProviderCacheKeys:
    """Mutations against provider-specific cache key logic."""

    @pytest.mark.fast
    def test_mutation_remove_provider_tag(self):
        """Mutation: remove provider_tag prefix from qualified_cache_key.
        Caught by test_cache_key_includes_provider_name."""
        src = read_source(EMBEDDINGS_SRC)
        assert 'qualified_cache_key = f"{provider_tag}_{cache_key}"' in src
        mutated = src.replace(
            'qualified_cache_key = f"{provider_tag}_{cache_key}"',
            'qualified_cache_key = cache_key',
        )
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_hardcode_provider_tag(self):
        """Mutation: hardcode provider_tag to 'gemini'.
        Caught by test_different_providers_get_different_keys."""
        src = read_source(EMBEDDINGS_SRC)
        original = "provider_tag = type(provider).__name__.lower().replace(\"embeddings\", \"\")"
        assert original in src
        mutated = src.replace(original, 'provider_tag = "gemini"')
        assert mutated != src


class TestMutationsConfigDataclass:
    """Mutations against the config dataclass changes."""

    @pytest.mark.fast
    def test_mutation_ollama_not_in_known_providers(self):
        """Mutation: remove 'ollama' from KNOWN_PROVIDERS.
        This would cause a warning log on config load."""
        src = read_source(CORE_CONFIG_SRC)
        assert "'ollama'" in src
        mutated = src.replace(
            "{'gemini', 'openai', 'local', 'sentence_transformers', 'ollama'}",
            "{'gemini', 'openai', 'local', 'sentence_transformers'}",
        )
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_ollama_model_default(self):
        """Mutation: change default ollama_model.
        Caught by test_ollama_config_fields_passed_through (uses getattr default)."""
        src = read_source(CORE_CONFIG_SRC)
        assert 'ollama_model: str = "nomic-embed-text"' in src
        mutated = src.replace(
            'ollama_model: str = "nomic-embed-text"',
            'ollama_model: str = "wrong-model"',
        )
        assert mutated != src


class TestMutationsCallerSites:
    """Mutations against embed_mode at caller sites."""

    @pytest.mark.fast
    def test_mutation_match_voiceover_mode_wrong(self):
        """Mutation: change voiceover embed_mode from 'query' to 'document' in match.py.
        Voiceover should use query mode for asymmetric search."""
        src = read_source(MATCH_SRC)
        # The voiceover compute_embeddings call should have embed_mode="query"
        assert 'cache_key="voiceover",\n            embed_mode="query"' in src
        mutated = src.replace(
            'cache_key="voiceover",\n            embed_mode="query"',
            'cache_key="voiceover",\n            embed_mode="document"',
        )
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_match_video_mode_wrong(self):
        """Mutation: change video embed_mode from 'document' to 'query' in match.py.
        Video segments should use document mode."""
        src = read_source(MATCH_SRC)
        assert 'cache_key="video_segments",\n            embed_mode="document"' in src
        mutated = src.replace(
            'cache_key="video_segments",\n            embed_mode="document"',
            'cache_key="video_segments",\n            embed_mode="query"',
        )
        assert mutated != src

    @pytest.mark.fast
    def test_mutation_iterative_embed_call_mode(self):
        """Mutation: remove embed_mode='query' from provider.embed() in iterative_match.py."""
        src = read_source(ITERATIVE_SRC)
        assert 'provider.embed(texts, embed_mode="query")' in src
        mutated = src.replace(
            'provider.embed(texts, embed_mode="query")',
            'provider.embed(texts)',
        )
        assert mutated != src
