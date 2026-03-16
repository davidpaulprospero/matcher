#!/usr/bin/env python3
"""
Tests for FAISS index build time logging in src/embeddings.py

Tests:
- Timing is logged when building FAISS index
- Log message includes vector count
- Log message includes dimension
- Log message includes index type (flat vs ivf)
- Timing is in milliseconds format
- Log level is INFO
- 100+ vectors scenario
"""

import sys
import re
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestBuildEmbeddingIndexTimingLogged:
    """Test that build_embedding_index logs timing information"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with indexing settings"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        config.indexing.ivf_nlist = 100
        config.indexing.ivf_nprobe = 10
        return config

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings array (100 vectors, 384 dimensions)"""
        import numpy as np
        return np.random.rand(100, 384).astype('float32')

    @pytest.mark.fast
    def test_timing_logged_for_flat_index(self, mock_config, mock_embeddings, caplog):
        """Test that timing is logged when building flat index"""
        from src.embeddings import build_embedding_index

        with caplog.at_level(logging.INFO):
            build_embedding_index(mock_embeddings, mock_config)

        # Check that timing message was logged
        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing message not logged"
        assert 'ms' in timing_messages[0].message, "Timing not in milliseconds"

    @pytest.mark.fast
    def test_timing_includes_vector_count(self, mock_config, mock_embeddings, caplog):
        """Test that timing message includes vector count"""
        from src.embeddings import build_embedding_index

        with caplog.at_level(logging.INFO):
            build_embedding_index(mock_embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert '100 vectors' in timing_messages[0].message, "Vector count not in log message"

    @pytest.mark.fast
    def test_timing_includes_dimension(self, mock_config, mock_embeddings, caplog):
        """Test that timing message includes vector dimension"""
        from src.embeddings import build_embedding_index

        with caplog.at_level(logging.INFO):
            build_embedding_index(mock_embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert 'dim=384' in timing_messages[0].message, "Dimension not in log message"

    @pytest.mark.fast
    def test_timing_includes_index_type_flat(self, mock_config, mock_embeddings, caplog):
        """Test that timing message includes index type (flat)"""
        from src.embeddings import build_embedding_index

        mock_config.indexing.index_type = 'flat'

        with caplog.at_level(logging.INFO):
            build_embedding_index(mock_embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert '(flat)' in timing_messages[0].message, "Index type 'flat' not in log message"

    @pytest.mark.fast
    def test_timing_includes_index_type_ivf(self, mock_config, caplog):
        """Test that timing message includes index type (ivf)"""
        import numpy as np
        from src.embeddings import build_embedding_index

        # IVF requires more vectors for training (at least 10 for nlist calculation)
        embeddings_ivf = np.random.rand(200, 384).astype('float32')
        mock_config.indexing.index_type = 'ivf'

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings_ivf, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing message not logged for IVF"
        assert '(ivf)' in timing_messages[0].message, "Index type 'ivf' not in log message"

    @pytest.mark.fast
    def test_timing_log_level_is_info(self, mock_config, mock_embeddings, caplog):
        """Test that timing is logged at INFO level"""
        from src.embeddings import build_embedding_index

        with caplog.at_level(logging.DEBUG):
            build_embedding_index(mock_embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert timing_messages[0].levelno == logging.INFO, "Timing not logged at INFO level"


class TestTimingFormat:
    """Test that timing format follows expected pattern"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        return config

    @pytest.mark.fast
    def test_timing_format_milliseconds(self, mock_config, caplog):
        """Test that timing is in 'X.Xms' format"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(50, 256).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        # Match pattern like "in 0.5ms" or "in 12.3ms"
        assert re.search(r'in \d+\.?\d*ms', timing_messages[0].message), \
            f"Timing format incorrect: {timing_messages[0].message}"

    @pytest.mark.fast
    def test_timing_is_positive_value(self, mock_config, caplog):
        """Test that logged timing is a positive number"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(50, 256).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        # Extract timing value
        match = re.search(r'in (\d+\.?\d*)ms', timing_messages[0].message)
        assert match, "Could not extract timing value"
        timing_value = float(match.group(1))
        assert timing_value >= 0, f"Timing should be non-negative, got {timing_value}"


class TestTimingWith100PlusVectors:
    """Test timing with 100+ vectors as specified in acceptance criteria"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        return config

    @pytest.mark.fast
    def test_timing_logged_for_100_vectors(self, mock_config, caplog):
        """Test that timing is logged for exactly 100 vectors"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(100, 768).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing not logged for 100 vectors"
        assert '100 vectors' in timing_messages[0].message

    @pytest.mark.fast
    def test_timing_logged_for_500_vectors(self, mock_config, caplog):
        """Test that timing is logged for 500 vectors"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(500, 768).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing not logged for 500 vectors"
        assert '500 vectors' in timing_messages[0].message

    @pytest.mark.fast
    def test_timing_logged_for_1000_vectors(self, mock_config, caplog):
        """Test that timing is logged for 1000 vectors"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(1000, 768).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing not logged for 1000 vectors"
        assert '1000 vectors' in timing_messages[0].message


class TestDifferentDimensions:
    """Test timing logging with different embedding dimensions"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        return config

    @pytest.mark.fast
    def test_timing_with_768_dim(self, mock_config, caplog):
        """Test timing with 768-dimensional embeddings (BERT-style)"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(50, 768).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert 'dim=768' in timing_messages[0].message

    @pytest.mark.fast
    def test_timing_with_1536_dim(self, mock_config, caplog):
        """Test timing with 1536-dimensional embeddings (OpenAI ada-002)"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(50, 1536).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert 'dim=1536' in timing_messages[0].message

    @pytest.mark.fast
    def test_timing_with_384_dim(self, mock_config, caplog):
        """Test timing with 384-dimensional embeddings (MiniLM)"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(50, 384).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert 'dim=384' in timing_messages[0].message


class TestNoTimingWhenFaissDisabled:
    """Test that no timing is logged when FAISS is disabled"""

    @pytest.mark.fast
    def test_no_timing_when_use_faiss_false(self, caplog):
        """Test that no timing is logged when use_faiss=False"""
        import numpy as np
        from src.embeddings import build_embedding_index

        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = False

        embeddings = np.random.rand(100, 384).astype('float32')

        with caplog.at_level(logging.INFO):
            result = build_embedding_index(embeddings, config)

        assert result is None, "Should return None when FAISS disabled"
        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) == 0, "Should not log timing when FAISS disabled"


class TestTimingWithListInput:
    """Test timing logging when embeddings are provided as list (not numpy array)"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        return config

    @pytest.mark.fast
    def test_timing_with_list_embeddings(self, mock_config, caplog):
        """Test timing is logged when embeddings are a list of lists"""
        from src.embeddings import build_embedding_index

        # Create embeddings as list of lists (not numpy)
        embeddings = [[0.1] * 256 for _ in range(100)]

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1, "Timing not logged for list embeddings"
        assert '100 vectors' in timing_messages[0].message
        assert 'dim=256' in timing_messages[0].message


class TestIVFIndexTiming:
    """Test timing for IVF index type specifically"""

    @pytest.fixture
    def mock_config_ivf(self):
        """Create mock config for IVF index"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'ivf'
        config.indexing.ivf_nlist = 10
        config.indexing.ivf_nprobe = 5
        return config

    @pytest.mark.fast
    def test_ivf_timing_includes_training_time(self, mock_config_ivf, caplog):
        """Test that IVF timing includes training phase"""
        import numpy as np
        from src.embeddings import build_embedding_index

        # IVF needs enough vectors for training
        embeddings = np.random.rand(500, 384).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config_ivf)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        assert len(timing_messages) >= 1
        # IVF should still show timing (training + add)
        assert 'ms' in timing_messages[0].message


class TestLogMessageFormat:
    """Test the complete log message format"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'flat'
        return config

    @pytest.mark.fast
    def test_log_message_contains_all_elements(self, mock_config, caplog):
        """Test that log message contains checkmark, index type, vectors, dim, and timing"""
        import numpy as np
        from src.embeddings import build_embedding_index

        embeddings = np.random.rand(150, 512).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, mock_config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        msg = timing_messages[0].message

        assert '✓' in msg, "Checkmark missing from log message"
        assert '(flat)' in msg, "Index type missing from log message"
        assert '150 vectors' in msg, "Vector count missing from log message"
        assert 'dim=512' in msg, "Dimension missing from log message"
        assert 'ms' in msg, "Timing missing from log message"

    @pytest.mark.fast
    def test_log_message_format_ivf(self, caplog):
        """Test log message format for IVF index"""
        import numpy as np
        from src.embeddings import build_embedding_index

        config = MagicMock()
        config.indexing = MagicMock()
        config.indexing.use_faiss = True
        config.indexing.index_type = 'ivf'
        config.indexing.ivf_nlist = 10
        config.indexing.ivf_nprobe = 5

        embeddings = np.random.rand(200, 256).astype('float32')

        with caplog.at_level(logging.INFO):
            build_embedding_index(embeddings, config)

        timing_messages = [r for r in caplog.records if 'Built FAISS index' in r.message]
        msg = timing_messages[0].message

        assert '(ivf)' in msg, "Index type 'ivf' missing from log message"
        assert '200 vectors' in msg, "Vector count missing from log message"
        assert 'dim=256' in msg, "Dimension missing from log message"
