"""
Tests for src/transcription/whisper_client.py

Tests the thread-safe WhisperClient with GPU lock management,
model initialization with device detection, and transcription.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import threading

from src.transcription.whisper_client import (
    WhisperClient, cleanup_model, get_available_gpu_memory,
    _select_model_for_memory, MODEL_MEMORY_REQUIREMENTS, MODEL_DOWNGRADE_ORDER
)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global model state before each test"""
    import src.transcription.whisper_client as wc
    wc._shared_model = None
    wc._model_config = {}
    yield
    # Cleanup after test
    wc._shared_model = None
    wc._model_config = {}


class TestWhisperClientInit:
    """Test WhisperClient initialization"""

    @pytest.mark.fast
    def test_init_defaults(self):
        """Test initialization with default parameters"""
        client = WhisperClient()
        assert client.model_name == "base"
        assert client.compute_type == "auto"

    @pytest.mark.fast
    def test_init_custom_model(self):
        """Test initialization with custom model name"""
        client = WhisperClient(model_name="medium")
        assert client.model_name == "medium"
        assert client.compute_type == "auto"

    @pytest.mark.fast
    def test_init_custom_compute_type(self):
        """Test initialization with custom compute type"""
        client = WhisperClient(model_name="small", compute_type="int8")
        assert client.model_name == "small"
        assert client.compute_type == "int8"


class TestModelInitialization:
    """Test WhisperModel initialization and caching"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_with_cuda(self, mock_logger):
        """Test model initialization with CUDA available"""
        mock_model_instance = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        # Mock GPU properties properly
        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3  # 8GB
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        # Create mock WhisperModel class
        MockWhisperModel = Mock(return_value=mock_model_instance)

        # Mock imports inside get_model()
        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            # For other imports, use real import
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto")
            model = client.get_model()

            # Should return the mocked model
            assert model is mock_model_instance

            # Should have called WhisperModel with CUDA device
            MockWhisperModel.assert_called_once_with(
                "base",
                device="cuda",
                compute_type="float16",  # auto → float16 with CUDA
                num_workers=1,
                cpu_threads=4
            )

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_without_cuda(self, mock_logger):
        """Test model initialization without CUDA (CPU fallback)"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        # Mock imports - torch raises ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto")
            model = client.get_model()

            # Should return the mocked model
            assert model is mock_model_instance

            # Should have called WhisperModel with CPU device
            MockWhisperModel.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",  # auto → int8 without CUDA
                num_workers=1,
                cpu_threads=4
            )

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_explicit_int8(self, mock_logger):
        """Test model initialization with explicit int8 compute type"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="small", compute_type="int8")
            model = client.get_model()

            # Should use float16 with CUDA (int8 mode also checks CUDA)
            MockWhisperModel.assert_called_once_with(
                "small",
                device="cuda",
                compute_type="float16",
                num_workers=1,
                cpu_threads=4
            )

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_cached(self, mock_logger):
        """Test model caching (should not recreate on subsequent calls)"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto")

            # First call initializes
            model1 = client.get_model()
            assert MockWhisperModel.call_count == 1

            # Second call uses cached model
            model2 = client.get_model()
            assert MockWhisperModel.call_count == 1  # Still 1, not 2
            assert model1 is model2

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_reinitialize_on_config_change(self, mock_logger):
        """Test model reinitialization when config changes"""
        mock_model1 = Mock()
        mock_model2 = Mock()
        MockWhisperModel = Mock(side_effect=[mock_model1, mock_model2])

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # First client with "base" model
            client1 = WhisperClient(model_name="base", compute_type="auto")
            model1 = client1.get_model()
            assert MockWhisperModel.call_count == 1

            # Second client with different model should reinitialize
            client2 = WhisperClient(model_name="medium", compute_type="auto")
            model2 = client2.get_model()
            assert MockWhisperModel.call_count == 2
            assert model1 is not model2

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_initialization_error(self, mock_logger):
        """Test error handling during model initialization"""
        with patch('faster_whisper.WhisperModel', side_effect=RuntimeError("Model load failed")):
            client = WhisperClient()

            with pytest.raises(RuntimeError, match="Model load failed"):
                client.get_model()


class TestTranscription:
    """Test transcription with GPU lock"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_success(self, mock_logger):
        """Test successful transcription"""
        # Create mock segments
        mock_seg1 = Mock()
        mock_seg1.start = 0.0
        mock_seg1.end = 3.0
        mock_seg1.text = " First segment "
        mock_seg1.words = None

        mock_seg2 = Mock()
        mock_seg2.start = 3.0
        mock_seg2.end = 6.0
        mock_seg2.text = " Second segment "
        mock_seg2.words = None

        # Mock model.transcribe()
        mock_model = Mock()
        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], mock_info)

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3")

            assert len(result) == 2
            assert result[0]['start'] == 0.0
            assert result[0]['end'] == 3.0
            assert result[0]['text'] == "First segment"  # Stripped
            assert result[1]['start'] == 3.0
            assert result[1]['end'] == 6.0
            assert result[1]['text'] == "Second segment"

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_with_word_timestamps(self, mock_logger):
        """Test transcription with word-level timestamps"""
        # Create mock word
        mock_word = Mock()
        mock_word.word = "Hello"
        mock_word.start = 0.0
        mock_word.end = 0.5

        # Create mock segment with words
        mock_seg = Mock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "Hello world"
        mock_seg.words = [mock_word]

        mock_model = Mock()
        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3", word_timestamps=True)

            assert len(result) == 1
            assert 'words' in result[0]
            assert len(result[0]['words']) == 1
            assert result[0]['words'][0]['word'] == "Hello"
            assert result[0]['words'][0]['start'] == 0.0

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_with_language(self, mock_logger):
        """Test transcription with explicit language"""
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            client.transcribe("/path/to/audio.mp3", language="en")

            # Should pass language to model.transcribe()
            call_args = mock_model.transcribe.call_args
            assert call_args[1]['language'] == "en"

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_vad_parameters(self, mock_logger):
        """Test transcription with VAD parameters"""
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            client.transcribe(
                "/path/to/audio.mp3",
                vad_filter=True,
                min_silence_duration_ms=300,
                speech_pad_ms=50
            )

            call_args = mock_model.transcribe.call_args
            assert call_args[1]['vad_filter'] is True
            assert call_args[1]['vad_parameters']['min_silence_duration_ms'] == 300
            assert call_args[1]['vad_parameters']['speech_pad_ms'] == 50

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_error_handling(self, mock_logger):
        """Test error handling during transcription"""
        mock_model = Mock()
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3")

            # Should return empty list on error
            assert result == []

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_empty_result(self, mock_logger):
        """Test transcription with no segments returned"""
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3")

            assert result == []


class TestGPULocking:
    """Test GPU lock thread safety"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_gpu_lock_acquired(self, mock_logger):
        """Test that GPU lock is held during transcription.

        Note: model.transcribe() runs inside a ThreadPoolExecutor (US-79-002 timeout guard),
        so we verify the lock is held by checking it's NOT freely acquirable from a separate thread.
        """
        import src.transcription.whisper_client as wc

        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()

            # Track lock state - check from a separate thread that lock is NOT free
            lock_was_held = False

            def check_lock(*args, **kwargs):
                nonlocal lock_was_held
                # Try to acquire lock from this thread (non-blocking)
                # If the outer thread holds it, acquire() returns False
                acquired = wc._gpu_lock.acquire(blocking=False)
                if acquired:
                    # Lock was free — release and mark as not held
                    wc._gpu_lock.release()
                    lock_was_held = False
                else:
                    # Lock held by another thread (the caller) — expected
                    lock_was_held = True
                return ([], Mock())

            mock_model.transcribe.side_effect = check_lock

            client.transcribe("/path/to/audio.mp3")

            # Lock should have been held during transcription
            assert lock_was_held


class TestCleanup:
    """Test model cleanup and memory management"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_cleanup_with_model_loaded(self, mock_logger):
        """Test cleanup when model is loaded"""
        import src.transcription.whisper_client as wc

        mock_model = Mock()
        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            client.get_model()

            # Model should be loaded
            assert wc._shared_model is not None

            # Cleanup
            client.cleanup()

            # Model should be unloaded
            assert wc._shared_model is None
            assert wc._model_config == {}

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_cleanup_without_model(self, mock_logger):
        """Test cleanup when no model is loaded"""
        import src.transcription.whisper_client as wc

        client = WhisperClient()

        # No model loaded
        assert wc._shared_model is None

        # Cleanup should not error
        client.cleanup()

        assert wc._shared_model is None

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_cleanup_with_cuda(self, mock_logger):
        """Test cleanup with CUDA cache clearing"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0
        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = gpu_props
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()

        # Save original __import__
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            # Use original import for everything else
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient()
            client.get_model()

            # Cleanup imports torch again, so need to ensure it returns mock_torch
            client.cleanup()

            # Should have cleared CUDA cache
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.synchronize.assert_called_once()

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_cleanup_model_function(self, mock_logger):
        """Test module-level cleanup_model() function"""
        import src.transcription.whisper_client as wc

        mock_model = Mock()
        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            client.get_model()

            # Use module-level cleanup function
            cleanup_model()

            # Model should be unloaded
            assert wc._shared_model is None


class TestThreadSafety:
    """Test thread safety of shared model access"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_concurrent_model_access(self, mock_logger):
        """Test that concurrent access doesn't create multiple models"""
        mock_model = Mock()
        call_count = 0

        def create_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_model

        with patch('faster_whisper.WhisperModel', side_effect=create_model):
            # Create multiple threads accessing the model simultaneously
            def get_model_thread():
                client = WhisperClient()
                client.get_model()

            threads = [threading.Thread(target=get_model_thread) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have only created model once despite 5 concurrent threads
            assert call_count == 1


class TestCoverageGaps:
    """Test coverage gaps for uncovered lines"""

    # Note: Line 71 (double-check pattern inside lock) is a defensive coding pattern
    # that's nearly impossible to test reliably without race conditions. It's covered
    # by the thread safety test in TestThreadSafety.test_concurrent_model_access.

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_cuda_not_available_lines_98_99(self, mock_logger):
        """Test lines 98-99: device='cpu', actual_compute='int8' when CUDA is False"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False  # CUDA not available
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.memory_reserved.return_value = 0

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch  # Torch is available but CUDA is not
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto")
            model = client.get_model()

            # Should use CPU with int8 when CUDA is not available
            MockWhisperModel.assert_called_once_with(
                "base",
                device="cpu",  # Line 98
                compute_type="int8",  # Line 99
                num_workers=1,
                cpu_threads=4
            )

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_cleanup_torch_import_error_lines_237_238(self, mock_logger):
        """Test lines 237-238: pass when torch ImportError during cleanup"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        original_import = builtins.__import__
        cleanup_called = [False]

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                # During cleanup, raise ImportError (lines 237-238)
                if cleanup_called[0]:
                    raise ImportError("torch not available during cleanup")
                else:
                    # During model init, return mock with CUDA unavailable
                    mock_torch = Mock()
                    mock_torch.cuda.is_available.return_value = False
                    return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient()
            client.get_model()

            # Mark that cleanup is now happening
            cleanup_called[0] = True

            # Cleanup should not error even when torch import fails
            client.cleanup()

            # Model should still be unloaded
            assert wc._shared_model is None


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_no_words_attribute(self, mock_logger):
        """Test segment without words attribute (shouldn't crash)"""
        mock_seg = Mock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "Test"
        # No words attribute
        del mock_seg.words

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_seg], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3", word_timestamps=True)

            # Should handle missing words gracefully
            assert len(result) == 1
            assert 'words' not in result[0]

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_empty_words_list(self, mock_logger):
        """Test segment with empty words list"""
        mock_seg = Mock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "Test"
        mock_seg.words = []  # Empty list

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_seg], Mock())

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient()
            result = client.transcribe("/path/to/audio.mp3", word_timestamps=True)

            # Should handle empty words list
            assert len(result) == 1
            assert 'words' not in result[0]


class TestMemoryLogging:
    """Test GPU memory logging during model init and cleanup"""

    @pytest.mark.fast
    def test_memory_logging_during_init_with_cuda(self):
        """Test that GPU memory is logged before/after model initialization (US-38-007)"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024  # 100MB
        mock_torch.cuda.memory_reserved.return_value = 200 * 1024 * 1024  # 200MB

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with patch('src.transcription.whisper_client.logger') as mock_logger:
                client = WhisperClient(model_name="base", compute_type="auto")
                client.get_model()

                # Verify memory logging was called
                log_calls = [str(c) for c in mock_logger.info.call_args_list]

                # Check for before init log
                before_log_found = any('before model init' in c for c in log_calls)
                assert before_log_found, f"Expected 'before model init' log. Calls: {log_calls}"

                # Check for after init log
                after_log_found = any('after model init' in c for c in log_calls)
                assert after_log_found, f"Expected 'after model init' log. Calls: {log_calls}"

                # Check for memory delta log
                delta_log_found = any('memory delta' in c.lower() for c in log_calls)
                assert delta_log_found, f"Expected memory delta log. Calls: {log_calls}"

    @pytest.mark.fast
    def test_memory_logging_during_init_without_cuda(self):
        """Test memory logging fallback when CUDA unavailable (US-38-007)"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with patch('src.transcription.whisper_client.logger') as mock_logger:
                client = WhisperClient(model_name="base", compute_type="auto")
                client.get_model()

                # Verify memory logging was called (with 0.0MB since no CUDA)
                log_calls = [str(c) for c in mock_logger.info.call_args_list]

                # Should still log memory (0.0MB values)
                before_log_found = any('before model init' in c for c in log_calls)
                assert before_log_found, f"Expected 'before model init' log. Calls: {log_calls}"

    @pytest.mark.fast
    def test_memory_logging_during_cleanup_with_cuda(self):
        """Test that GPU memory delta is logged during cleanup (US-38-007)"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True

        # Simulate memory usage: high before cleanup, low after
        memory_calls = [0]

        def get_memory_allocated():
            memory_calls[0] += 1
            # First call (before cleanup): 500MB, Second call (after cleanup): 50MB
            return 500 * 1024 * 1024 if memory_calls[0] <= 2 else 50 * 1024 * 1024

        mock_torch.cuda.memory_allocated.side_effect = get_memory_allocated
        mock_torch.cuda.memory_reserved.return_value = 600 * 1024 * 1024
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with patch('src.transcription.whisper_client.logger') as mock_logger:
                client = WhisperClient()
                client.get_model()

                # Reset mock logger to only capture cleanup logs
                mock_logger.reset_mock()

                # Cleanup
                client.cleanup()

                # Verify cleanup memory logging
                log_calls = [str(c) for c in mock_logger.info.call_args_list]

                # Check for before cleanup log
                before_log_found = any('before cleanup' in c for c in log_calls)
                assert before_log_found, f"Expected 'before cleanup' log. Calls: {log_calls}"

                # Check for after cleanup log
                after_log_found = any('after cleanup' in c for c in log_calls)
                assert after_log_found, f"Expected 'after cleanup' log. Calls: {log_calls}"

                # Check for memory freed log
                freed_log_found = any('freed by cleanup' in c.lower() for c in log_calls)
                assert freed_log_found, f"Expected 'freed by cleanup' log. Calls: {log_calls}"

    @pytest.mark.fast
    def test_get_gpu_memory_mb_returns_tuple(self):
        """Test _get_gpu_memory_mb helper function returns correct tuple (US-38-007)"""
        from src.transcription.whisper_client import _get_gpu_memory_mb

        # Test without mocking - should return (0.0, 0.0) if CUDA unavailable
        # or actual values if CUDA is available
        result = _get_gpu_memory_mb()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    @pytest.mark.fast
    def test_get_gpu_memory_mb_with_cuda_available(self):
        """Test _get_gpu_memory_mb with mocked CUDA (US-38-007)"""
        import builtins

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024  # 100MB
        mock_torch.cuda.memory_reserved.return_value = 200 * 1024 * 1024  # 200MB

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            from src.transcription import whisper_client
            # Force reimport to get mocked torch
            import importlib
            importlib.reload(whisper_client)

            result = whisper_client._get_gpu_memory_mb()

            # Should return approximately 100MB allocated, 200MB reserved
            assert result[0] == pytest.approx(100.0, abs=1.0)
            assert result[1] == pytest.approx(200.0, abs=1.0)

            # Restore module
            importlib.reload(whisper_client)


class TestGPUMemoryPreCheck:
    """Tests for GPU memory pre-check and automatic model downgrade (US-60-007)"""

    @pytest.mark.fast
    def test_get_available_gpu_memory_returns_float(self):
        """Test get_available_gpu_memory returns float even when CUDA unavailable"""
        result = get_available_gpu_memory()
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_get_available_gpu_memory_with_cuda(self):
        """Test get_available_gpu_memory with mocked CUDA"""
        import builtins

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 500 * 1024 * 1024  # 500MB used
        mock_torch.cuda.memory_reserved.return_value = 600 * 1024 * 1024  # 600MB reserved

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3  # 8GB total
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            from src.transcription import whisper_client
            import importlib
            importlib.reload(whisper_client)

            result = whisper_client.get_available_gpu_memory()

            # 8192MB total - 600MB reserved = ~7592MB available
            assert result == pytest.approx(8192 - 600, abs=10.0)

            importlib.reload(whisper_client)

    @pytest.mark.fast
    def test_model_memory_requirements_defined(self):
        """Test that model memory requirements are defined for all common models"""
        expected_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for model in expected_models:
            assert model in MODEL_MEMORY_REQUIREMENTS
            assert MODEL_MEMORY_REQUIREMENTS[model] > 0

    @pytest.mark.fast
    def test_model_downgrade_order_complete(self):
        """Test that model downgrade order includes all common models"""
        expected_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for model in expected_models:
            assert model in MODEL_DOWNGRADE_ORDER

    @pytest.mark.fast
    def test_select_model_for_memory_sufficient(self):
        """Test _select_model_for_memory returns requested model when memory sufficient"""
        # Request 'base' with 3000MB available (plenty of memory)
        result = _select_model_for_memory("base", available_memory_mb=3000, min_memory_mb=2000)
        assert result == "base"

    @pytest.mark.fast
    def test_select_model_for_memory_downgrade_medium_to_small(self):
        """Test automatic downgrade from medium to small when memory insufficient"""
        # Request 'medium' (needs ~2000MB) but only 1500MB available
        with patch('src.transcription.whisper_client.logger') as mock_logger:
            result = _select_model_for_memory("medium", available_memory_mb=1500, min_memory_mb=2000)

        assert result == "small"  # Should downgrade to small (needs ~1000MB)
        # Should have logged a warning about downgrade
        mock_logger.warning.assert_called()

    @pytest.mark.fast
    def test_select_model_for_memory_downgrade_large_to_base(self):
        """Test automatic downgrade from large to base when memory very limited"""
        # Request 'large' (needs ~3000MB) but only 800MB available
        with patch('src.transcription.whisper_client.logger') as mock_logger:
            result = _select_model_for_memory("large", available_memory_mb=800, min_memory_mb=2000)

        assert result == "base"  # Should downgrade to base (needs ~500MB)
        mock_logger.warning.assert_called()

    @pytest.mark.fast
    def test_select_model_for_memory_fallback_to_tiny(self):
        """Test fallback to tiny model when memory very low"""
        # Only 300MB available, even base won't fit
        with patch('src.transcription.whisper_client.logger') as mock_logger:
            result = _select_model_for_memory("medium", available_memory_mb=300, min_memory_mb=2000)

        assert result == "tiny"
        mock_logger.warning.assert_called()

    @pytest.mark.fast
    def test_select_model_for_memory_unknown_model(self):
        """Test handling of unknown model name"""
        # Unknown model should use default (assumed small enough)
        result = _select_model_for_memory("custom-model", available_memory_mb=3000, min_memory_mb=2000)
        # Unknown model with sufficient memory should work
        assert result == "custom-model"

    @pytest.mark.fast
    def test_whisper_client_init_with_memory_params(self):
        """Test WhisperClient accepts memory configuration parameters"""
        client = WhisperClient(
            model_name="medium",
            minimum_gpu_memory_mb=4000,
            auto_downgrade_model=True
        )
        assert client.model_name == "medium"
        assert client.minimum_gpu_memory_mb == 4000
        assert client.auto_downgrade_model is True

    @pytest.mark.fast
    def test_whisper_client_init_defaults(self):
        """Test WhisperClient default memory configuration"""
        client = WhisperClient()
        assert client.minimum_gpu_memory_mb == 2000
        assert client.auto_downgrade_model is True

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_performs_memory_check_and_downgrade(self, mock_logger):
        """Test get_model checks GPU memory and downgrades model automatically (US-60-007)"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 7000 * 1024 * 1024  # 7GB used
        mock_torch.cuda.memory_reserved.return_value = 7500 * 1024 * 1024  # 7.5GB reserved

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3  # 8GB total
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # Request large model but only ~692MB available (8192 - 7500)
            client = WhisperClient(
                model_name="large",
                minimum_gpu_memory_mb=2000,
                auto_downgrade_model=True
            )
            model = client.get_model()

            # Should have created model with downgraded size
            assert MockWhisperModel.call_count == 1
            call_args = MockWhisperModel.call_args
            # Model should be downgraded to base (500MB) since only ~692MB available
            assert call_args[0][0] == "base"

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_no_downgrade_when_disabled(self, mock_logger):
        """Test get_model doesn't downgrade when auto_downgrade_model=False"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024 * 1024
        mock_torch.cuda.memory_reserved.return_value = 200 * 1024 * 1024

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # Request large model with downgrade disabled
            client = WhisperClient(
                model_name="large",
                auto_downgrade_model=False  # Disable downgrade
            )
            model = client.get_model()

            # Should use requested model despite any memory concerns
            call_args = MockWhisperModel.call_args
            assert call_args[0][0] == "large"

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_get_model_warns_when_below_threshold(self, mock_logger):
        """Test get_model logs warning when GPU memory below threshold (US-60-007)"""
        import src.transcription.whisper_client as wc
        import builtins

        mock_model = Mock()
        MockWhisperModel = Mock(return_value=mock_model)

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 7000 * 1024 * 1024  # 7GB used
        mock_torch.cuda.memory_reserved.return_value = 7000 * 1024 * 1024

        gpu_props = Mock()
        gpu_props.total_memory = 8 * 1024**3  # 8GB total = 1192MB available
        mock_torch.cuda.get_device_properties.return_value = gpu_props

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                return mock_torch
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(
                model_name="base",
                minimum_gpu_memory_mb=2000  # Higher than available 1192MB
            )
            client.get_model()

            # Should have logged warning about low memory
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any('below threshold' in c for c in warning_calls)


class TestWhisperThreadingConfig:
    """Tests for configurable whisper_num_workers and whisper_cpu_threads (US-79-007)"""

    @pytest.mark.fast
    def test_config_default_whisper_num_workers(self):
        """Test TranscriptionConfig default whisper_num_workers is 1"""
        from src.config.sections.core import TranscriptionConfig
        config = TranscriptionConfig()
        assert config.whisper_num_workers == 1

    @pytest.mark.fast
    def test_config_default_whisper_cpu_threads(self):
        """Test TranscriptionConfig default whisper_cpu_threads is 4"""
        from src.config.sections.core import TranscriptionConfig
        config = TranscriptionConfig()
        assert config.whisper_cpu_threads == 4

    @pytest.mark.fast
    def test_config_custom_whisper_num_workers(self):
        """Test TranscriptionConfig accepts custom whisper_num_workers"""
        from src.config.sections.core import TranscriptionConfig
        config = TranscriptionConfig(whisper_num_workers=2)
        assert config.whisper_num_workers == 2

    @pytest.mark.fast
    def test_config_custom_whisper_cpu_threads(self):
        """Test TranscriptionConfig accepts custom whisper_cpu_threads"""
        from src.config.sections.core import TranscriptionConfig
        config = TranscriptionConfig(whisper_cpu_threads=8)
        assert config.whisper_cpu_threads == 8

    @pytest.mark.fast
    def test_config_rejects_whisper_num_workers_below_1(self):
        """Test TranscriptionConfig rejects whisper_num_workers < 1"""
        from src.config.sections.core import TranscriptionConfig
        with pytest.raises(ValueError, match="whisper_num_workers"):
            TranscriptionConfig(whisper_num_workers=0)

    @pytest.mark.fast
    def test_config_rejects_whisper_cpu_threads_below_1(self):
        """Test TranscriptionConfig rejects whisper_cpu_threads < 1"""
        from src.config.sections.core import TranscriptionConfig
        with pytest.raises(ValueError, match="whisper_cpu_threads"):
            TranscriptionConfig(whisper_cpu_threads=0)

    @pytest.mark.fast
    def test_whisper_client_accepts_num_workers(self):
        """Test WhisperClient accepts custom num_workers"""
        client = WhisperClient(num_workers=2)
        assert client.num_workers == 2

    @pytest.mark.fast
    def test_whisper_client_accepts_cpu_threads(self):
        """Test WhisperClient accepts custom cpu_threads"""
        client = WhisperClient(cpu_threads=8)
        assert client.cpu_threads == 8

    @pytest.mark.fast
    def test_whisper_client_default_num_workers(self):
        """Test WhisperClient default num_workers is 1"""
        client = WhisperClient()
        assert client.num_workers == 1

    @pytest.mark.fast
    def test_whisper_client_default_cpu_threads(self):
        """Test WhisperClient default cpu_threads is 4"""
        client = WhisperClient()
        assert client.cpu_threads == 4

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_custom_num_workers_passed_to_whisper_model(self, mock_logger):
        """Test custom whisper_num_workers value is passed to WhisperModel constructor"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto", num_workers=3)
            client.get_model()

            MockWhisperModel.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                num_workers=3,
                cpu_threads=4
            )

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_custom_cpu_threads_passed_to_whisper_model(self, mock_logger):
        """Test custom whisper_cpu_threads value is passed to WhisperModel constructor"""
        mock_model_instance = Mock()
        MockWhisperModel = Mock(return_value=mock_model_instance)

        def mock_import(name, *args, **kwargs):
            if name == 'faster_whisper':
                mock_module = Mock()
                mock_module.WhisperModel = MockWhisperModel
                return mock_module
            elif name == 'torch':
                raise ImportError("No torch")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            client = WhisperClient(model_name="base", compute_type="auto", cpu_threads=12)
            client.get_model()

            MockWhisperModel.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                num_workers=1,
                cpu_threads=12
            )


class TestGPUTranscriptionTimeout:
    """Tests for GPU transcription timeout guard (US-79-002)"""

    @pytest.mark.fast
    def test_init_default_timeout(self):
        """Test WhisperClient default gpu_transcription_timeout is 300s"""
        client = WhisperClient()
        assert client.gpu_transcription_timeout == 300

    @pytest.mark.fast
    def test_init_custom_timeout(self):
        """Test WhisperClient accepts custom gpu_transcription_timeout"""
        client = WhisperClient(gpu_transcription_timeout=60)
        assert client.gpu_transcription_timeout == 60

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_timeout_raises_transient_error(self, mock_logger):
        """Test that transcription exceeding timeout raises TransientTranscriptionError"""
        import time as time_module
        from src.transcription.exceptions import TransientTranscriptionError

        mock_model = Mock()

        # Simulate a hung transcription that blocks for longer than timeout
        def slow_transcribe(*args, **kwargs):
            time_module.sleep(5)  # Sleep longer than timeout
            return ([], Mock())

        mock_model.transcribe.side_effect = slow_transcribe

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            # Use very short timeout (1s) so test runs fast
            client = WhisperClient(gpu_transcription_timeout=1)

            with pytest.raises(TransientTranscriptionError, match="timed out"):
                client.transcribe("/path/to/audio.mp3")

        # Verify warning was logged with video_id, elapsed time, and timeout
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("timeout" in c.lower() for c in warning_calls)

    @patch('src.transcription.whisper_client.logger')
    @pytest.mark.fast
    def test_transcribe_within_timeout_returns_normally(self, mock_logger):
        """Test that transcription completing within timeout returns segments normally"""
        mock_seg = Mock()
        mock_seg.start = 0.0
        mock_seg.end = 3.0
        mock_seg.text = " Test segment "
        mock_seg.words = None

        mock_model = Mock()
        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        with patch('faster_whisper.WhisperModel', return_value=mock_model):
            client = WhisperClient(gpu_transcription_timeout=60)
            result = client.transcribe("/path/to/audio.mp3")

            assert len(result) == 1
            assert result[0]['start'] == 0.0
            assert result[0]['end'] == 3.0
            assert result[0]['text'] == "Test segment"

    @pytest.mark.fast
    def test_config_validation_rejects_below_30(self):
        """Test TranscriptionConfig rejects gpu_transcription_timeout < 30"""
        from src.config.sections.core import TranscriptionConfig

        with pytest.raises(ValueError, match="gpu_transcription_timeout"):
            TranscriptionConfig(gpu_transcription_timeout=10)

    @pytest.mark.fast
    def test_config_validation_accepts_30(self):
        """Test TranscriptionConfig accepts gpu_transcription_timeout = 30"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig(gpu_transcription_timeout=30)
        assert config.gpu_transcription_timeout == 30

    @pytest.mark.fast
    def test_config_default_timeout(self):
        """Test TranscriptionConfig default gpu_transcription_timeout is 300"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig()
        assert config.gpu_transcription_timeout == 300
