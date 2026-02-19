"""
Whisper model client with GPU lock management.

Provides thread-safe access to shared WhisperModel for parallel transcription.

CRITICAL FIX:
The _lock error was caused by multiple ThreadPoolExecutor workers trying to
simultaneously access the GPU through separate WhisperModel instances.
ctranslate2 (used by faster-whisper) cannot handle concurrent GPU access.

SOLUTION:
- Use a SINGLE shared WhisperModel instance
- Protect GPU transcription with a threading.Lock() mutex
- Parallelize only the CPU-bound work (audio extraction, I/O)
- Serialize GPU transcription through the shared model
"""

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Global shared model and GPU mutex
_gpu_lock = threading.RLock()  # RLock is REENTRANT - allows same thread to acquire multiple times
_shared_model = None  # Shared WhisperModel instance
_model_config = {}  # Model configuration cache

# Language detection metrics (US-137-009)
# Tracks usage of different detection methods across all transcriptions
_language_detection_metrics = {
    'whisper': 0,
    'langdetect': 0,
    'langid': 0,
    'consensus': 0,
    'weighted': 0,
}


def _get_gpu_memory_mb() -> tuple[float, float]:
    """
    Get current GPU memory usage.

    Returns:
        Tuple of (allocated_mb, reserved_mb), or (0.0, 0.0) if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return allocated, reserved
    except ImportError:
        pass
    return 0.0, 0.0


def check_gpu_health() -> dict:
    """
    Check GPU health before starting transcription batch.

    Verifies CUDA availability and memory.

    Returns:
        Dict with keys: 'available' (bool), 'message' (str), 'details' (dict)
    """
    result = {
        'available': False,
        'message': 'GPU health check did not run',
        'details': {}
    }

    try:
        import torch
        if not torch.cuda.is_available():
            result['message'] = 'CUDA not available - will use CPU'
            result['details'] = {'cuda_available': False}
            return result

        # CUDA is available - check memory
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        allocated_mb, reserved_mb = _get_gpu_memory_mb()
        free_memory_mb = total_memory_mb - reserved_mb

        result['available'] = True
        result['message'] = f'GPU available: {device_name}'
        result['details'] = {
            'cuda_available': True,
            'device_count': device_count,
            'device_name': device_name,
            'total_memory_mb': round(total_memory_mb, 1),
            'allocated_mb': round(allocated_mb, 1),
            'reserved_mb': round(reserved_mb, 1),
            'free_memory_mb': round(free_memory_mb, 1),
        }

        # Warn if low memory (< 1GB free)
        if free_memory_mb < 1024:
            result['message'] += f' - WARNING: Low GPU memory ({free_memory_mb:.0f}MB free)'
            logger.warning(f"Low GPU memory: {free_memory_mb:.0f}MB free out of {total_memory_mb:.0f}MB")

    except ImportError:
        result['message'] = 'PyTorch not installed - cannot check GPU'
        result['details'] = {'torch_import_error': True}
    except Exception as e:
        result['message'] = f'GPU check failed: {str(e)}'
        result['details'] = {'error': str(e)}

    return result


def check_model_health(model_name: str = 'base', compute_type: str = 'auto') -> dict:
    """
    Check Whisper model health by verifying it can load.

    Args:
        model_name: Whisper model name (default: 'base')
        compute_type: Compute type (default: 'auto')

    Returns:
        Dict with keys: 'loadable' (bool), 'message' (str), 'details' (dict)
    """
    result = {
        'loadable': False,
        'message': 'Model health check did not run',
        'details': {}
    }

    try:
        from faster_whisper import WhisperModel

        # Try to load the model briefly to verify it works
        # Use minimal computation to test
        test_model = WhisperModel(
            model_name,
            device='cpu',  # Test with CPU to avoid GPU memory issues
            compute_type=compute_type if compute_type != 'auto' else 'int8'
        )

        result['loadable'] = True
        result['message'] = f'Model {model_name} loaded successfully'
        result['details'] = {
            'model_name': model_name,
            'compute_type': compute_type,
            'device': 'cpu'
        }

        # Clean up
        del test_model

    except ImportError as e:
        result['message'] = f'faster-whisper not installed: {str(e)}'
        result['details'] = {'import_error': str(e)}
    except Exception as e:
        result['message'] = f'Model load failed: {str(e)}'
        result['details'] = {'error': str(e)}

    return result


def check_ffmpeg_health() -> dict:
    """
    Check FFmpeg availability for audio extraction.

    Returns:
        Dict with keys: 'available' (bool), 'message' (str), 'details' (dict)
    """
    result = {
        'available': False,
        'message': 'FFmpeg health check did not run',
        'details': {}
    }

    try:
        import subprocess
        process = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if process.returncode == 0:
            # Parse version from output
            version_line = process.stdout.split('\n')[0] if process.stdout else 'unknown'
            result['available'] = True
            result['message'] = f'FFmpeg available: {version_line}'
            result['details'] = {
                'version_line': version_line,
                'return_code': process.returncode
            }
        else:
            result['message'] = 'FFmpeg not working properly'
            result['details'] = {'return_code': process.returncode}

    except FileNotFoundError:
        result['message'] = 'FFmpeg not found in PATH'
        result['details'] = {'ffmpeg_not_found': True}
    except subprocess.TimeoutExpired:
        result['message'] = 'FFmpeg check timed out'
        result['details'] = {'timeout': True}
    except Exception as e:
        result['message'] = f'FFmpeg check failed: {str(e)}'
        result['details'] = {'error': str(e)}

    return result


def run_transcription_health_checks(
    model_name: str = 'base',
    compute_type: str = 'auto',
    skip_model_check: bool = False
) -> dict:
    """
    Run all transcription health checks before batch start.

    Args:
        model_name: Whisper model name
        compute_type: Compute type
        skip_model_check: If True, skip model loading test (faster startup)

    Returns:
        Dict with 'all_passed' (bool), 'checks' (list of results), 'message' (str)
    """
    checks = []

    # Check GPU
    gpu_check = check_gpu_health()
    checks.append({'name': 'gpu', **gpu_check})

    # Check FFmpeg
    ffmpeg_check = check_ffmpeg_health()
    checks.append({'name': 'ffmpeg', **ffmpeg_check})

    # Check model (optional - can skip for faster startup)
    if not skip_model_check:
        model_check = check_model_health(model_name, compute_type)
        checks.append({'name': 'model', **model_check})
        all_passed = gpu_check['available'] and ffmpeg_check['available'] and model_check['loadable']
    else:
        all_passed = gpu_check['available'] and ffmpeg_check['available']

    # Generate summary message
    failed_checks = [c['name'] for c in checks if not c.get('available', c.get('loadable', False))]
    if failed_checks:
        message = f"Health checks failed: {', '.join(failed_checks)}"
    else:
        message = "All transcription health checks passed"

    return {
        'all_passed': all_passed,
        'checks': checks,
        'message': message
    }


def _calculate_quality_metrics(result: list) -> dict:
    """Calculate quality metrics from transcription result (US-110-007).

    Args:
        result: List of segment dicts from transcription.

    Returns:
        Dict with 'avg_word_confidence' and 'min_segment_confidence'.
    """
    if not result:
        return {'avg_word_confidence': 0.0, 'min_segment_confidence': 1.0}

    all_word_confidences = []
    segment_confidences = []

    for seg in result:
        # Collect segment confidence
        seg_conf = seg.get('segment_confidence')
        if seg_conf is not None:
            segment_confidences.append(seg_conf)

        # Collect word-level confidences
        words = seg.get('words', [])
        for w in words:
            conf = w.get('confidence')
            if conf is not None:
                all_word_confidences.append(conf)

        # Also check avg_word_confidence at segment level
        avg_word_conf = seg.get('avg_word_confidence')
        if avg_word_conf is not None:
            all_word_confidences.append(avg_word_conf)

    # Calculate metrics
    avg_word_conf = (
        sum(all_word_confidences) / len(all_word_confidences)
        if all_word_confidences else 0.0
    )
    min_seg_conf = (
        min(segment_confidences) if segment_confidences else 1.0
    )

    return {
        'avg_word_confidence': avg_word_conf,
        'min_segment_confidence': min_seg_conf,
    }


def _detect_language_with_library(text: str, library: str) -> tuple[str, float]:
    """
    Detect language using specified library (US-124-006).

    Args:
        text: Text to analyze for language detection
        library: One of "langdetect", "langid", "fasttext"

    Returns:
        Tuple of (language_code, confidence_score)
    """
    if not text or not text.strip():
        return 'en', 0.0

    try:
        if library == "langdetect":
            from langdetect import detect, LangDetectException
            lang = detect(text)
            # langdetect doesn't provide confidence, estimate based on text length
            confidence = min(0.9, len(text) / 500.0) if len(text) < 500 else 0.9
            return lang, confidence

        elif library == "langid":
            import langid
            lang, confidence = langid.texts(text)
            # langid returns confidence as probability, normalize if needed
            if isinstance(confidence, (int, float)) and confidence > 1.0:
                confidence = confidence / 100.0
            return lang, float(confidence) if confidence else 0.7

        elif library == "fasttext":
            try:
                import fasttext
                # Use lid.176.bin model - detects language with confidence
                # Note: Requires fasttext model file to be downloaded
                # Using language identification model
                model = fasttext.load_model('/usr/share/fasttext/lid.176.bin')
                predictions = model.predict(text.replace('\n', ' '), k=1)
                lang = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
                return lang, confidence
            except (ImportError, FileNotFoundError):
                # Fallback if fasttext model not available
                logger.debug("FastText model not available, falling back to langdetect")
                from langdetect import detect, LangDetectException
                lang = detect(text)
                confidence = min(0.9, len(text) / 500.0)
                return lang, confidence

        else:
            logger.warning(f"Unknown language detection library: {library}, using langdetect")
            from langdetect import detect, LangDetectException
            lang = detect(text)
            return lang, 0.7

    except ImportError:
        logger.warning(f"Library '{library}' not available for language detection")
        return 'en', 0.0
    except Exception as e:
        logger.debug(f"Language detection failed with {library}: {e}")
        return 'en', 0.0


def _detect_language_consensus(text: str) -> tuple[str, float]:
    """
    Detect language using consensus from multiple libraries (US-124-006).

    Runs langdetect and langid (if available) and returns the language
    that both agree on, or the one with higher confidence if they disagree.

    Args:
        text: Text to analyze for language detection

    Returns:
        Tuple of (language_code, confidence_score)
    """
    if not text or not text.strip():
        return 'en', 0.0

    results = []

    # Try langdetect
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        confidence = min(0.9, len(text) / 500.0)
        results.append(('langdetect', lang, confidence))
    except ImportError:
        pass
    except Exception:
        pass

    # Try langid
    try:
        import langid
        lang, conf = langid.texts(text)
        if isinstance(conf, (int, float)) and conf > 1.0:
            conf = conf / 100.0
        results.append(('langid', lang, float(conf) if conf else 0.7))
    except ImportError:
        pass
    except Exception:
        pass

    # Try fasttext if available
    try:
        import fasttext
        model = fasttext.load_model('/usr/share/fasttext/lid.176.bin')
        predictions = model.predict(text.replace('\n', ' '), k=1)
        lang = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])
        results.append(('fasttext', lang, confidence))
    except (ImportError, FileNotFoundError):
        pass
    except Exception:
        pass

    if not results:
        logger.warning("No language detection libraries available, defaulting to English")
        return 'en', 0.0

    # Find consensus or highest confidence
    if len(results) == 1:
        return results[0][1], results[0][2]

    # Check for consensus (same language from multiple libraries)
    lang_counts = {}
    for lib, lang, conf in results:
        if lang not in lang_counts:
            lang_counts[lang] = {'count': 0, 'total_conf': 0.0, 'libs': []}
        lang_counts[lang]['count'] += 1
        lang_counts[lang]['total_conf'] += conf
        lang_counts[lang]['libs'].append(lib)

    # Find language with most votes
    best_lang = None
    best_count = 0
    best_conf = 0.0

    for lang, data in lang_counts.items():
        if data['count'] > best_count:
            best_count = data['count']
            best_lang = lang
            best_conf = data['total_conf'] / data['count']

    # If consensus (multiple libraries agree), boost confidence
    if best_count > 1:
        confidence = min(1.0, best_conf + 0.1)  # Boost for consensus
    else:
        confidence = best_conf

    return best_lang, confidence


# Default weights for weighted language detection (US-137-009)
LANGUAGE_DETECTION_WEIGHTS = {
    'whisper': 0.60,
    'langdetect': 0.25,
    'langid': 0.15,
}


def _detect_language_weighted(
    text: str,
    whisper_language: str,
    whisper_confidence: float,
    min_combined_confidence: float = 0.7,
) -> tuple[str, float, dict]:
    """
    Detect language using weighted confidence combination (US-137-009).

    Combines Whisper's detection with langdetect and langid using weighted confidence:
      - Whisper: 60% weight
      - langdetect: 25% weight
      - langid: 15% weight

    Args:
        text: Text to analyze for language detection
        whisper_language: Language detected by Whisper
        whisper_confidence: Whisper's confidence score
        min_combined_confidence: Minimum combined confidence threshold

    Returns:
        Tuple of (language_code, combined_confidence, detection_breakdown)
        where detection_breakdown contains per-source confidence scores
    """
    if not text or not text.strip():
        return 'en', 0.0, {'whisper': 0.0, 'langdetect': 0.0, 'langid': 0.0}

    # Initialize detection results
    detection_results = {
        'whisper': {'language': whisper_language, 'confidence': whisper_confidence},
        'langdetect': {'language': None, 'confidence': 0.0},
        'langid': {'language': None, 'confidence': 0.0},
    }

    # Get langdetect result
    try:
        from langdetect import detect, LangDetectException
        langdetect_lang = detect(text)
        # Estimate confidence based on text length
        langdetect_conf = min(0.9, len(text) / 500.0) if len(text) < 500 else 0.9
        detection_results['langdetect'] = {'language': langdetect_lang, 'confidence': langdetect_conf}
    except ImportError:
        logger.debug("langdetect not available for weighted detection")
    except Exception:
        logger.debug("langdetect detection failed")

    # Get langid result
    try:
        import langid
        langid_lang, langid_conf = langid.texts(text)
        if isinstance(langid_conf, (int, float)) and langid_conf > 1.0:
            langid_conf = langid_conf / 100.0
        detection_results['langid'] = {'language': langid_lang, 'confidence': float(langid_conf) if langid_conf else 0.7}
    except ImportError:
        logger.debug("langid not available for weighted detection")
    except Exception:
        logger.debug("langid detection failed")

    # Calculate weighted confidence
    weighted_confidence = 0.0
    whisper_weight = LANGUAGE_DETECTION_WEIGHTS['whisper']
    langdetect_weight = LANGUAGE_DETECTION_WEIGHTS['langdetect']
    langid_weight = LANGUAGE_DETECTION_WEIGHTS['langid']

    # Add Whisper contribution
    weighted_confidence += whisper_confidence * whisper_weight

    # Add langdetect contribution if available
    if detection_results['langdetect']['language']:
        weighted_confidence += detection_results['langdetect']['confidence'] * langdetect_weight
    else:
        # Redistribute weight if langdetect unavailable
        whisper_weight = whisper_weight / (whisper_weight + langid_weight) if (whisper_weight + langid_weight) > 0 else 0.6
        langid_weight = 1.0 - whisper_weight

    # Add langid contribution if available
    if detection_results['langid']['language']:
        weighted_confidence += detection_results['langid']['confidence'] * langid_weight
    else:
        # Redistribute weight if langid unavailable
        whisper_weight = whisper_weight / (whisper_weight + langdetect_weight) if (whisper_weight + langdetect_weight) > 0 else 0.6
        langdetect_weight = 1.0 - whisper_weight
        # Recalculate with redistributed weights
        weighted_confidence = (
            whisper_confidence * whisper_weight +
            detection_results['langdetect']['confidence'] * langdetect_weight
        )

    # Determine final language using weighted voting
    # Count votes weighted by confidence
    lang_votes = {}
    for source, weight in [('whisper', whisper_weight), ('langdetect', langdetect_weight), ('langid', langid_weight)]:
        if detection_results[source]['language']:
            lang = detection_results[source]['language']
            conf = detection_results[source]['confidence']
            if lang not in lang_votes:
                lang_votes[lang] = 0.0
            lang_votes[lang] += conf * weight

    # Get language with highest weighted vote
    final_language = whisper_language  # Default to Whisper
    if lang_votes:
        final_language = max(lang_votes.keys(), key=lambda l: lang_votes[l])

    # Build detection breakdown for logging
    detection_breakdown = {
        'whisper': whisper_confidence,
        'langdetect': detection_results['langdetect']['confidence'],
        'langid': detection_results['langid']['confidence'],
    }

    return final_language, weighted_confidence, detection_breakdown


def get_language_detection_metrics() -> dict:
    """
    Get language detection metrics (US-137-009).

    Returns a copy of the language detection method usage counts.

    Returns:
        Dict with detection method counts: {'whisper': N, 'langdetect': N, 'langid': N, 'consensus': N, 'weighted': N}
    """
    return _language_detection_metrics.copy()


def record_language_detection(source: str) -> None:
    """
    Record a language detection event (US-137-009).

    Args:
        source: The detection method used ('whisper', 'langdetect', 'langid', 'consensus', 'weighted')
    """
    global _language_detection_metrics
    if source in _language_detection_metrics:
        _language_detection_metrics[source] += 1


def reset_language_detection_metrics() -> None:
    """Reset language detection metrics to zero (US-137-009)."""
    global _language_detection_metrics
    _language_detection_metrics = {
        'whisper': 0,
        'langdetect': 0,
        'langid': 0,
        'consensus': 0,
        'weighted': 0,
    }


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in MB (US-60-007).

    Queries CUDA device for total memory and subtracts currently allocated memory
    to determine how much is available for new allocations.

    Returns:
        Available GPU memory in MB, or 0.0 if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            # Account for some reserved memory overhead
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            # Available = total - max(allocated, reserved) to be conservative
            used = max(allocated, reserved)
            available = total - used
            return available
    except ImportError:
        pass
    return 0.0


def get_gpu_utilization() -> float:
    """
    Get current GPU utilization percentage (US-137-006).

    Uses nvidia-smi CLI to query GPU utilization as a percentage.
    Falls back to returning -1.0 if nvidia-smi is not available.

    Returns:
        GPU utilization as percentage (0-100), or -1.0 if unavailable
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse first GPU's utilization (value between 0-100)
            utilization = float(result.stdout.strip().split('\n')[0])
            return utilization
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return -1.0  # Return -1 to indicate unavailable


# GPU memory management globals (US-137-010)
_gpu_memory_monitor_thread = None
_gpu_memory_monitor_running = False
_gpu_memory_history = []  # Track memory trends
_gpu_memory_last_pressure_warning = 0  # Timestamp of last warning
_GPU_MEMORY_HISTORY_MAX = 60  # Keep last 60 samples (2 minutes at 2s intervals)
_GPU_MEMORY_WARNING_INTERVAL = 30  # Minimum seconds between warnings


def get_gpu_memory_percent() -> float:
    """
    Get current GPU memory usage as percentage (US-137-010).

    Returns:
        GPU memory usage as percentage (0-100+), or -1.0 if unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated()
            # Allow for some overhead, use max of allocated vs reserved
            reserved = torch.cuda.memory_reserved()
            used = max(allocated, reserved)
            percent = (used / total) * 100.0
            return percent
    except ImportError:
        pass
    return -1.0  # Return -1 to indicate unavailable


def is_memory_pressure_active(threshold_percent: float = 85.0) -> bool:
    """
    Check if GPU memory pressure is currently active (US-137-010).

    Args:
        threshold_percent: Memory threshold percentage to check against

    Returns:
        True if memory usage exceeds threshold, False otherwise
    """
    memory_percent = get_gpu_memory_percent()
    if memory_percent < 0:
        return False  # Can't determine, assume no pressure
    return memory_percent >= threshold_percent


def _memory_trend_analysis() -> str:
    """
    Analyze memory trend from history (US-137-010).

    Returns:
        Trend string: 'rising', 'stable', or 'falling'
    """
    if len(_gpu_memory_history) < 5:
        return 'stable'

    recent = _gpu_memory_history[-5:]
    avg_recent = sum(recent) / len(recent)

    older = _gpu_memory_history[-10:-5] if len(_gpu_memory_history) >= 10 else _gpu_memory_history[:-5]
    if older:
        avg_older = sum(older) / len(older)
        diff = avg_recent - avg_older

        if diff > 5.0:
            return 'rising'
        elif diff < -5.0:
            return 'falling'

    return 'stable'


def monitor_gpu_memory(
    enabled: bool = True,
    threshold_percent: float = 85.0,
    check_interval: float = 2.0,
    callback=None
) -> None:
    """
    GPU memory and detect Background thread to monitor pressure (US-137-010).

    This function runs a background thread that continuously monitors GPU memory
    usage and triggers callbacks when memory pressure is detected.

    Args:
        enabled: Whether monitoring is enabled
        threshold_percent: Memory threshold percentage to trigger warnings
        check_interval: How often to check memory (seconds)
        callback: Optional callback function( memory_percent, trend) to call on each check
    """
    global _gpu_memory_monitor_thread, _gpu_memory_monitor_running, _gpu_memory_history

    if not enabled:
        # Stop monitoring if running
        _gpu_memory_monitor_running = False
        if _gpu_memory_monitor_thread and _gpu_memory_monitor_thread.is_alive():
            _gpu_memory_monitor_thread.join(timeout=5.0)
        _gpu_memory_monitor_thread = None
        return

    if _gpu_memory_monitor_thread and _gpu_memory_monitor_thread.is_alive():
        # Already running
        return

    def monitor_loop():
        global _gpu_memory_history, _gpu_memory_last_pressure_warning

        logger.info(f"GPU memory monitor started (threshold: {threshold_percent}%, interval: {check_interval}s)")

        while _gpu_memory_monitor_running:
            try:
                memory_percent = get_gpu_memory_percent()

                if memory_percent >= 0:
                    # Update history
                    _gpu_memory_history.append(memory_percent)
                    if len(_gpu_memory_history) > _GPU_MEMORY_HISTORY_MAX:
                        _gpu_memory_history.pop(0)

                    # Get trend
                    trend = _memory_trend_analysis()

                    # Log warning if threshold exceeded
                    if memory_percent >= threshold_percent:
                        current_time = time.time()
                        if current_time - _gpu_memory_last_pressure_warning > _GPU_MEMORY_WARNING_INTERVAL:
                            logger.warning(
                                f"GPU memory pressure detected: {memory_percent:.1f}% "
                                f"(threshold: {threshold_percent}%, trend: {trend})"
                            )
                            _gpu_memory_last_pressure_warning = current_time

                    # Call callback if provided
                    if callback:
                        try:
                            callback(memory_percent, trend)
                        except Exception as e:
                            logger.debug(f"GPU memory callback error: {e}")

            except Exception as e:
                logger.debug(f"GPU memory monitor error: {e}")

            # Sleep for check interval
            time.sleep(check_interval)

        logger.info("GPU memory monitor stopped")

    _gpu_memory_monitor_running = True
    _gpu_memory_history = []
    _gpu_memory_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    _gpu_memory_monitor_thread.start()


def stop_gpu_memory_monitor() -> None:
    """Stop the GPU memory monitoring thread (US-137-010)."""
    global _gpu_memory_monitor_running
    _gpu_memory_monitor_running = False


def get_gpu_memory_stats() -> dict:
    """
    Get GPU memory monitoring statistics (US-137-010).

    Returns:
        Dict with current memory stats and history
    """
    memory_percent = get_gpu_memory_percent()
    trend = _memory_trend_analysis() if _gpu_memory_history else 'unknown'

    return {
        'memory_percent': memory_percent,
        'trend': trend,
        'history_count': len(_gpu_memory_history),
        'history': _gpu_memory_history.copy() if _gpu_memory_history else [],
        'monitor_running': _gpu_memory_monitor_thread is not None and _gpu_memory_monitor_thread.is_alive(),
    }


def handle_memory_pressure(
    strategy: str = "wait_and_retry",
    threshold_percent: float = 85.0,
    max_wait_seconds: float = 60.0,
    check_interval: float = 2.0,
    reduction_factor: float = 0.5,
    current_batch_size: int = 50,
    min_batch_size: int = 5,
    logger=None
) -> tuple[bool, int, str]:
    """
    Handle GPU memory pressure with configurable recovery strategy (US-137-010).

    Args:
        strategy: Recovery strategy ('wait_and_retry', 'reduce_batch_size', 'fallback_to_cpu')
        threshold_percent: Memory threshold to trigger recovery
        max_wait_seconds: Maximum time to wait for memory to free
        check_interval: How often to check memory during wait
        reduction_factor: Factor to reduce batch size by
        current_batch_size: Current batch size
        min_batch_size: Minimum batch size to maintain
        logger: Optional logger instance

    Returns:
        Tuple of (success, suggested_batch_size, message)
        - success: True if pressure resolved, False if should fallback to CPU
        - suggested_batch_size: Recommended batch size (may be reduced)
        - message: Status message
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    memory_percent = get_gpu_memory_percent()

    if memory_percent < 0:
        # Can't determine memory, assume OK
        return True, current_batch_size, "Unable to determine GPU memory, continuing"

    if memory_percent < threshold_percent:
        # No pressure, continue normally
        return True, current_batch_size, f"Memory OK ({memory_percent:.1f}%)"

    # Memory pressure detected
    logger.info(f"GPU memory pressure: {memory_percent:.1f}% >= {threshold_percent}%")

    if strategy == "wait_and_retry":
        # Wait for memory to free up
        logger.info(f"Strategy: wait_and_retry (max wait: {max_wait_seconds}s)")
        elapsed = 0.0
        while elapsed < max_wait_seconds:
            time.sleep(check_interval)
            elapsed += check_interval

            new_memory = get_gpu_memory_percent()
            if new_memory < 0 or new_memory < threshold_percent:
                wait_time = elapsed
                logger.info(f"Memory recovered after {wait_time:.1f}s (now {new_memory:.1f}%)")
                return True, current_batch_size, f"Recovered after {wait_time:.1f}s"

        # Timeout
        logger.warning(f"Memory did not recover after {max_wait_seconds}s, suggesting batch reduction")
        return False, current_batch_size, f"Timeout waiting for memory recovery ({max_wait_seconds}s)"

    elif strategy == "reduce_batch_size":
        # Reduce batch size and continue
        new_batch_size = max(int(current_batch_size * reduction_factor), min_batch_size)
        logger.info(
            f"Strategy: reduce_batch_size "
            f"(current: {current_batch_size}, new: {new_batch_size}, factor: {reduction_factor})"
        )
        return True, new_batch_size, f"Batch size reduced from {current_batch_size} to {new_batch_size}"

    elif strategy == "fallback_to_cpu":
        # Fallback to CPU
        logger.warning("Strategy: fallback_to_cpu - switching to CPU processing")
        return False, current_batch_size, "Falling back to CPU processing"

    else:
        logger.warning(f"Unknown strategy: {strategy}, defaulting to wait_and_retry")
        return handle_memory_pressure(
            "wait_and_retry", threshold_percent, max_wait_seconds,
            check_interval, reduction_factor, current_batch_size, min_batch_size, logger
        )


def calculate_memory_aware_batch_size(
    base_batch_size: int,
    memory_percent: float,
    threshold_percent: float = 85.0,
    reduction_factor: float = 0.5,
    min_batch_size: int = 5
) -> int:
    """
    Calculate memory-aware batch size based on current GPU memory usage (US-137-010).

    Args:
        base_batch_size: Original batch size
        memory_percent: Current GPU memory usage percentage
        threshold_percent: Threshold at which to start reducing
        reduction_factor: Factor to reduce by per threshold exceeded
        min_batch_size: Minimum batch size to maintain

    Returns:
        Adjusted batch size
    """
    if memory_percent < 0 or memory_percent < threshold_percent:
        return base_batch_size

    # Calculate how many thresholds we've exceeded
    excess = memory_percent - (threshold_percent - 10)  # Start reducing 10% before threshold
    if excess <= 0:
        return base_batch_size

    # Each 10% over threshold reduces batch size
    reduction_steps = max(1, int(excess / 10))
    adjusted = int(base_batch_size * (reduction_factor ** reduction_steps))

    return max(adjusted, min_batch_size)


# Model size estimates in MB (approximate VRAM at float16)
# Used for automatic model downgrade when GPU memory is insufficient
MODEL_MEMORY_REQUIREMENTS = {
    "large-v3": 3000,
    "large-v2": 3000,
    "large": 3000,
    "medium": 2000,
    "small": 1000,
    "base": 500,
    "tiny": 400,
}

# Model downgrade chain (larger -> smaller)
MODEL_DOWNGRADE_ORDER = ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]


# Duration thresholds in seconds for automatic model selection (US-110-010)
DURATION_THRESHOLD_SHORT = 5 * 60    # 5 minutes
DURATION_THRESHOLD_MEDIUM = 30 * 60  # 30 minutes


def _select_model_for_duration(audio_duration_seconds: float, requested_model: str) -> tuple[str, str]:
    """
    Select appropriate Whisper model based on audio duration (US-110-010).

    Uses smaller models for shorter audio to improve speed while maintaining
    sufficient accuracy:
      - tiny:  audio < 5 minutes (fastest, suitable for short clips)
      - base:  5-30 minutes (balanced speed/quality)
      - small: 30+ minutes (better quality for longer content)

    Args:
        audio_duration_seconds: Duration of the audio file in seconds
        requested_model: The model name specified by user (manual override)

    Returns:
        Tuple of (model_name, reason) explaining why the model was selected
    """
    if audio_duration_seconds < DURATION_THRESHOLD_SHORT:
        selected_model = "tiny"
        reason = f"audio duration {audio_duration_seconds:.1f}s < {DURATION_THRESHOLD_SHORT}s (short audio)"
    elif audio_duration_seconds < DURATION_THRESHOLD_MEDIUM:
        selected_model = "base"
        reason = f"audio duration {audio_duration_seconds:.1f}s in [{DURATION_THRESHOLD_SHORT}s, {DURATION_THRESHOLD_MEDIUM}s) (medium audio)"
    else:
        selected_model = "small"
        reason = f"audio duration {audio_duration_seconds:.1f}s >= {DURATION_THRESHOLD_MEDIUM}s (long audio)"

    return selected_model, reason


def _select_model_for_memory(requested_model: str, available_memory_mb: float, min_memory_mb: float) -> str:
    """
    Select appropriate model based on available GPU memory (US-60-007).

    If requested model requires more memory than available, automatically
    downgrades to a smaller model that fits within memory constraints.

    Args:
        requested_model: Model name requested by user
        available_memory_mb: Available GPU memory in MB
        min_memory_mb: Minimum memory threshold from config

    Returns:
        Model name to use (may be downgraded from requested)
    """
    # If we have plenty of memory, use requested model
    if available_memory_mb >= min_memory_mb:
        model_req = MODEL_MEMORY_REQUIREMENTS.get(requested_model, 500)
        if available_memory_mb >= model_req:
            return requested_model

    # Find position of requested model in downgrade chain
    try:
        start_idx = MODEL_DOWNGRADE_ORDER.index(requested_model)
    except ValueError:
        # Unknown model, assume it's small enough
        start_idx = len(MODEL_DOWNGRADE_ORDER) - 1

    # Try each model from requested downward
    for model in MODEL_DOWNGRADE_ORDER[start_idx:]:
        model_req = MODEL_MEMORY_REQUIREMENTS.get(model, 500)
        if available_memory_mb >= model_req:
            if model != requested_model:
                logger.warning(
                    f"Insufficient GPU memory ({available_memory_mb:.0f}MB available). "
                    f"Downgrading model from '{requested_model}' to '{model}'"
                )
            return model

    # If nothing fits, return tiny as last resort
    logger.warning(
        f"Very low GPU memory ({available_memory_mb:.0f}MB). Using 'tiny' model."
    )
    return "tiny"


class WhisperClient:
    """
    Thread-safe Whisper model client with GPU locking.

    Manages a shared WhisperModel instance for efficient GPU usage
    in parallel transcription scenarios.
    """

    def __init__(
        self,
        model_name: str = "base",
        compute_type: str = "auto",
        minimum_gpu_memory_mb: int = 2000,
        auto_downgrade_model: bool = True,
        gpu_transcription_timeout: int = 300,
        num_workers: int = 1,
        cpu_threads: int = 4,
        auto_fallback_to_cpu: bool = True,  # US-110-004
        auto_model_selection: bool = True,  # US-110-010
        model_version: str = None,  # US-124-010: Model version to pin (e.g., "v3")
        # GPU memory management (US-137-010)
        gpu_memory_monitoring_enabled: bool = True,
        gpu_memory_threshold_percent: float = 85.0,
        memory_recovery_strategy: str = "wait_and_retry",
        memory_check_interval_seconds: float = 2.0,
        memory_recovery_max_wait_seconds: float = 60.0,
        memory_batch_reduction_factor: float = 0.5,
        min_batch_size_under_pressure: int = 5,
    ):
        """
        Initialize Whisper client with model configuration.

        Args:
            model_name: Whisper model name (base, small, medium, large, etc.)
            compute_type: Compute type (auto, float16, int8)
            minimum_gpu_memory_mb: Minimum GPU memory required (US-60-007)
            auto_downgrade_model: Automatically downgrade model if insufficient memory
            gpu_transcription_timeout: Max seconds for a single transcribe() call (US-79-002)
            num_workers: Workers for WhisperModel batched decoding (US-79-007)
            cpu_threads: CPU threads for ctranslate2 operations (US-79-007)
            auto_fallback_to_cpu: Automatically fall back to CPU when GPU fails (US-110-004)
            auto_model_selection: Automatically select model based on audio duration (US-110-010)
            model_version: Version to pin model to (e.g., "v3" for large-v3) (US-124-010)
            gpu_memory_monitoring_enabled: Enable GPU memory monitoring (US-137-010)
            gpu_memory_threshold_percent: Memory threshold to trigger preemption (US-137-010)
            memory_recovery_strategy: Strategy for memory recovery (US-137-010)
            memory_check_interval_seconds: How often to check memory (US-137-010)
            memory_recovery_max_wait_seconds: Max wait for memory recovery (US-137-010)
            memory_batch_reduction_factor: Factor to reduce batch size (US-137-010)
            min_batch_size_under_pressure: Minimum batch size under pressure (US-137-010)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.compute_type = compute_type
        self.minimum_gpu_memory_mb = minimum_gpu_memory_mb
        self.auto_downgrade_model = auto_downgrade_model
        self.gpu_transcription_timeout = gpu_transcription_timeout
        self.num_workers = num_workers
        self.cpu_threads = cpu_threads
        self.auto_fallback_to_cpu = auto_fallback_to_cpu
        self.auto_model_selection = auto_model_selection
        self._cpu_fallback_mode = False  # Track if we've fallen back to CPU

        # GPU memory management (US-137-010)
        self.gpu_memory_monitoring_enabled = gpu_memory_monitoring_enabled
        self.gpu_memory_threshold_percent = gpu_memory_threshold_percent
        self.memory_recovery_strategy = memory_recovery_strategy
        self.memory_check_interval_seconds = memory_check_interval_seconds
        self.memory_recovery_max_wait_seconds = memory_recovery_max_wait_seconds
        self.memory_batch_reduction_factor = memory_batch_reduction_factor
        self.min_batch_size_under_pressure = min_batch_size_under_pressure

    def _apply_version_pinning(self, model_name: str) -> str:
        """
        Apply version pinning to model name (US-124-010).

        Combines model name with version to create pinned model identifier.
        Examples:
            - "large" + "v3" -> "large-v3"
            - "small" + "v2" -> "small-v2"

        Args:
            model_name: Base model name (e.g., "large", "small")

        Returns:
            Model name with version appended (e.g., "large-v3")
        """
        if not self.model_version:
            return model_name

        # Normalize version string (remove @ prefix if present)
        version = self.model_version.lstrip('@')

        # Check if model already has version
        if '-' in model_name or '@' in model_name:
            # Model already has version specified, don't append
            logger.debug(f"Model '{model_name}' already has version, ignoring pinned version '{version}'")
            return model_name

        # Combine model and version
        pinned_model = f"{model_name}-{version}"
        logger.debug(f"Applied version pinning: {model_name} -> {pinned_model}")
        return pinned_model

    def _validate_model_version(self, model_name: str) -> bool:
        """
        Validate that the specified model version exists (US-124-010).

        Checks if the model with given version is available for download.
        Note: This is a basic validation - the actual download will fail
        with a more specific error if the version doesn't exist.

        Args:
            model_name: Model name with potential version (e.g., "large-v3")

        Returns:
            True if validation passes (or if validation is skipped)
        """
        if not self.model_version:
            return True

        # Valid version suffixes for faster-whisper models
        valid_versions = {'v1', 'v2', 'v3', 'v3-turbo', 'v3.5-turbo'}

        # Check if the version format is valid
        version = self.model_version.lstrip('@')
        if version not in valid_versions:
            logger.warning(
                f"Model version '{version}' may not be valid. "
                f"Valid versions: {', '.join(sorted(valid_versions))}"
            )

        return True

    def get_model(self, model_name: str = None):
        """
        Get or create the shared WhisperModel instance.

        Thread-safe initialization with double-checked locking.
        Performs GPU memory pre-check and automatic model downgrade if needed (US-60-007).

        Args:
            model_name: Optional model name override. If provided, uses this model instead of
                self.model_name. Useful for duration-based model selection (US-110-010).

        Returns:
            WhisperModel instance
        """
        global _shared_model, _model_config

        # Use provided model_name or fall back to configured model
        base_model = model_name if model_name else self.model_name

        # Apply version pinning (US-124-010)
        # Combine model name with version: e.g., "large" + "v3" = "large-v3"
        pinned_model = self._apply_version_pinning(base_model)

        # Determine actual model to use (may be downgraded based on GPU memory)
        # Use pinned_model (with version) as the base for downgrade logic
        actual_model = pinned_model
        if self.auto_downgrade_model:
            available_memory = get_available_gpu_memory()
            if available_memory > 0:  # Only check if CUDA available
                actual_model = _select_model_for_memory(
                    base_model,
                    available_memory,
                    self.minimum_gpu_memory_mb
                )
            elif available_memory == 0:
                # CUDA not available - warn if below threshold
                logger.info("CUDA not available - GPU memory check skipped")

        # Check if we need to (re)initialize
        current_config = {"model": actual_model, "compute_type": self.compute_type}

        if _shared_model is not None and _model_config == current_config:
            return _shared_model

        with _gpu_lock:
            # Double-check after acquiring lock
            if _shared_model is not None and _model_config == current_config:
                return _shared_model

            # Log GPU memory before initialization
            mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
            available_mem = get_available_gpu_memory()
            logger.info(f"GPU memory before model init: allocated={mem_before_alloc:.1f}MB, reserved={mem_before_reserved:.1f}MB, available={available_mem:.1f}MB")

            # Warn if below threshold (US-60-007)
            if available_mem > 0 and available_mem < self.minimum_gpu_memory_mb:
                logger.warning(
                    f"GPU memory ({available_mem:.0f}MB) below threshold ({self.minimum_gpu_memory_mb}MB). "
                    f"Performance may be degraded."
                )

            # Validate model version before initialization (US-124-010)
            if not self._validate_model_version(actual_model):
                raise ValueError(f"Invalid model version '{self.model_version}' for model '{actual_model}'")

            logger.info(f"Initializing WhisperModel...")
            logger.info(f"  Model: {actual_model}" + (f" (downgraded from {base_model})" if actual_model != base_model else ""))
            if self.model_version:
                logger.info(f"  Pinned version: {self.model_version}")
            logger.info(f"  Compute type: {self.compute_type}")

            try:
                logger.debug("Step 1: Importing faster_whisper...")
                from faster_whisper import WhisperModel
                logger.debug("Step 1: Done")

                # Determine device
                device = "cuda"
                actual_compute = self.compute_type

                if self.compute_type == "auto" or self.compute_type == "int8":
                    logger.debug("Step 2: Checking CUDA availability...")
                    try:
                        import torch
                        cuda_available = torch.cuda.is_available()
                        logger.debug(f"Step 2: CUDA available = {cuda_available}")
                        if cuda_available:
                            device = "cuda"
                            actual_compute = "float16"
                            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            logger.debug(f"Step 2: GPU memory = {gpu_mem:.1f} GB")
                        else:
                            device = "cpu"
                            actual_compute = "int8"
                    except ImportError:
                        logger.debug("Step 2: torch not available, using CPU")
                        device = "cpu"
                        actual_compute = "int8"

                logger.info(f"Step 3: Creating WhisperModel on {device} ({actual_compute})...")
                logger.info("(This may take 30-60 seconds on first run to download model)")

                sys.stdout.flush()
                sys.stderr.flush()

                _shared_model = WhisperModel(
                    actual_model,
                    device=device,
                    compute_type=actual_compute,
                    num_workers=self.num_workers,
                    cpu_threads=self.cpu_threads
                )
                _model_config = current_config

                logger.info(f"Step 3: Done!")
                logger.info(f"✓ Model ready on {device} ({actual_compute})")

                # Log GPU memory after initialization
                mem_after_alloc, mem_after_reserved = _get_gpu_memory_mb()
                logger.info(f"GPU memory after model init: allocated={mem_after_alloc:.1f}MB, reserved={mem_after_reserved:.1f}MB")
                mem_delta = mem_after_alloc - mem_before_alloc
                logger.info(f"GPU memory delta from model init: {mem_delta:.1f}MB")

                return _shared_model

            except Exception as e:
                logger.error(f"✗ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _retry_with_alternative_settings(
        self,
        audio_path: str,
        language: str,
        vad_filter: bool,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        original_result: list,
        original_quality_metrics: dict,
        min_quality_threshold: float
    ) -> Optional[tuple[list, dict]]:
        """
        Retry transcription with alternative settings when quality gate fails (US-137-005).

        This method is called when the quality gate check fails. It attempts to improve
        transcription quality by trying different settings:
        1. Enable VAD filter if not already enabled
        2. Enable word timestamps for better accuracy
        3. Try with language specified (if auto-detected)

        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            vad_filter: Current VAD filter setting
            min_silence_duration_ms: Current VAD parameter
            speech_pad_ms: Current VAD parameter
            original_result: The original transcription result
            original_quality_metrics: Quality metrics from original transcription
            min_quality_threshold: The quality threshold that was not met

        Returns:
            Tuple of (improved_result, improved_quality_metrics) if improvement found,
            None if no improvement could be achieved
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        # Get audio name for logging
        audio_name = os.path.basename(audio_path)

        logger.info(
            f"Quality gate retry for {audio_name}: attempting alternative settings "
            f"(vad_filter=True, word_timestamps=True)"
        )

        try:
            # Get model (will acquire GPU lock)
            model = self.get_model()

            # Retry with enhanced settings
            retry_start_time = time.monotonic()

            def _do_retry_transcribe():
                return model.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=True,  # Always enable VAD for quality retry
                    vad_parameters=dict(
                        min_silence_duration_ms=min_silence_duration_ms,
                        speech_pad_ms=speech_pad_ms
                    ),
                    word_timestamps=True  # Enable word timestamps for better accuracy
                )

            retry_timeout = self.gpu_transcription_timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_retry_transcribe)
                try:
                    retry_segments, retry_info = future.result(timeout=retry_timeout)
                except FuturesTimeoutError:
                    elapsed = time.monotonic() - retry_start_time
                    logger.warning(
                        f"Quality retry timeout for {audio_name}: "
                        f"elapsed={elapsed:.1f}s, timeout={retry_timeout}s"
                    )
                    return None

            # Process retry result
            retry_result = []
            for seg in retry_segments:
                seg_data = {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                }

                seg_logprob = getattr(seg, 'avg_logprob', None)
                if seg_logprob is not None and isinstance(seg_logprob, (int, float)):
                    seg_confidence = max(0.0, min(1.0, pow(2, seg_logprob)))
                    seg_data['segment_confidence'] = seg_confidence

                if hasattr(seg, 'words') and seg.words:
                    words_data = []
                    word_confidences = []
                    for w in seg.words:
                        word_info = {"word": w.word, "start": w.start, "end": w.end}
                        if hasattr(w, 'probability'):
                            prob = w.probability
                            if isinstance(prob, (int, float)):
                                word_info['confidence'] = prob
                                word_confidences.append(prob)
                        words_data.append(word_info)

                    seg_data["words"] = words_data
                    if word_confidences:
                        seg_data['avg_word_confidence'] = sum(word_confidences) / len(word_confidences)

                retry_result.append(seg_data)

            # Calculate quality metrics for retry result
            retry_quality_metrics = _calculate_quality_metrics(retry_result)
            avg_word_conf_new = retry_quality_metrics.get('avg_word_confidence', 0.0)

            logger.info(
                f"Quality gate retry complete for {audio_name}: "
                f"avg_word_confidence={avg_word_conf_new:.3f} (was {original_quality_metrics.get('avg_word_confidence', 0):.3f})"
            )

            # Check if retry improved quality
            if avg_word_conf_new >= min_quality_threshold:
                logger.info(
                    f"Quality gate PASSED after retry for {audio_name}: "
                    f"avg_word_confidence={avg_word_conf_new:.3f} >= min_quality_threshold={min_quality_threshold:.3f}"
                )
                return retry_result, retry_quality_metrics
            elif avg_word_conf_new > original_quality_metrics.get('avg_word_confidence', 0):
                # Improved but still below threshold - return the better result
                logger.info(
                    f"Quality gate retry improved but still below threshold for {audio_name}: "
                    f"avg_word_confidence={avg_word_conf_new:.3f} < min_quality_threshold={min_quality_threshold:.3f}"
                )
                return retry_result, retry_quality_metrics
            else:
                logger.warning(
                    f"Quality gate retry did not improve for {audio_name}, keeping original result"
                )
                return None

        except Exception as e:
            logger.warning(f"Quality gate retry failed for {audio_name}: {e}")
            return None

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        vad_filter: bool = False,  # Default False - VAD too aggressive for YouTube
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 10,
        word_timestamps: bool = False,
        min_language_confidence: float = 0.8,  # US-110-003
        streaming: bool = None,  # US-124-002: Streaming mode for long audio
        chunk_length: int = 30,  # US-124-002: Chunk length in seconds for streaming
        min_confidence_threshold: float = 0.7,  # US-124-003: Retry threshold for low confidence
        max_retries: int = 2,  # US-124-003: Max retries for low confidence
        retry_on_low_confidence: bool = True,  # US-124-003: Enable/disable retry logic
        vocabulary: List[str] = None,  # US-124-005: Custom vocabulary hints
        language_detection_library: str = "langdetect",  # US-124-006: Fallback library (langdetect, langid, fasttext)
        prefer_whisper_over_fallback: bool = False,  # US-124-006: Prefer Whisper over fallback
        use_consensus_detection: bool = False,  # US-124-006: Use consensus from multiple libraries
        language_detection_weighted_fallback: bool = True,  # US-137-009: Use weighted confidence combination
        min_combined_confidence: float = 0.7,  # US-137-009: Minimum combined confidence threshold
        post_processing: bool = True,  # US-124-011: Enable segment post-processing
        quality_gate_enabled: bool = True,  # US-137-005: Enable quality gate
        min_quality_threshold: float = 0.5  # US-137-005: Min avg word confidence for quality gate
    ) -> List[dict]:
        """
        Transcribe audio file with GPU lock.

        Acquires GPU lock before transcription, releases after.
        Automatically selects optimal model based on audio duration if auto_model_selection is enabled.
        For audio >30 minutes, streaming mode is automatically enabled to keep memory usage constant.

        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            vad_filter: Whether to apply Voice Activity Detection
            min_silence_duration_ms: Minimum silence duration to split segments
            speech_pad_ms: Padding around detected speech
            word_timestamps: Whether to include word-level timestamps
            min_language_confidence: Minimum confidence threshold for Whisper language
                detection. When Whisper confidence is below this, fallback to langdetect.
            streaming: Enable streaming mode for long audio. If None, auto-detects based on
                audio duration (>30 minutes enables streaming). When enabled, processes audio
                in chunks to keep memory usage constant (US-124-002).
            chunk_length: Length of audio chunks in seconds for streaming mode (default 30s).
                Only used when streaming is enabled or auto-detected for long audio.
            min_confidence_threshold: Minimum confidence threshold for transcription quality.
                If the minimum segment confidence falls below this, retry logic may be triggered
                (US-124-003). Default is 0.7.
            max_retries: Maximum number of retries when transcription confidence is below threshold
                (US-124-003). Default is 2.
            retry_on_low_confidence: Whether to retry transcription when confidence is below
                min_confidence_threshold (US-124-003). Default is True.
            vocabulary: Custom vocabulary hints to improve transcription accuracy for domain-specific
                terms (US-124-005). Format: List of words or short phrases (e.g., ["TensorFlow",
                "PyTorch", "transformer"]). These hints are passed to faster-whisper via the prompt
                parameter to improve recognition of technical terms, proper nouns, or niche vocabulary.
            language_detection_library: Fallback library for language detection when Whisper
                confidence is below threshold (US-124-006). Options: "langdetect" (default),
                "langid", "fasttext". Set to None to disable fallback.
            prefer_whisper_over_fallback: When True, always use Whisper's language detection
                even if confidence is below threshold (US-124-006). Default False.
            use_consensus_detection: When True, runs multiple detection libraries and uses
                consensus voting for more accurate detection (US-124-006). Default False.
            language_detection_weighted_fallback: When True, combines Whisper, langdetect, and
                langid using weighted confidence for more robust detection (US-137-009).
                Default True.
            min_combined_confidence: Minimum combined confidence threshold for weighted fallback
                (US-137-009). Default 0.7. When weighted confidence is below this, falls back
                to Whisper alone.
            post_processing: When True, applies intelligent segment post-processing including
                merging segments with same speaker and splitting at natural language boundaries
                (US-124-011). Default True.

        Returns:
            List of segment dicts with 'start', 'end', 'text', optionally 'words',
            and language detection info ('language', 'language_confidence', 'language_source')
        """
        # Get audio duration and determine model to use (US-110-010)
        # Also determine streaming mode based on duration (US-124-002)
        effective_model = self.model_name
        model_selection_reason = "manual model_selection"
        audio_duration = 0.0

        if self.auto_model_selection:
            # Import here to avoid circular imports
            from .utils import get_audio_duration
            audio_duration = get_audio_duration(audio_path) or 0.0
            duration_based_model, model_selection_reason = _select_model_for_duration(
                audio_duration, self.model_name
            )
            effective_model = duration_based_model
            logger.info(
                f"Auto model selection for {Path(audio_path).name[:40]}: "
                f"selected '{effective_model}' ({model_selection_reason}), "
                f"original config: '{self.model_name}'"
            )

        # Auto-detect streaming mode for long audio (>30 minutes) (US-124-002)
        # If streaming is explicitly set, use that value; otherwise auto-detect
        use_streaming = streaming
        if use_streaming is None:
            # Auto-detect: enable streaming for audio >30 minutes
            use_streaming = audio_duration > DURATION_THRESHOLD_MEDIUM  # 30 minutes

        streaming_enabled = use_streaming
        if streaming_enabled:
            logger.info(
                f"Streaming mode enabled for {Path(audio_path).name[:40]}: "
                f"audio_duration={audio_duration:.1f}s, chunk_length={chunk_length}s"
            )

        logger.debug(f"Acquiring GPU lock for {Path(audio_path).name[:40]}...")

        with _gpu_lock:
            logger.debug("Lock acquired, getting model...")
            # Pass effective_model to get_model to ensure correct model is loaded
            model = self.get_model(effective_model)

            audio_name = Path(audio_path).name[:40]
            logger.debug(f"Starting transcription of {audio_name}...")

            try:
                # Run model.transcribe() with timeout guard (US-79-002)
                # GPU calls can hang indefinitely; this prevents blocking the pipeline
                start_time = time.monotonic()

                # Track initial memory for streaming mode (US-124-002)
                initial_memory_mb = 0.0
                if streaming_enabled:
                    mem_alloc, mem_reserved = _get_gpu_memory_mb()
                    initial_memory_mb = mem_alloc
                    logger.debug(f"Streaming mode initial memory: {initial_memory_mb:.1f}MB")

                def _do_transcribe():
                    # Build transcribe kwargs - add chunk_length for streaming (US-124-002)
                    transcribe_kwargs = {
                        'language': language,
                        'vad_filter': vad_filter,
                        'vad_parameters': dict(
                            min_silence_duration_ms=min_silence_duration_ms,
                            speech_pad_ms=speech_pad_ms
                        ),
                        'word_timestamps': word_timestamps
                    }

                    # Add chunk_length for streaming mode to process audio in chunks
                    # This keeps memory usage constant for long audio files
                    if streaming_enabled:
                        transcribe_kwargs['chunk_length'] = chunk_length
                        logger.debug(f"Using chunk_length={chunk_length}s for streaming transcription")

                    # Add vocabulary hints via prompt parameter (US-124-005)
                    # faster-whisper supports prompt to guide transcription with context
                    if vocabulary:
                        vocab_text = ', '.join(vocabulary)
                        # Build a prompt that includes vocabulary hints
                        # This helps faster-whisper recognize domain-specific terms
                        vocab_prompt = f"Vocabulary hints: {vocab_text}."
                        # If there's an existing prompt, append vocabulary hints
                        if transcribe_kwargs.get('prompt'):
                            transcribe_kwargs['prompt'] = f"{transcribe_kwargs['prompt']} {vocab_prompt}"
                        else:
                            transcribe_kwargs['prompt'] = vocab_prompt
                        logger.debug(f"Using vocabulary hints: {vocabulary}")

                    return model.transcribe(audio_path, **transcribe_kwargs)

                timeout = self.gpu_transcription_timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_transcribe)
                    try:
                        segments, info = future.result(timeout=timeout)
                    except FuturesTimeoutError:
                        elapsed = time.monotonic() - start_time
                        logger.warning(
                            f"GPU transcription timeout for {audio_name}: "
                            f"elapsed={elapsed:.1f}s, timeout={timeout}s"
                        )
                        from src.transcription.exceptions import TransientTranscriptionError
                        raise TransientTranscriptionError(
                            f"GPU transcription timed out after {elapsed:.1f}s "
                            f"(limit: {timeout}s) for {audio_name}"
                        )

                logger.debug("Transcription done, processing segments...")
                result = []
                segment_confidences = []  # Track segment confidences for quality metrics

                for seg in segments:
                    seg_data = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip()
                    }

                    # Extract segment-level confidence from avg_logprob (US-110-007)
                    seg_logprob = getattr(seg, 'avg_logprob', None)
                    if seg_logprob is not None and isinstance(seg_logprob, (int, float)):
                        # Convert log probability to confidence (0-1)
                        seg_confidence = max(0.0, min(1.0, pow(2, seg_logprob)))
                        seg_data['segment_confidence'] = seg_confidence
                        segment_confidences.append(seg_confidence)

                    # Include word-level timestamps and confidence if available (US-110-007)
                    if word_timestamps and hasattr(seg, 'words') and seg.words:
                        words_data = []
                        word_confidences = []
                        for w in seg.words:
                            word_info = {"word": w.word, "start": w.start, "end": w.end}
                            # Word-level confidence (probability)
                            if hasattr(w, 'probability'):
                                prob = w.probability
                                # Only add numeric confidence values (not Mock objects)
                                if isinstance(prob, (int, float)):
                                    word_info['confidence'] = prob
                                    word_confidences.append(prob)
                            words_data.append(word_info)

                        seg_data["words"] = words_data

                        # Calculate average word confidence for this segment
                        if word_confidences:
                            seg_data['avg_word_confidence'] = sum(word_confidences) / len(word_confidences)

                    result.append(seg_data)

                # Language detection (US-110-003, US-124-006)
                # Extract Whisper's language detection
                whisper_language = getattr(info, 'language', None) or 'en'
                whisper_confidence_raw = getattr(info, 'language_probability', None)
                # Ensure we have a valid numeric confidence value (not a Mock or other type)
                whisper_confidence = whisper_confidence_raw if isinstance(whisper_confidence_raw, (int, float)) else 1.0

                # Determine final language and confidence
                detected_language = whisper_language
                detected_confidence = whisper_confidence
                language_source = 'whisper'

                # Combine all text for language detection
                combined_text = ' '.join(seg.get('text', '') for seg in result if seg.get('text'))

                # Determine if fallback is needed (US-124-006)
                need_fallback = not prefer_whisper_over_fallback and whisper_confidence < min_language_confidence

                # Use consensus detection if enabled (US-124-006)
                if use_consensus_detection and result and combined_text.strip():
                    consensus_lang, consensus_conf = _detect_language_consensus(combined_text)
                    if consensus_lang:
                        detected_language = consensus_lang
                        detected_confidence = consensus_conf
                        language_source = 'consensus'
                        record_language_detection('consensus')
                        logger.info(
                            f"Consensus language detection: {consensus_lang} (confidence: {consensus_conf:.2f})"
                        )
                    need_fallback = False  # Consensus already handled

                # Use weighted fallback if enabled (US-137-009)
                if language_detection_weighted_fallback and result and combined_text.strip():
                    weighted_lang, weighted_conf, detection_breakdown = _detect_language_weighted(
                        combined_text, whisper_language, whisper_confidence, min_combined_confidence
                    )
                    if weighted_lang:
                        # Log detection breakdown
                        logger.info(
                            f"Weighted language detection: {weighted_lang} "
                            f"(combined: {weighted_conf:.2f}, breakdown: Whisper={detection_breakdown['whisper']:.2f}, "
                            f"langdetect={detection_breakdown['langdetect']:.2f}, langid={detection_breakdown['langid']:.2f})"
                        )
                        # Use weighted result if confidence meets threshold
                        if weighted_conf >= min_combined_confidence:
                            detected_language = weighted_lang
                            detected_confidence = weighted_conf
                            language_source = 'weighted'
                            record_language_detection('weighted')
                        else:
                            # Weighted confidence below threshold, keep Whisper but log warning
                            logger.warning(
                                f"Weighted confidence ({weighted_conf:.2f}) below threshold "
                                f"({min_combined_confidence:.2f}). Using Whisper alone: {whisper_language}"
                            )
                    need_fallback = False  # Weighted fallback already handled

                # Fallback to configured library if Whisper confidence is below threshold (US-124-006)
                if need_fallback and result and combined_text.strip():
                    # Check if fallback library is disabled
                    if language_detection_library is None:
                        logger.debug("Fallback language detection disabled (library=None)")
                    else:
                        fallback_lang, fallback_conf = _detect_language_with_library(
                            combined_text, language_detection_library
                        )
                        if fallback_lang:
                            detected_language = fallback_lang
                            detected_confidence = fallback_conf
                            language_source = language_detection_library
                            record_language_detection(language_detection_library)
                            logger.warning(
                                f"Whisper language confidence ({whisper_confidence:.2f}) below threshold "
                                f"({min_language_confidence:.2f}). Using {language_detection_library} fallback: {fallback_lang}"
                            )

                # Log warning if language detection confidence is still below threshold
                if detected_confidence < min_language_confidence:
                    logger.warning(
                        f"Language detection confidence ({detected_confidence:.2f}) below threshold "
                        f"({min_language_confidence:.2f}) for {audio_name}. "
                        f"Source: {language_source}, Language: {detected_language}"
                    )

                # Record language detection method (US-137-009)
                # Only record if not already recorded by specific detection
                if language_source == 'whisper':
                    record_language_detection('whisper')

                # Add language info to each segment for caching
                for seg in result:
                    seg['language'] = detected_language
                    seg['language_confidence'] = detected_confidence
                    seg['language_source'] = language_source

                # Verbose logging for transcription result
                total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in result)
                logger.debug(f"Transcription complete: {audio_name}")
                logger.debug(f"  Segments: {len(result)}, Total duration: {total_duration:.1f}s")
                if result:
                    logger.debug(f"  First segment: '{result[0].get('text', '')[:50]}...'")
                    logger.debug(f"  Word timestamps: {'yes' if result[0].get('words') else 'no'}")

                # Log memory usage for streaming mode (US-124-002)
                if streaming_enabled:
                    mem_alloc, mem_reserved = _get_gpu_memory_mb()
                    memory_delta = mem_alloc - initial_memory_mb
                    logger.info(
                        f"Streaming mode memory check for {audio_name}: "
                        f"initial={initial_memory_mb:.1f}MB, final={mem_alloc:.1f}MB, "
                        f"delta={memory_delta:+.1f}MB"
                    )

                # Calculate quality metrics (US-110-007)
                quality_metrics = _calculate_quality_metrics(result)

                # Confidence-based retry logic (US-124-003)
                # Check if we need to retry due to low confidence
                if retry_on_low_confidence and quality_metrics:
                    min_seg_conf = quality_metrics.get('min_segment_confidence', 1.0)
                    avg_word_conf = quality_metrics.get('avg_word_confidence', 1.0)

                    if min_seg_conf < min_confidence_threshold or avg_word_conf < min_confidence_threshold:
                        logger.info(
                            f"Low confidence detected for {audio_name}: "
                            f"min_segment_confidence={min_seg_conf:.3f}, "
                            f"avg_word_confidence={avg_word_conf:.3f}, "
                            f"threshold={min_confidence_threshold:.3f}. "
                            f"Retrying with enhanced settings..."
                        )

                        retry_count = 0
                        best_result = result
                        best_quality_metrics = quality_metrics

                        while retry_count < max_retries:
                            retry_count += 1

                            # Retry with word_timestamps=True for better accuracy
                            # and vad_filter=True for better speech detection
                            logger.info(
                                f"Retry {retry_count}/{max_retries} for {audio_name} "
                                f"(word_timestamps=True, vad_filter=True)"
                            )

                            try:
                                # Re-run transcription with enhanced settings
                                retry_start_time = time.monotonic()

                                def _do_retry_transcribe():
                                    return model.transcribe(
                                        audio_path,
                                        language=language,
                                        vad_filter=True,  # Enable VAD for better segment detection
                                        vad_parameters=dict(
                                            min_silence_duration_ms=min_silence_duration_ms,
                                            speech_pad_ms=speech_pad_ms
                                        ),
                                        word_timestamps=True  # Enable word timestamps for better accuracy
                                    )

                                retry_timeout = self.gpu_transcription_timeout
                                with ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(_do_retry_transcribe)
                                    try:
                                        retry_segments, retry_info = future.result(timeout=retry_timeout)
                                    except FuturesTimeoutError:
                                        elapsed = time.monotonic() - retry_start_time
                                        logger.warning(
                                            f"Retry transcription timeout for {audio_name}: "
                                            f"elapsed={elapsed:.1f}s, timeout={retry_timeout}s"
                                        )
                                        break

                                # Process retry result
                                retry_result = []
                                for seg in retry_segments:
                                    seg_data = {
                                        "start": seg.start,
                                        "end": seg.end,
                                        "text": seg.text.strip()
                                    }

                                    seg_logprob = getattr(seg, 'avg_logprob', None)
                                    if seg_logprob is not None and isinstance(seg_logprob, (int, float)):
                                        seg_confidence = max(0.0, min(1.0, pow(2, seg_logprob)))
                                        seg_data['segment_confidence'] = seg_confidence

                                    if hasattr(seg, 'words') and seg.words:
                                        words_data = []
                                        word_confidences = []
                                        for w in seg.words:
                                            word_info = {"word": w.word, "start": w.start, "end": w.end}
                                            if hasattr(w, 'probability'):
                                                prob = w.probability
                                                # Only add numeric confidence values (not Mock objects)
                                                if isinstance(prob, (int, float)):
                                                    word_info['confidence'] = prob
                                                    word_confidences.append(prob)
                                            words_data.append(word_info)

                                        seg_data["words"] = words_data
                                        if word_confidences:
                                            seg_data['avg_word_confidence'] = sum(word_confidences) / len(word_confidences)

                                    # Add language info from original result
                                    seg_data['language'] = detected_language
                                    seg_data['language_confidence'] = detected_confidence
                                    seg_data['language_source'] = language_source

                                    retry_result.append(seg_data)

                                # Calculate quality metrics for retry result
                                retry_quality_metrics = _calculate_quality_metrics(retry_result)

                                min_seg_conf_new = retry_quality_metrics.get('min_segment_confidence', 1.0)
                                avg_word_conf_new = retry_quality_metrics.get('avg_word_confidence', 1.0)

                                logger.info(
                                    f"Retry {retry_count} complete for {audio_name}: "
                                    f"min_segment_confidence={min_seg_conf_new:.3f}, "
                                    f"avg_word_confidence={avg_word_conf_new:.3f}"
                                )

                                # Keep the result with better confidence
                                if min_seg_conf_new > min_seg_conf:
                                    best_result = retry_result
                                    best_quality_metrics = retry_quality_metrics
                                    min_seg_conf = min_seg_conf_new

                                    # Check if we've met the threshold
                                    if min_seg_conf >= min_confidence_threshold:
                                        logger.info(
                                            f"Confidence threshold met after retry {retry_count} for {audio_name}"
                                        )
                                        break
                                else:
                                    # If this retry didn't improve, try different settings on next iteration
                                    if retry_count < max_retries:
                                        logger.debug(
                                            f"Retry {retry_count} did not improve confidence, "
                                            f"will try additional settings if available"
                                        )

                            except Exception as retry_error:
                                logger.warning(
                                    f"Retry {retry_count} failed for {audio_name}: {retry_error}"
                                )
                                break

                        # Use best result found
                        if best_result is not result:
                            logger.info(
                                f"Using improved transcription from retry for {audio_name}: "
                                f"min_segment_confidence={best_quality_metrics.get('min_segment_confidence', 0):.3f}"
                            )
                            result = best_result
                            quality_metrics = best_quality_metrics
                        else:
                            logger.warning(
                                f"Retries did not improve confidence for {audio_name}, "
                                f"using original result"
                            )

                logger.debug(f"Done: {len(result)} segments")

                # Quality gate check (US-137-005)
                # Check if transcription meets minimum quality threshold
                if quality_gate_enabled and quality_metrics:
                    avg_word_conf = quality_metrics.get('avg_word_confidence', 1.0)

                    if avg_word_conf < min_quality_threshold:
                        # Log quality gate failure with segment-level detail
                        low_quality_segments = []
                        for seg in result:
                            seg_conf = seg.get('segment_confidence', 0)
                            seg_text = seg.get('text', '')
                            seg_avg_word_conf = seg.get('avg_word_confidence', 0)
                            low_quality_segments.append({
                                'text': seg_text[:100],  # Truncate for logging
                                'segment_confidence': seg_conf,
                                'avg_word_confidence': seg_avg_word_conf,
                                'start': seg.get('start', 0),
                                'end': seg.get('end', 0)
                            })

                        logger.warning(
                            f"Quality gate FAILED for {audio_name}: "
                            f"avg_word_confidence={avg_word_conf:.3f} < min_quality_threshold={min_quality_threshold:.3f}. "
                            f"Segments below threshold: {len(low_quality_segments)}"
                        )

                        # Try retry with alternative settings
                        retry_result = self._retry_with_alternative_settings(
                            audio_path=audio_path,
                            language=language,
                            vad_filter=vad_filter,
                            min_silence_duration_ms=min_silence_duration_ms,
                            speech_pad_ms=speech_pad_ms,
                            original_result=result,
                            original_quality_metrics=quality_metrics,
                            min_quality_threshold=min_quality_threshold
                        )

                        if retry_result is not None:
                            result, quality_metrics = retry_result
                            avg_word_conf = quality_metrics.get('avg_word_confidence', 1.0)

                            # Final check after retry
                            if avg_word_conf < min_quality_threshold:
                                # Raise QualityGateError with segment details
                                from .exceptions import QualityGateError
                                raise QualityGateError(
                                    f"Transcription quality gate failed for {audio_name}: "
                                    f"avg_word_confidence={avg_word_conf:.3f} < min_quality_threshold={min_quality_threshold:.3f} "
                                    f"(even after retry with alternative settings)",
                                    avg_confidence=avg_word_conf,
                                    min_threshold=min_quality_threshold,
                                    segment_details=low_quality_segments
                                )
                        else:
                            # Retry returned None - raise quality gate error
                            from .exceptions import QualityGateError
                            raise QualityGateError(
                                f"Transcription quality gate failed for {audio_name}: "
                                f"avg_word_confidence={avg_word_conf:.3f} < min_quality_threshold={min_quality_threshold:.3f}",
                                avg_confidence=avg_word_conf,
                                min_threshold=min_quality_threshold,
                                segment_details=low_quality_segments
                            )
                    else:
                        logger.info(
                            f"Quality gate PASSED for {audio_name}: "
                            f"avg_word_confidence={avg_word_conf:.3f} >= min_quality_threshold={min_quality_threshold:.3f}"
                        )

                # Apply segment post-processing (US-124-011)
                if post_processing:
                    from .utils import post_process_segments
                    result = post_process_segments(result)

                return result, quality_metrics

            except Exception as e:
                # Re-raise TransientTranscriptionError and QualityGateError without catching them
                from src.transcription.exceptions import TransientTranscriptionError, QualityGateError, is_gpu_error
                if isinstance(e, (TransientTranscriptionError, QualityGateError)):
                    raise

                # GPU-to-CPU fallback (US-110-004)
                # Check if this is a GPU error and fallback is enabled
                if self.auto_fallback_to_cpu and not self._cpu_fallback_mode and is_gpu_error(e):
                    logger.warning(f"GPU error detected during transcription: {e}")
                    logger.warning("Attempting automatic fallback to CPU compute type...")

                    # Force CPU mode and retry
                    self.force_cpu_mode()

                    # Retry the transcription with CPU
                    try:
                        logger.info(f"Retrying transcription on CPU for {audio_name}...")
                        # Re-get the model (will now be CPU)
                        model = self.get_model()

                        # Re-run transcription with CPU
                        start_time = time.monotonic()

                        def _do_transcribe_cpu():
                            return model.transcribe(
                                audio_path,
                                language=language,
                                vad_filter=vad_filter,
                                vad_parameters=dict(
                                    min_silence_duration_ms=min_silence_duration_ms,
                                    speech_pad_ms=speech_pad_ms
                                ),
                                word_timestamps=word_timestamps
                            )

                        timeout = self.gpu_transcription_timeout
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(_do_transcribe_cpu)
                            try:
                                segments, info = future.result(timeout=timeout)
                            except FuturesTimeoutError:
                                elapsed = time.monotonic() - start_time
                                logger.warning(
                                    f"CPU transcription timeout for {audio_name}: "
                                    f"elapsed={elapsed:.1f}s, timeout={timeout}s"
                                )
                                from src.transcription.exceptions import TransientTranscriptionError
                                raise TransientTranscriptionError(
                                    f"CPU transcription timed out after {elapsed:.1f}s "
                                    f"(limit: {timeout}s) for {audio_name}"
                                )

                        # Process the result from CPU transcription
                        result = []
                        for seg in segments:
                            seg_data = {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip()
                            }
                            if word_timestamps and hasattr(seg, 'words') and seg.words:
                                seg_data["words"] = [
                                    {"word": w.word, "start": w.start, "end": w.end}
                                    for w in seg.words
                                ]
                            result.append(seg_data)

                        # Language detection (US-110-003, US-124-006) - same as GPU path
                        whisper_language = getattr(info, 'language', None) or 'en'
                        whisper_confidence = getattr(info, 'language_probability', 1.0) or 1.0
                        detected_language = whisper_language
                        detected_confidence = whisper_confidence
                        language_source = 'whisper'

                        # Combine all text for language detection
                        combined_text = ' '.join(seg.get('text', '') for seg in result if seg.get('text'))

                        # Determine if fallback is needed (US-124-006)
                        need_fallback = not prefer_whisper_over_fallback and whisper_confidence < min_language_confidence

                        # Use consensus detection if enabled (US-124-006)
                        if use_consensus_detection and result and combined_text.strip():
                            consensus_lang, consensus_conf = _detect_language_consensus(combined_text)
                            if consensus_lang:
                                detected_language = consensus_lang
                                detected_confidence = consensus_conf
                                language_source = 'consensus'
                                record_language_detection('consensus')
                            need_fallback = False

                        # Use weighted fallback if enabled (US-137-009)
                        if language_detection_weighted_fallback and result and combined_text.strip():
                            weighted_lang, weighted_conf, detection_breakdown = _detect_language_weighted(
                                combined_text, whisper_language, whisper_confidence, min_combined_confidence
                            )
                            if weighted_lang:
                                logger.info(
                                    f"Weighted language detection: {weighted_lang} "
                                    f"(combined: {weighted_conf:.2f}, breakdown: Whisper={detection_breakdown['whisper']:.2f}, "
                                    f"langdetect={detection_breakdown['langdetect']:.2f}, langid={detection_breakdown['langid']:.2f})"
                                )
                                if weighted_conf >= min_combined_confidence:
                                    detected_language = weighted_lang
                                    detected_confidence = weighted_conf
                                    language_source = 'weighted'
                                    record_language_detection('weighted')
                                else:
                                    logger.warning(
                                        f"Weighted confidence ({weighted_conf:.2f}) below threshold "
                                        f"({min_combined_confidence:.2f}). Using Whisper alone: {whisper_language}"
                                    )
                            need_fallback = False

                        # Fallback to configured library if needed (US-124-006)
                        if need_fallback and result and combined_text.strip():
                            if language_detection_library is None:
                                logger.debug("Fallback language detection disabled")
                            else:
                                fallback_lang, fallback_conf = _detect_language_with_library(
                                    combined_text, language_detection_library
                                )
                                if fallback_lang:
                                    detected_language = fallback_lang
                                    detected_confidence = fallback_conf
                                    language_source = language_detection_library
                                    record_language_detection(language_detection_library)

                        # Record Whisper usage (US-137-009)
                        if language_source == 'whisper':
                            record_language_detection('whisper')

                        for seg in result:
                            seg['language'] = detected_language
                            seg['language_confidence'] = detected_confidence
                            seg['language_source'] = language_source

                        total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in result)
                        logger.info(f"CPU transcription successful for {audio_name}: {len(result)} segments, {total_duration:.1f}s")

                        # Apply segment post-processing (US-124-011)
                        if post_processing:
                            from .utils import post_process_segments
                            result = post_process_segments(result)

                        return result

                    except Exception as cpu_error:
                        # CPU fallback also failed - log and continue with empty result
                        logger.error(f"CPU fallback transcription also failed: {cpu_error}")
                        import traceback
                        traceback.print_exc()

                logger.error(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()
                return []

    def cleanup(self):
        """
        Unload the shared whisper model and free GPU memory.

        Call this after batch transcription is complete to reclaim memory
        for subsequent pipeline stages (embedding, matching).
        """
        global _shared_model, _model_config
        import gc

        with _gpu_lock:
            if _shared_model is not None:
                # Log GPU memory before cleanup
                mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
                logger.info(f"GPU memory before cleanup: allocated={mem_before_alloc:.1f}MB, reserved={mem_before_reserved:.1f}MB")

                logger.info("Unloading transcription model to free memory...")
                del _shared_model
                _shared_model = None
                _model_config = {}

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.debug("Cleared CUDA cache")
                except ImportError:
                    pass

                # Log GPU memory after cleanup with delta
                mem_after_alloc, mem_after_reserved = _get_gpu_memory_mb()
                mem_delta = mem_before_alloc - mem_after_alloc
                logger.info(f"GPU memory after cleanup: allocated={mem_after_alloc:.1f}MB, reserved={mem_after_reserved:.1f}MB")
                logger.info(f"GPU memory freed by cleanup: {mem_delta:.1f}MB")

                logger.info("Transcription model unloaded")

    def force_cpu_mode(self):
        """
        Force CPU compute type for subsequent transcriptions (US-110-004).

        This method clears the current GPU model and marks the client to use
        CPU for all future transcriptions. Used when GPU transcription fails
        due to CUDA errors or memory issues.
        """
        global _shared_model, _model_config
        import gc

        if self._cpu_fallback_mode:
            logger.debug("Already in CPU fallback mode")
            return

        logger.warning("GPU transcription failed - falling back to CPU compute type")

        with _gpu_lock:
            # Unload the current GPU model
            if _shared_model is not None:
                logger.info("Unloading GPU model for CPU fallback...")
                del _shared_model
                _shared_model = None
                _model_config = {}

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

            # Mark as CPU fallback mode
            self._cpu_fallback_mode = True
            # Use CPU compute type
            self.compute_type = "int8"
            logger.info("CPU fallback mode enabled - future transcriptions will use CPU")

    def start_memory_monitor(self) -> None:
        """
        Start GPU memory monitoring thread (US-137-010).

        Should be called before starting a transcription batch to enable
        continuous memory monitoring during batch processing.
        """
        if not self.gpu_memory_monitoring_enabled:
            logger.debug("GPU memory monitoring disabled in config")
            return

        # Start the background monitoring thread
        monitor_gpu_memory(
            enabled=True,
            threshold_percent=self.gpu_memory_threshold_percent,
            check_interval=self.memory_check_interval_seconds,
            callback=None  # Could add callback for real-time alerts
        )
        logger.info(
            f"GPU memory monitoring started "
            f"(threshold: {self.gpu_memory_threshold_percent}%, "
            f"interval: {self.memory_check_interval_seconds}s)"
        )

    def stop_memory_monitor(self) -> None:
        """
        Stop GPU memory monitoring thread (US-137-010).

        Should be called after batch transcription is complete.
        """
        stop_gpu_memory_monitor()
        logger.info("GPU memory monitoring stopped")

    def get_memory_stats(self) -> dict:
        """
        Get GPU memory monitoring statistics (US-137-010).

        Returns:
            Dict with current memory stats and history
        """
        return get_gpu_memory_stats()

    def check_and_handle_memory_pressure(self, current_batch_size: int = 50) -> tuple[bool, int, str]:
        """
        Check for GPU memory pressure and handle according to configured strategy (US-137-010).

        Args:
            current_batch_size: Current batch size

        Returns:
            Tuple of (should_continue, suggested_batch_size, message)
            - should_continue: True if can continue, False if should fallback to CPU
            - suggested_batch_size: Recommended batch size (may be reduced)
            - message: Status message
        """
        if not self.gpu_memory_monitoring_enabled:
            return True, current_batch_size, "Memory monitoring disabled"

        success, batch_size, message = handle_memory_pressure(
            strategy=self.memory_recovery_strategy,
            threshold_percent=self.gpu_memory_threshold_percent,
            max_wait_seconds=self.memory_recovery_max_wait_seconds,
            check_interval=self.memory_check_interval_seconds,
            reduction_factor=self.memory_batch_reduction_factor,
            current_batch_size=current_batch_size,
            min_batch_size=self.min_batch_size_under_pressure,
            logger=logger
        )

        if not success and self.auto_fallback_to_cpu:
            # Strategy failed, fallback to CPU
            logger.warning(f"Memory pressure handling failed: {message}. Falling back to CPU.")
            self.force_cpu_mode()
            return False, batch_size, f"{message} - CPU fallback enabled"

        return success, batch_size, message

    def get_memory_aware_batch_size(self, base_batch_size: int) -> int:
        """
        Get batch size adjusted for current GPU memory usage (US-137-010).

        Args:
            base_batch_size: Original batch size from config

        Returns:
            Adjusted batch size based on current memory usage
        """
        if not self.gpu_memory_monitoring_enabled:
            return base_batch_size

        memory_percent = get_gpu_memory_percent()
        if memory_percent < 0:
            return base_batch_size

        return calculate_memory_aware_batch_size(
            base_batch_size=base_batch_size,
            memory_percent=memory_percent,
            threshold_percent=self.gpu_memory_threshold_percent,
            reduction_factor=self.memory_batch_reduction_factor,
            min_batch_size=self.min_batch_size_under_pressure
        )


# Module-level cleanup function for backward compatibility
def cleanup_model():
    """
    Unload the shared whisper model and free GPU memory.

    Backward-compatible function that wraps WhisperClient.cleanup().
    """
    client = WhisperClient()
    client.cleanup()
