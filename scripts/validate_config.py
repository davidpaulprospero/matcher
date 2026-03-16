#!/usr/bin/env python3
"""
Configuration Integration Validation Script

Validates that config.yaml is properly integrated with all components.

Usage:
    python validate_config.py           # Full validation
    python validate_config.py --quick   # Quick check
    python validate_config.py --perf    # Performance benchmark
"""

import os
import sys
import time
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path (parent of scripts directory)
# Use multiple methods to ensure robustness when called from different working directories
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths (config.yaml) work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_header, print_ok, print_warn, print_error, set_verbosity, get_verbosity
from script_utils import load_config_for_script, add_config_argument

# Global warnings tracker for strict mode
_warnings: List[str] = []


def _add_warning(warning: str) -> None:
    """Add a warning message (tracked for strict mode)"""
    _warnings.append(warning)
    print_warn(warning)


def _get_warnings() -> List[str]:
    """Get all warnings"""
    return _warnings.copy()


def _clear_warnings() -> None:
    """Clear warnings list"""
    _warnings.clear()


def validate_config_loading(config_path: str = "config.yaml", skip_validation: bool = False) -> Any:
    """Test config loading performance and correctness"""
    print_header("CONFIG LOADING")

    verbose: bool = get_verbosity() >= 2

    from src.config import get_config, get_config_metrics

    # Use shared config loader (handles missing file gracefully)
    config: Any = load_config_for_script(config_path, required=True, skip_validation=skip_validation)
    if config is None:
        # Error already printed by load_config_for_script
        sys.exit(1)

    # Test loading performance (reload to measure load time)
    start: float = time.perf_counter()
    from src.config import Config
    Config.from_yaml(config_path, skip_final_validation=skip_validation)
    load_time: float = (time.perf_counter() - start) * 1000

    if load_time < 100:
        print_ok(f"Config loaded in {load_time:.1f}ms (target: <100ms)")
    else:
        print_warn(f"Config loaded in {load_time:.1f}ms (target: <100ms)")

    # Check metrics
    metrics: Dict[str, int] = get_config_metrics()
    print_ok(f"Load count: {metrics['load_count']}")

    if verbose:
        # Verbose: show more details
        print(f"    → Config path: {config._config_path}")
        print(f"    → Config hash: {config._config_hash}")
        print(f"    → Cache hits: {metrics.get('cache_hits', 0)}")

    # Check CSafeLoader
    try:
        from yaml import CSafeLoader
        print_ok("Using CSafeLoader (C-based, optimized)")
    except ImportError:
        print_warn("CSafeLoader not available (using pure Python)")

    return config


def validate_config_structure(config: Any) -> bool:
    """Validate config structure has all required sections"""
    print_header("CONFIG STRUCTURE")

    required_sections: List[Tuple[str, str]] = [
        ('project', 'ProjectConfig'),
        ('transcription', 'TranscriptionConfig'),
        ('embedding', 'EmbeddingConfig'),
        ('indexing', 'IndexingConfig'),
        ('vision', 'VisionConfig'),
        ('scene_detection', 'SceneDetectionConfig'),
        ('matching', 'MatchingConfig'),
        ('keyword', 'KeywordConfig'),
        ('enhanced', 'EnhancedFeaturesConfig'),
        ('downloading', 'DownloadingConfig'),
        ('output', 'OutputConfig'),
        ('logging', 'LoggingConfig'),
        ('cache', 'CacheConfig'),
        ('pipeline', 'PipelineConfig'),
    ]

    all_valid: bool = True
    for section_name, expected_type in required_sections:
        if hasattr(config, section_name):
            section: Any = getattr(config, section_name)
            actual_type: str = type(section).__name__
            print_ok(f"{section_name}: {actual_type}")
        else:
            print_error(f"{section_name}: MISSING")
            all_valid = False

    return all_valid


def validate_api_keys(config: Any) -> None:
    """Validate API keys are available"""
    print_header("API KEYS")

    verbose: bool = get_verbosity() >= 2

    api_keys: Dict[str, Tuple[str, Optional[str]]] = {
        'GEMINI_API_KEY': ('gemini', config.api_keys.gemini_api_key),
        'ANTHROPIC_API_KEY': ('anthropic', config.api_keys.anthropic_api_key),
        'PEXELS_API_KEY': ('pexels', config.api_keys.pexels_api_key),
        'PIXABAY_API_KEY': ('pixabay', config.api_keys.pixabay_api_key),
    }

    for env_name, (provider, key) in api_keys.items():
        if key:
            if verbose:
                # Verbose: show masked key for verification
                masked: str = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
                print_ok(f"{env_name}: Set ({len(key)} chars) - {masked}")
            else:
                print_ok(f"{env_name}: Set ({len(key)} chars)")
        else:
            # Check if required
            required_checks: Dict[str, bool] = {
                'GEMINI_API_KEY': config.embedding.provider == 'gemini' or config.matching.primary_provider == 'gemini',
                'ANTHROPIC_API_KEY': config.matching.secondary_provider == 'anthropic',
                'PEXELS_API_KEY': config.stock_footage.pexels_enabled,
                'PIXABAY_API_KEY': config.stock_footage.pixabay_enabled,
            }

            if required_checks.get(env_name, False):
                print_warn(f"{env_name}: Not set (required for {provider})")
            else:
                print_ok(f"{env_name}: Not set (optional)")

    # Verbose: show why each key is required
    if verbose:
        print("\n  --- Provider Configuration ---")
        print(f"    embedding.provider: {config.embedding.provider}")
        print(f"    matching.primary_provider: {config.matching.primary_provider}")
        print(f"    matching.secondary_provider: {config.matching.secondary_provider}")
        print(f"    stock_footage.pexels_enabled: {config.stock_footage.pexels_enabled}")
        print(f"    stock_footage.pixabay_enabled: {config.stock_footage.pixabay_enabled}")


def validate_paths(config: Any) -> None:
    """Validate configured paths"""
    print_header("PATHS")

    verbose: bool = get_verbosity() >= 2

    paths: List[Tuple[str, str]] = [
        ('project_dir', config.project_dir),
        ('downloaded_videos_dir', config.downloaded_videos_dir),
        ('otio_output_dir', config.otio_output_dir),
        ('cache.cache_dir', config.cache.cache_dir),
        ('logging.log_dir', config.logging.log_dir),
    ]

    for name, path in paths:
        p: Path = Path(path)
        if p.exists():
            if verbose:
                # Verbose: show absolute path and stats
                abs_path: str = str(p.resolve())
                is_absolute: bool = p.is_absolute()
                print_ok(f"{name}: {path} (exists)")
                print(f"    → Absolute: {abs_path}, Is Absolute: {is_absolute}")
            else:
                print_ok(f"{name}: {path} (exists)")
        else:
            # Check parent directory
            parent_exists: bool = p.parent.exists()
            if verbose:
                print_warn(f"{name}: {path} (will be created)")
                print(f"    → Parent exists: {parent_exists}, Parent: {p.parent}")
            else:
                print_warn(f"{name}: {path} (will be created)")


def validate_value_ranges(config: Any) -> bool:
    """Validate config values are in valid ranges"""
    print_header("VALUE RANGES")

    verbose: bool = get_verbosity() >= 2

    checks: List[Tuple[bool, str, str]] = [
        (0 <= config.matching.min_confidence <= 1,
         f"matching.min_confidence = {config.matching.min_confidence}",
         "must be 0-1"),

        (0 <= config.matching.high_confidence_threshold <= 1,
         f"matching.high_confidence_threshold = {config.matching.high_confidence_threshold}",
         "must be 0-1"),

        (config.matching.max_clip_reuse >= 0,
         f"matching.max_clip_reuse = {config.matching.max_clip_reuse}",
         "must be >= 0"),

        (config.transcription.max_workers >= 1,
         f"transcription.max_workers = {config.transcription.max_workers}",
         "must be >= 1"),

        (config.embedding.batch_size >= 1,
         f"embedding.batch_size = {config.embedding.batch_size}",
         "must be >= 1"),

        (config.keyword.max_keywords >= 1,
         f"keyword.max_keywords = {config.keyword.max_keywords}",
         "must be >= 1"),

        (config.enhanced.min_confidence >= 0,
         f"enhanced.min_confidence = {config.enhanced.min_confidence}",
         "must be >= 0"),
    ]

    all_valid: bool = True
    for valid, value_str, requirement in checks:
        if verbose:
            # Verbose: show requirement in output
            detail: str = f" ({requirement})" if not valid else ""
            if valid:
                print_ok(f"{value_str}{detail}")
            else:
                print_error(f"{value_str}{detail}")
                all_valid = False
        else:
            if valid:
                print_ok(value_str)
            else:
                print_error(f"{value_str} - {requirement}")
                all_valid = False

    return all_valid


def validate_nested_access(config: Any) -> bool:
    """Test nested config access"""
    print_header("NESTED ACCESS")

    test_paths: List[Tuple[str, Any]] = [
        ("matching.min_confidence", 0.7),
        ("transcription.model", "base"),
        ("embedding.provider", "gemini"),
        ("duration_tiers.short.min_seconds", 20),
    ]

    all_valid: bool = True
    for path, expected_type_or_value in test_paths:
        value: Any = config.get_nested(path)
        if value is not None:
            print_ok(f"{path} = {value}")
        else:
            print_error(f"{path} = NOT FOUND")
            all_valid = False

    return all_valid


def validate_config_validation(config: Any) -> bool:
    """Run config's built-in validation"""
    print_header("BUILT-IN VALIDATION")

    errors: List[str] = config.validate()

    if errors:
        print_warn(f"Validation warnings ({len(errors)}):")
        for error in errors:
            print(f"    • {error}")
        return False
    else:
        print_ok("All validation checks passed")
        return True


def performance_benchmark() -> None:
    """Benchmark config loading performance"""
    print_header("PERFORMANCE BENCHMARK")

    from src.config import Config

    # Warm up
    for _ in range(3):
        Config.from_yaml("config.yaml")

    # Benchmark
    iterations: int = 20
    times: List[float] = []

    for _ in range(iterations):
        start: float = time.perf_counter()
        Config.from_yaml("config.yaml")
        times.append((time.perf_counter() - start) * 1000)

    avg_time: float = sum(times) / len(times)
    min_time: float = min(times)
    max_time: float = max(times)

    print_ok(f"Average load time: {avg_time:.2f}ms")
    print_ok(f"Min load time: {min_time:.2f}ms")
    print_ok(f"Max load time: {max_time:.2f}ms")

    if avg_time < 100:
        print_ok("Performance target met (<100ms)")
    else:
        print_warn("Performance target not met (>100ms)")

    # Test get_config() caching
    from src.config import load_config, get_config, get_config_metrics

    load_config("config.yaml")  # Initial load

    cache_times: List[float] = []
    for _ in range(1000):
        start = time.perf_counter()
        get_config()
        cache_times.append((time.perf_counter() - start) * 1000)

    avg_cache_time: float = sum(cache_times) / len(cache_times)

    print_ok(f"Cached access time: {avg_cache_time * 1000:.3f}μs")

    metrics: Dict[str, int] = get_config_metrics()
    print_ok(f"Cache hits: {metrics['cache_hits']}")


def test_hot_reload() -> None:
    """Test hot reload capability"""
    print_header("HOT RELOAD")

    from src.config import load_config, reload_config
    import tempfile
    import shutil

    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("project:\n  name: test-project\n  version: 1.0.0\n")
        temp_path: str = f.name

    try:
        # Load initial config
        config: Any = load_config(temp_path)
        initial_hash: str = config._config_hash
        print_ok(f"Initial load: hash={initial_hash}")

        # Modify file
        time.sleep(0.1)
        with open(temp_path, 'w') as f:
            f.write("project:\n  name: modified-project\n  version: 2.0.0\n")

        # Reload
        reloaded: bool = config.reload()

        if reloaded:
            print_ok(f"Hot reload detected change: hash={config._config_hash}")
        else:
            print_warn("Hot reload did not detect change")

        # Verify no change when file unchanged
        reloaded_again: bool = config.reload()
        if not reloaded_again:
            print_ok("No reload when file unchanged")
        else:
            print_warn("Unnecessary reload triggered")

    finally:
        os.unlink(temp_path)


# =============================================================================
# NEW VALIDATION FUNCTIONS (US-131-009)
# =============================================================================

def validate_package_versions() -> bool:
    """Validate required Python package versions are installed"""
    print_header("PACKAGE VERSIONS")

    verbose: bool = get_verbosity() >= 2
    all_valid: bool = True

    # Required packages with minimum versions
    required_packages: Dict[str, Tuple[str, str]] = {
        'yaml': ('6.0', 'PyYAML'),
        'torch': ('2.0', 'torch'),
        'sentence_transformers': ('2.0', 'sentence-transformers'),
        'faster_whisper': ('1.0.0', 'faster-whisper'),
        'anthropic': ('0.50.0', 'anthropic'),
        'yt_dlp': ('2024.1.0', 'yt-dlp'),
        'scenedetect': ('0.6', 'scenedetect'),
        'opencv_python': ('4.8', 'opencv-python'),
        'librosa': ('0.10.0', 'librosa'),
        'imagehash': ('4.3.0', 'imagehash'),
        'PIL': ('9.0.0', 'Pillow'),
    }

    for module_name, (min_version, display_name) in required_packages.items():
        try:
            if module_name == 'yaml':
                import yaml
                # yaml is 6.x
                version = yaml.__version__
            elif module_name == 'PIL':
                from PIL import Image
                version = Image.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')

            if verbose:
                print_ok(f"{display_name}: {version} (required: >={min_version})")
            else:
                print_ok(f"{display_name}: {version}")
        except ImportError:
            print_error(f"{display_name}: NOT INSTALLED")
            all_valid = False
        except Exception as e:
            print_warn(f"{display_name}: Unable to determine version - {e}")

    return all_valid


def validate_external_tools() -> bool:
    """Validate yt-dlp and ffmpeg availability and versions"""
    print_header("EXTERNAL TOOLS")
    all_valid: bool = True
    verbose: bool = get_verbosity() >= 2

    # Check yt-dlp
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_ok(f"yt-dlp: {version}")
            if verbose:
                # Additional check for supported formats
                result2 = subprocess.run(
                    ['yt-dlp', '--list-extractors'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result2.returncode == 0:
                    extractors = len(result2.stdout.strip().split('\n'))
                    print(f"    → {extractors} extractors available")
        else:
            print_error(f"yt-dlp: Failed to get version")
            all_valid = False
    except FileNotFoundError:
        print_error("yt-dlp: NOT FOUND (install with: pip install yt-dlp)")
        all_valid = False
    except subprocess.TimeoutExpired:
        print_error("yt-dlp: Timeout checking version")
        all_valid = False

    # Check ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from first line: ffmpeg version 4.4.1 ...
            first_line = result.stdout.split('\n')[0]
            match = re.search(r'ffmpeg version ([^ ]+)', first_line)
            if match:
                version = match.group(1)
                print_ok(f"ffmpeg: {version}")
            else:
                print_ok(f"ffmpeg: Found")
                if verbose:
                    print(f"    → {first_line}")
        else:
            print_error("ffmpeg: Failed to get version")
            all_valid = False
    except FileNotFoundError:
        print_error("ffmpeg: NOT FOUND (install with: apt install ffmpeg / brew install ffmpeg)")
        all_valid = False
    except subprocess.TimeoutExpired:
        print_error("ffmpeg: Timeout checking version")
        all_valid = False

    # Check ffprobe (often needed)
    try:
        result = subprocess.run(
            ['ffprobe', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            match = re.search(r'ffprobe version ([^ ]+)', first_line)
            version = match.group(1) if match else "Found"
            print_ok(f"ffprobe: {version}")
        else:
            print_warn("ffprobe: Not available")
            _add_warning("ffprobe not found - some features may not work")
    except FileNotFoundError:
        print_warn("ffprobe: NOT FOUND")
        _add_warning("ffprobe not found - video metadata extraction may fail")

    return all_valid


def validate_cross_section_consistency(config: Any) -> bool:
    """Validate cross-section consistency checks"""
    print_header("CROSS-SECTION CONSISTENCY")
    all_valid: bool = True

    # Check 1: download.retry_budget should have values when download is enabled
    if hasattr(config, 'download') and hasattr(config.download, 'enabled'):
        if config.download.enabled:
            # Check caption_first retry budget
            if hasattr(config.download, 'caption_first'):
                if hasattr(config.download.caption_first, 'retry_budget'):
                    rb = config.download.caption_first.retry_budget
                    if hasattr(rb, 'enabled') and rb.enabled:
                        if not hasattr(rb, 'max_attempts') or rb.max_attempts is None:
                            _add_warning("download.caption_first.retry_budget.enabled=true but max_attempts not set")
                            all_valid = False
                        elif rb.max_attempts <= 0:
                            _add_warning("download.caption_first.retry_budget.max_attempts should be > 0 when enabled")
                            all_valid = False

            # Check main retry budget
            if hasattr(config.download, 'retry_budget'):
                rb = config.download.retry_budget
                if hasattr(rb, 'enabled') and rb.enabled:
                    if not hasattr(rb, 'max_attempts') or rb.max_attempts is None:
                        _add_warning("download.retry_budget.enabled=true but max_attempts not set")
                        all_valid = False

    # Check 2: Mullvad VPN requires download enabled
    if hasattr(config, 'download') and hasattr(config.download, 'mullvad'):
        if hasattr(config.download.mullvad, 'enabled') and config.download.mullvad.enabled:
            if not config.download.enabled:
                _add_warning("download.mullvad.enabled=true requires download.enabled=true")
                all_valid = False

    # Check 3: Region backoff requires Mullvad enabled
    if hasattr(config, 'download') and hasattr(config.download, 'region_backoff'):
        if hasattr(config.download.region_backoff, 'enabled') and config.download.region_backoff.enabled:
            if not hasattr(config.download, 'mullvad') or not getattr(config.download.mullvad, 'enabled', False):
                _add_warning("download.region_backoff.enabled=true requires download.mullvad.enabled=true")
                all_valid = False

    # Check 4: Per-keyword circuit breaker requires download enabled
    if hasattr(config, 'download') and hasattr(config.download, 'per_keyword_circuit_breaker'):
        if hasattr(config.download.per_keyword_circuit_breaker, 'enabled') and config.download.per_keyword_circuit_breaker.enabled:
            if not config.download.enabled:
                _add_warning("download.per_keyword_circuit_breaker.enabled=true requires download.enabled=true")
                all_valid = False

    # Check 5: Rate limit predictor requires rate_limit enabled
    if hasattr(config, 'download') and hasattr(config.download, 'rate_limit_predictor'):
        if hasattr(config.download.rate_limit_predictor, 'enabled') and config.download.rate_limit_predictor.enabled:
            if not hasattr(config.download, 'rate_limit') or not getattr(config.download.rate_limit, 'enabled', True):
                _add_warning("download.rate_limit_predictor.enabled=true requires download.rate_limit.enabled=true")
                all_valid = False

    # Check 6: Cloud backup requires checkpoint enabled
    if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'cloud_backup'):
        if hasattr(config.pipeline.cloud_backup, 'enabled') and config.pipeline.cloud_backup.enabled:
            if not hasattr(config.pipeline, 'checkpoint') or not getattr(config.pipeline.checkpoint, 'enabled', True):
                _add_warning("pipeline.cloud_backup.enabled=true requires pipeline.checkpoint.enabled=true")
                all_valid = False

    # Check 7: Hot backup requires checkpoint enabled
    if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'hot_backup'):
        if hasattr(config.pipeline.hot_backup, 'enabled') and config.pipeline.hot_backup.enabled:
            if not hasattr(config.pipeline, 'checkpoint') or not getattr(config.pipeline.checkpoint, 'enabled', True):
                _add_warning("pipeline.hot_backup.enabled=true requires pipeline.checkpoint.enabled=true")
                all_valid = False

    if all_valid:
        print_ok("All cross-section consistency checks passed")

    return all_valid


def validate_deprecated_options(config: Any) -> bool:
    """Check for deprecated config options"""
    print_header("DEPRECATED OPTIONS")
    deprecated_found: List[Dict[str, Any]] = []

    # Map of deprecated options to their replacements
    deprecated_options: Dict[str, Dict[str, str]] = {
        'download.caption_first': {
            'status': 'deprecated',
            'message': 'Caption-first is always enabled. This field is ignored.',
            'replacement': 'N/A (always on)'
        },
        'google-generativeai': {
            'status': 'deprecated',
            'message': 'google-generativeai is DEPRECATED (support ended Nov 30, 2025)',
            'replacement': 'google-genai'
        },
    }

    # Check for deprecated options in config
    # Note: This is a basic check - a full implementation would traverse the config

    # Check for google-generativeai in requirements vs code
    try:
        import google.generativeai
        deprecated_found.append({
            'option': 'google-generativeai package',
            'status': 'deprecated',
            'message': 'google-generativeai is DEPRECATED (support ended Nov 30, 2025)',
            'replacement': 'google-genai',
            'severity': 'warning'
        })
    except ImportError:
        pass  # Not installed, ok

    # Print results
    if deprecated_found:
        for dep in deprecated_found:
            print_warn(f"{dep['option']}: {dep['message']}")
            if get_verbosity() >= 2:
                print(f"    → Replacement: {dep['replacement']}")
        return False
    else:
        print_ok("No deprecated options detected")
        return True


# JSON output functions for new validations

def validate_package_versions_json() -> Dict[str, Any]:
    """Validate package versions and return JSON results"""
    required_packages: Dict[str, Tuple[str, str]] = {
        'yaml': ('6.0', 'PyYAML'),
        'torch': ('2.0', 'torch'),
        'sentence_transformers': ('2.0', 'sentence-transformers'),
        'faster_whisper': ('1.0.0', 'faster-whisper'),
        'anthropic': ('0.50.0', 'anthropic'),
        'yt_dlp': ('2024.1.0', 'yt-dlp'),
        'scenedetect': ('0.6', 'scenedetect'),
        'opencv_python': ('4.8', 'opencv-python'),
        'librosa': ('0.10.0', 'librosa'),
        'imagehash': ('4.3.0', 'imagehash'),
        'PIL': ('9.0.0', 'Pillow'),
    }

    packages: List[Dict[str, Any]] = []
    all_valid: bool = True

    for module_name, (min_version, display_name) in required_packages.items():
        try:
            if module_name == 'yaml':
                import yaml
                version = yaml.__version__
            elif module_name == 'PIL':
                from PIL import Image
                version = Image.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')

            packages.append({
                'name': display_name,
                'module': module_name,
                'installed': True,
                'version': version,
                'required': min_version,
                'valid': True
            })
        except ImportError:
            packages.append({
                'name': display_name,
                'module': module_name,
                'installed': False,
                'version': None,
                'required': min_version,
                'valid': False
            })
            all_valid = False

    return {
        'packages': packages,
        'valid': all_valid,
        'total': len(packages),
        'installed': sum(1 for p in packages if p['installed'])
    }


def validate_external_tools_json() -> Dict[str, Any]:
    """Validate external tools and return JSON results"""
    tools: List[Dict[str, Any]] = []
    all_valid: bool = True

    # Check yt-dlp
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            tools.append({
                'name': 'yt-dlp',
                'installed': True,
                'version': result.stdout.strip(),
                'required': '2024.1.0',
                'valid': True
            })
        else:
            tools.append({
                'name': 'yt-dlp',
                'installed': False,
                'version': None,
                'required': '2024.1.0',
                'valid': False
            })
            all_valid = False
    except FileNotFoundError:
        tools.append({
            'name': 'yt-dlp',
            'installed': False,
            'version': None,
            'required': '2024.1.0',
            'valid': False
        })
        all_valid = False

    # Check ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            match = re.search(r'ffmpeg version ([^ ]+)', first_line)
            version = match.group(1) if match else "unknown"
            tools.append({
                'name': 'ffmpeg',
                'installed': True,
                'version': version,
                'required': 'any',
                'valid': True
            })
        else:
            tools.append({
                'name': 'ffmpeg',
                'installed': False,
                'version': None,
                'required': 'any',
                'valid': False
            })
            all_valid = False
    except FileNotFoundError:
        tools.append({
            'name': 'ffmpeg',
            'installed': False,
            'version': None,
            'required': 'any',
            'valid': False
        })
        all_valid = False

    # Check ffprobe
    try:
        result = subprocess.run(
            ['ffprobe', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            match = re.search(r'ffprobe version ([^ ]+)', first_line)
            version = match.group(1) if match else "unknown"
            tools.append({
                'name': 'ffprobe',
                'installed': True,
                'version': version,
                'required': 'recommended',
                'valid': True
            })
        else:
            tools.append({
                'name': 'ffprobe',
                'installed': False,
                'version': None,
                'required': 'recommended',
                'valid': True  # Not required, just warning
            })
    except FileNotFoundError:
        tools.append({
            'name': 'ffprobe',
            'installed': False,
            'version': None,
            'required': 'recommended',
            'valid': True  # Not required, just warning
        })

    return {
        'tools': tools,
        'valid': all_valid,
        'total': len(tools),
        'installed': sum(1 for t in tools if t['installed'])
    }


def validate_cross_section_consistency_json(config: Any) -> Dict[str, Any]:
    """Validate cross-section consistency and return JSON results"""
    issues: List[Dict[str, Any]] = []

    # Check 1: download.retry_budget should have values when download is enabled
    if hasattr(config, 'download') and hasattr(config.download, 'enabled'):
        if config.download.enabled:
            if hasattr(config.download, 'caption_first'):
                if hasattr(config.download.caption_first, 'retry_budget'):
                    rb = config.download.caption_first.retry_budget
                    if hasattr(rb, 'enabled') and rb.enabled:
                        if not hasattr(rb, 'max_attempts') or rb.max_attempts is None or rb.max_attempts <= 0:
                            issues.append({
                                'type': 'missing_config',
                                'severity': 'error',
                                'message': 'download.caption_first.retry_budget.enabled=true but max_attempts not set or <= 0'
                            })

            if hasattr(config.download, 'retry_budget'):
                rb = config.download.retry_budget
                if hasattr(rb, 'enabled') and rb.enabled:
                    if not hasattr(rb, 'max_attempts') or rb.max_attempts is None:
                        issues.append({
                            'type': 'missing_config',
                            'severity': 'error',
                            'message': 'download.retry_budget.enabled=true but max_attempts not set'
                        })

    # Check 2: Mullvad VPN requires download enabled
    if hasattr(config, 'download') and hasattr(config.download, 'mullvad'):
        if hasattr(config.download.mullvad, 'enabled') and config.download.mullvad.enabled:
            if not config.download.enabled:
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'download.mullvad.enabled=true requires download.enabled=true'
                })

    # Check 3: Region backoff requires Mullvad enabled
    if hasattr(config, 'download') and hasattr(config.download, 'region_backoff'):
        if hasattr(config.download.region_backoff, 'enabled') and config.download.region_backoff.enabled:
            if not hasattr(config.download, 'mullvad') or not getattr(config.download.mullvad, 'enabled', False):
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'download.region_backoff.enabled=true requires download.mullvad.enabled=true'
                })

    # Check 4: Per-keyword circuit breaker requires download enabled
    if hasattr(config, 'download') and hasattr(config.download, 'per_keyword_circuit_breaker'):
        if hasattr(config.download.per_keyword_circuit_breaker, 'enabled') and config.download.per_keyword_circuit_breaker.enabled:
            if not config.download.enabled:
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'download.per_keyword_circuit_breaker.enabled=true requires download.enabled=true'
                })

    # Check 5: Rate limit predictor requires rate_limit enabled
    if hasattr(config, 'download') and hasattr(config.download, 'rate_limit_predictor'):
        if hasattr(config.download.rate_limit_predictor, 'enabled') and config.download.rate_limit_predictor.enabled:
            if not hasattr(config.download, 'rate_limit') or not getattr(config.download.rate_limit, 'enabled', True):
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'download.rate_limit_predictor.enabled=true requires download.rate_limit.enabled=true'
                })

    # Check 6: Cloud backup requires checkpoint enabled
    if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'cloud_backup'):
        if hasattr(config.pipeline.cloud_backup, 'enabled') and config.pipeline.cloud_backup.enabled:
            if not hasattr(config.pipeline, 'checkpoint') or not getattr(config.pipeline.checkpoint, 'enabled', True):
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'pipeline.cloud_backup.enabled=true requires pipeline.checkpoint.enabled=true'
                })

    # Check 7: Hot backup requires checkpoint enabled
    if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'hot_backup'):
        if hasattr(config.pipeline.hot_backup, 'enabled') and config.pipeline.hot_backup.enabled:
            if not hasattr(config.pipeline, 'checkpoint') or not getattr(config.pipeline.checkpoint, 'enabled', True):
                issues.append({
                    'type': 'dependency',
                    'severity': 'error',
                    'message': 'pipeline.hot_backup.enabled=true requires pipeline.checkpoint.enabled=true'
                })

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'issue_count': len(issues)
    }


def validate_deprecated_options_json(config: Any) -> Dict[str, Any]:
    """Check for deprecated config options and return JSON results"""
    deprecated: List[Dict[str, Any]] = []

    # Check for google-generativeai package
    try:
        import google.generativeai
        deprecated.append({
            'option': 'google-generativeai',
            'status': 'deprecated',
            'message': 'google-generativeai is DEPRECATED (support ended Nov 30, 2025)',
            'replacement': 'google-genai',
            'severity': 'warning'
        })
    except ImportError:
        pass

    # Check for deprecated config options (simple check for caption_first legacy)
    if hasattr(config, 'download') and hasattr(config.download, 'caption_first'):
        # caption_first is always on now, check if explicitly disabled
        cf = config.download.caption_first
        if hasattr(cf, 'enabled') and cf.enabled == False:
            deprecated.append({
                'option': 'download.caption_first.enabled',
                'status': 'ignored',
                'message': 'Caption-first is always enabled, this setting is ignored',
                'replacement': 'remove from config',
                'severity': 'info'
            })

    return {
        'valid': True,  # Deprecated options are warnings, not errors
        'deprecated': deprecated,
        'count': len(deprecated)
    }


def validate_config_structure_json(config: Any) -> Dict[str, Any]:
    """Validate config structure and return JSON results"""
    required_sections: List[Tuple[str, str]] = [
        ('project', 'ProjectConfig'),
        ('transcription', 'TranscriptionConfig'),
        ('embedding', 'EmbeddingConfig'),
        ('indexing', 'IndexingConfig'),
        ('vision', 'VisionConfig'),
        ('scene_detection', 'SceneDetectionConfig'),
        ('matching', 'MatchingConfig'),
        ('keyword', 'KeywordConfig'),
        ('enhanced', 'EnhancedFeaturesConfig'),
        ('downloading', 'DownloadingConfig'),
        ('output', 'OutputConfig'),
        ('logging', 'LoggingConfig'),
        ('cache', 'CacheConfig'),
        ('pipeline', 'PipelineConfig'),
    ]

    sections: List[Dict[str, Any]] = []
    all_valid: bool = True

    for section_name, expected_type in required_sections:
        if hasattr(config, section_name):
            section: Any = getattr(config, section_name)
            actual_type: str = type(section).__name__
            sections.append({
                'name': section_name,
                'type': actual_type,
                'valid': True
            })
        else:
            sections.append({
                'name': section_name,
                'type': None,
                'valid': False
            })
            all_valid = False

    return {
        'sections': sections,
        'valid': all_valid,
        'total': len(sections),
        'present': sum(1 for s in sections if s['valid'])
    }


def validate_api_keys_json(config: Any) -> Dict[str, Any]:
    """Validate API keys and return JSON results"""
    api_keys: Dict[str, Tuple[str, Optional[str]]] = {
        'GEMINI_API_KEY': ('gemini', config.api_keys.gemini_api_key),
        'ANTHROPIC_API_KEY': ('anthropic', config.api_keys.anthropic_api_key),
        'PEXELS_API_KEY': ('pexels', config.api_keys.pexels_api_key),
        'PIXABAY_API_KEY': ('pixabay', config.api_keys.pixabay_api_key),
    }

    keys: List[Dict[str, Any]] = []
    required_checks: Dict[str, bool] = {
        'GEMINI_API_KEY': config.embedding.provider == 'gemini' or config.matching.primary_provider == 'gemini',
        'ANTHROPIC_API_KEY': config.matching.secondary_provider == 'anthropic',
        'PEXELS_API_KEY': config.stock_footage.pexels_enabled,
        'PIXABAY_API_KEY': config.stock_footage.pixabay_enabled,
    }

    for env_name, (provider, key) in api_keys.items():
        is_set: bool = bool(key)
        is_required: bool = required_checks.get(env_name, False)

        keys.append({
            'name': env_name,
            'provider': provider,
            'set': is_set,
            'length': len(key) if key else 0,
            'required': is_required,
            'status': 'set' if is_set else ('required' if is_required else 'optional')
        })

    return {
        'keys': keys,
        'total': len(keys),
        'set': sum(1 for k in keys if k['set']),
        'required_missing': sum(1 for k in keys if not k['set'] and k['required'])
    }


def validate_paths_json(config: Any) -> Dict[str, Any]:
    """Validate paths and return JSON results"""
    paths: List[Tuple[str, str]] = [
        ('project_dir', config.project_dir),
        ('downloaded_videos_dir', config.downloaded_videos_dir),
        ('otio_output_dir', config.otio_output_dir),
        ('cache.cache_dir', config.cache.cache_dir),
        ('logging.log_dir', config.logging.log_dir),
    ]

    path_results: List[Dict[str, Any]] = []
    for name, path in paths:
        p: Path = Path(path)
        exists: bool = p.exists()
        path_results.append({
            'name': name,
            'path': path,
            'exists': exists,
            'status': 'exists' if exists else 'will_be_created'
        })

    return {
        'paths': path_results,
        'total': len(path_results),
        'exist': sum(1 for p in path_results if p['exists'])
    }


def validate_value_ranges_json(config: Any) -> Dict[str, Any]:
    """Validate config values are in valid ranges and return JSON results"""
    checks: List[Tuple[bool, str, str, Any]] = [
        (0 <= config.matching.min_confidence <= 1,
         "matching.min_confidence", "must be 0-1", config.matching.min_confidence),

        (0 <= config.matching.high_confidence_threshold <= 1,
         "matching.high_confidence_threshold", "must be 0-1", config.matching.high_confidence_threshold),

        (config.matching.max_clip_reuse >= 0,
         "matching.max_clip_reuse", "must be >= 0", config.matching.max_clip_reuse),

        (config.transcription.max_workers >= 1,
         "transcription.max_workers", "must be >= 1", config.transcription.max_workers),

        (config.embedding.batch_size >= 1,
         "embedding.batch_size", "must be >= 1", config.embedding.batch_size),

        (config.keyword.max_keywords >= 1,
         "keyword.max_keywords", "must be >= 1", config.keyword.max_keywords),

        (config.enhanced.min_confidence >= 0,
         "enhanced.min_confidence", "must be >= 0", config.enhanced.min_confidence),
    ]

    results: List[Dict[str, Any]] = []
    all_valid: bool = True

    for valid, name, requirement, value in checks:
        results.append({
            'name': name,
            'value': value,
            'valid': valid,
            'requirement': requirement
        })
        if not valid:
            all_valid = False

    return {
        'checks': results,
        'valid': all_valid,
        'total': len(results),
        'passed': sum(1 for r in results if r['valid'])
    }


def validate_nested_access_json(config: Any) -> Dict[str, Any]:
    """Test nested config access and return JSON results"""
    test_paths: List[Tuple[str, Any]] = [
        ("matching.min_confidence", 0.7),
        ("transcription.model", "base"),
        ("embedding.provider", "gemini"),
        ("duration_tiers.short.min_seconds", 20),
    ]

    results: List[Dict[str, Any]] = []
    all_valid: bool = True

    for path, expected_type_or_value in test_paths:
        value: Any = config.get_nested(path)
        valid: bool = value is not None
        results.append({
            'path': path,
            'value': value,
            'valid': valid
        })
        if not valid:
            all_valid = False

    return {
        'accesses': results,
        'valid': all_valid,
        'total': len(results),
        'found': sum(1 for r in results if r['valid'])
    }


def validate_config_validation_json(config: Any) -> Dict[str, Any]:
    """Run config's built-in validation and return JSON results"""
    errors: List[str] = config.validate()

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'error_count': len(errors)
    }


def run_json_validation(strict: bool = False, config_path: str = "config.yaml", skip_validation: bool = False) -> Dict[str, Any]:
    """Run all validations and return JSON results"""
    from src.config import load_config, get_config_metrics

    # Use shared config loader
    config: Any = load_config_for_script(config_path, required=True, skip_validation=skip_validation)
    if config is None:
        # Return error result
        return {
            'valid': False,
            'error': f"Config file not found or invalid: {config_path}",
            'config_file': config_path
        }

    # Get load metrics
    metrics: Dict[str, int] = get_config_metrics()

    # Run all validations
    structure_result = validate_config_structure_json(config)
    api_keys_result = validate_api_keys_json(config)
    paths_result = validate_paths_json(config)
    value_ranges_result = validate_value_ranges_json(config)
    nested_access_result = validate_nested_access_json(config)
    validation_result = validate_config_validation_json(config)

    # New validation functions (US-131-009)
    package_versions_result = validate_package_versions_json()
    external_tools_result = validate_external_tools_json()
    cross_section_result = validate_cross_section_consistency_json(config)
    deprecated_result = validate_deprecated_options_json(config)

    # Build overall result
    all_valid = (
        structure_result['valid'] and
        api_keys_result['required_missing'] == 0 and
        value_ranges_result['valid'] and
        nested_access_result['valid'] and
        validation_result['valid'] and
        package_versions_result['valid'] and
        external_tools_result['valid'] and
        cross_section_result['valid']
    )

    # In strict mode, deprecated options also fail
    if strict:
        all_valid = all_valid and deprecated_result['count'] == 0

    return {
        'valid': all_valid,
        'strict': strict,
        'config_file': config._config_path,
        'config_hash': config._config_hash,
        'project': {
            'name': config.project.name,
            'version': config.project.version
        },
        'load_metrics': {
            'load_count': metrics['load_count'],
            'cache_hits': metrics.get('cache_hits', 0)
        },
        'sections': structure_result,
        'api_keys': api_keys_result,
        'paths': paths_result,
        'value_ranges': value_ranges_result,
        'nested_access': nested_access_result,
        'validation': validation_result,
        'package_versions': package_versions_result,
        'external_tools': external_tools_result,
        'cross_section_consistency': cross_section_result,
        'deprecated_options': deprecated_result
    }


def main() -> None:
    """Main validation entry point"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Validate config.yaml integration")
    parser.add_argument('--quick', action='store_true', help='Quick validation only')
    parser.add_argument('--perf', action='store_true', help='Performance benchmark only')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--report-json', action='store_true', help='Output full validation report as JSON (alias for --json)')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors (for --json output)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable detailed output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress non-essential output')
    # Add config path argument using shared helper
    add_config_argument(parser, default="config.yaml", help_text="Path to config file to validate")
    args: argparse.Namespace = parser.parse_args()

    # --report-json is alias for --json
    if args.report_json:
        args.json = True

    # Set verbosity level
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Handle JSON output
    if args.json:
        result: Dict[str, Any] = run_json_validation(strict=args.strict, config_path=args.config, skip_validation=True)
        print(json.dumps(result, indent=2))
        # Validate JSON is parseable
        try:
            json.loads(json.dumps(result))
        except json.JSONDecodeError as e:
            print(f"  [ERROR] JSON output is invalid: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if result['valid'] else 1)

    # Main header - show in normal/verbose mode only
    if get_verbosity() >= 1:
        print("\n" + "=" * 60)
        print("  CONFIG.YAML INTEGRATION VALIDATION")
        print("=" * 60)

    if args.perf:
        performance_benchmark()
        return

    # Load config (skip validation to allow testing without API keys)
    config: Any = validate_config_loading(args.config, skip_validation=True)

    if args.quick:
        # Quick validation
        validate_config_structure(config)
        validate_config_validation(config)
        return

    # Full validation
    all_valid: bool = True

    all_valid &= validate_config_structure(config)
    validate_api_keys(config)
    validate_paths(config)
    all_valid &= validate_value_ranges(config)
    all_valid &= validate_nested_access(config)
    all_valid &= validate_config_validation(config)

    # NEW: Extended validation (US-131-009)
    all_valid &= validate_package_versions()
    all_valid &= validate_external_tools()
    all_valid &= validate_cross_section_consistency(config)
    validate_deprecated_options(config)

    # Advanced tests
    test_hot_reload()
    performance_benchmark()

    # Summary
    print_header("SUMMARY")

    # Check for warnings
    warnings = _get_warnings()
    if warnings:
        print_warn(f"{len(warnings)} warning(s) detected:")
        for w in warnings:
            print(f"  - {w}")

    if all_valid:
        print_ok("All validation checks passed!")
        print(f"\n  Config file: {config._config_path}")
        print(f"  Config hash: {config._config_hash}")
        print(f"  Project: {config.project.name} v{config.project.version}")
    else:
        print_error("Some validation checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
