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
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str):
    """Print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_ok(msg: str):
    """Print success message"""
    print(f"  ✓ {msg}")


def print_warn(msg: str):
    """Print warning message"""
    print(f"  ⚠ {msg}")


def print_error(msg: str):
    """Print error message"""
    print(f"  ✗ {msg}")


def validate_config_loading():
    """Test config loading performance and correctness"""
    print_header("CONFIG LOADING")
    
    from src.config import load_config, get_config, get_config_metrics
    
    # Test loading performance
    start = time.perf_counter()
    config = load_config("config.yaml")
    load_time = (time.perf_counter() - start) * 1000
    
    if load_time < 100:
        print_ok(f"Config loaded in {load_time:.1f}ms (target: <100ms)")
    else:
        print_warn(f"Config loaded in {load_time:.1f}ms (target: <100ms)")
    
    # Check metrics
    metrics = get_config_metrics()
    print_ok(f"Load count: {metrics['load_count']}")
    
    # Check CSafeLoader
    try:
        from yaml import CSafeLoader
        print_ok("Using CSafeLoader (C-based, optimized)")
    except ImportError:
        print_warn("CSafeLoader not available (using pure Python)")
    
    return config


def validate_config_structure(config):
    """Validate config structure has all required sections"""
    print_header("CONFIG STRUCTURE")
    
    required_sections = [
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
    
    all_valid = True
    for section_name, expected_type in required_sections:
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            actual_type = type(section).__name__
            print_ok(f"{section_name}: {actual_type}")
        else:
            print_error(f"{section_name}: MISSING")
            all_valid = False
    
    return all_valid


def validate_api_keys(config):
    """Validate API keys are available"""
    print_header("API KEYS")
    
    api_keys = {
        'GEMINI_API_KEY': ('gemini', config.api_keys.gemini_api_key),
        'ANTHROPIC_API_KEY': ('anthropic', config.api_keys.anthropic_api_key),
        'PEXELS_API_KEY': ('pexels', config.api_keys.pexels_api_key),
        'PIXABAY_API_KEY': ('pixabay', config.api_keys.pixabay_api_key),
    }
    
    for env_name, (provider, key) in api_keys.items():
        if key:
            print_ok(f"{env_name}: Set ({len(key)} chars)")
        else:
            # Check if required
            required_checks = {
                'GEMINI_API_KEY': config.embedding.provider == 'gemini' or config.matching.primary_provider == 'gemini',
                'ANTHROPIC_API_KEY': config.matching.secondary_provider == 'anthropic',
                'PEXELS_API_KEY': config.stock_footage.pexels_enabled,
                'PIXABAY_API_KEY': config.stock_footage.pixabay_enabled,
            }
            
            if required_checks.get(env_name, False):
                print_warn(f"{env_name}: Not set (required for {provider})")
            else:
                print_ok(f"{env_name}: Not set (optional)")


def validate_paths(config):
    """Validate configured paths"""
    print_header("PATHS")
    
    paths = [
        ('project_dir', config.project_dir),
        ('downloaded_videos_dir', config.downloaded_videos_dir),
        ('otio_output_dir', config.otio_output_dir),
        ('cache.cache_dir', config.cache.cache_dir),
        ('logging.log_dir', config.logging.log_dir),
    ]
    
    for name, path in paths:
        p = Path(path)
        if p.exists():
            print_ok(f"{name}: {path} (exists)")
        else:
            print_warn(f"{name}: {path} (will be created)")


def validate_value_ranges(config):
    """Validate config values are in valid ranges"""
    print_header("VALUE RANGES")
    
    checks = [
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
    
    all_valid = True
    for valid, value_str, requirement in checks:
        if valid:
            print_ok(value_str)
        else:
            print_error(f"{value_str} - {requirement}")
            all_valid = False
    
    return all_valid


def validate_nested_access(config):
    """Test nested config access"""
    print_header("NESTED ACCESS")
    
    test_paths = [
        ("matching.min_confidence", 0.7),
        ("transcription.model", "base"),
        ("embedding.provider", "gemini"),
        ("duration_tiers.short.min_seconds", 20),
    ]
    
    all_valid = True
    for path, expected_type_or_value in test_paths:
        value = config.get_nested(path)
        if value is not None:
            print_ok(f"{path} = {value}")
        else:
            print_error(f"{path} = NOT FOUND")
            all_valid = False
    
    return all_valid


def validate_config_validation(config):
    """Run config's built-in validation"""
    print_header("BUILT-IN VALIDATION")
    
    errors = config.validate()
    
    if errors:
        print_warn(f"Validation warnings ({len(errors)}):")
        for error in errors:
            print(f"    • {error}")
        return False
    else:
        print_ok("All validation checks passed")
        return True


def performance_benchmark():
    """Benchmark config loading performance"""
    print_header("PERFORMANCE BENCHMARK")
    
    from src.config import Config
    
    # Warm up
    for _ in range(3):
        Config.from_yaml("config.yaml")
    
    # Benchmark
    iterations = 20
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        Config.from_yaml("config.yaml")
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
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
    
    cache_times = []
    for _ in range(1000):
        start = time.perf_counter()
        get_config()
        cache_times.append((time.perf_counter() - start) * 1000)
    
    avg_cache_time = sum(cache_times) / len(cache_times)
    
    print_ok(f"Cached access time: {avg_cache_time * 1000:.3f}μs")
    
    metrics = get_config_metrics()
    print_ok(f"Cache hits: {metrics['cache_hits']}")


def test_hot_reload():
    """Test hot reload capability"""
    print_header("HOT RELOAD")
    
    from src.config import load_config, reload_config
    import tempfile
    import shutil
    
    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("project:\n  name: test-project\n  version: 1.0.0\n")
        temp_path = f.name
    
    try:
        # Load initial config
        config = load_config(temp_path)
        initial_hash = config._config_hash
        print_ok(f"Initial load: hash={initial_hash}")
        
        # Modify file
        time.sleep(0.1)
        with open(temp_path, 'w') as f:
            f.write("project:\n  name: modified-project\n  version: 2.0.0\n")
        
        # Reload
        reloaded = config.reload()
        
        if reloaded:
            print_ok(f"Hot reload detected change: hash={config._config_hash}")
        else:
            print_warn("Hot reload did not detect change")
        
        # Verify no change when file unchanged
        reloaded_again = config.reload()
        if not reloaded_again:
            print_ok("No reload when file unchanged")
        else:
            print_warn("Unnecessary reload triggered")
        
    finally:
        os.unlink(temp_path)


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate config.yaml integration")
    parser.add_argument('--quick', action='store_true', help='Quick validation only')
    parser.add_argument('--perf', action='store_true', help='Performance benchmark only')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  CONFIG.YAML INTEGRATION VALIDATION")
    print("=" * 60)
    
    if args.perf:
        performance_benchmark()
        return
    
    # Load config
    config = validate_config_loading()
    
    if args.quick:
        # Quick validation
        validate_config_structure(config)
        validate_config_validation(config)
        return
    
    # Full validation
    all_valid = True
    
    all_valid &= validate_config_structure(config)
    validate_api_keys(config)
    validate_paths(config)
    all_valid &= validate_value_ranges(config)
    all_valid &= validate_nested_access(config)
    all_valid &= validate_config_validation(config)
    
    # Advanced tests
    test_hot_reload()
    performance_benchmark()
    
    # Summary
    print_header("SUMMARY")
    
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
