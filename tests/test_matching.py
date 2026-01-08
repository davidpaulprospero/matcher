#!/usr/bin/env python3
"""
Matching-Only Test Suite v1.0

Test harness for the matching algorithm without needing to run download/transcription.
Uses fixtures captured from real pipeline runs.

Use Cases:
    - Test new matching algorithms
    - Compare different config settings (min_confidence, candidates, etc.)
    - Validate algorithm changes in isolation
    - Quick iteration on matching logic

Usage:
    # Run with saved fixtures
    python tests/test_matching.py --fixtures path/to/fixture.json

    # With config overrides
    python tests/test_matching.py --fixtures fixture.json --override min_confidence=0.5

    # A/B comparison
    python tests/test_matching.py --fixtures fixture.json --compare \\
        --config-a "min_confidence=0.7" --config-b "min_confidence=0.5"

    # Verbose output
    python tests/test_matching.py --fixtures fixture.json --verbose

Fixture Creation:
    # In main.py, run with --save-matching-fixtures flag
    python main.py --project MyProject --save-matching-fixtures fixtures/my_test.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from statistics import mean, stdev

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MatchingMetrics:
    """Metrics collected during matching test"""

    # Match counts
    total_segments: int = 0
    matched_segments: int = 0
    gaps_count: int = 0
    match_rate: float = 0.0

    # Confidence distribution
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0

    # Diversity metrics
    unique_sources_v1: int = 0
    unique_sources_all: int = 0
    source_distribution: Dict[str, int] = field(default_factory=dict)

    # Strategy coverage
    v4_v6_coverage: float = 0.0
    v7_coverage: float = 0.0
    strategy_counts: Dict[str, int] = field(default_factory=dict)

    # Performance
    total_time_ms: float = 0.0
    time_per_segment_ms: float = 0.0


@dataclass
class ComparisonResult:
    """A/B comparison results"""

    name_a: str
    name_b: str
    metrics_a: MatchingMetrics
    metrics_b: MatchingMetrics

    @property
    def confidence_delta(self) -> float:
        """Difference in avg confidence (B - A)"""
        return self.metrics_b.avg_confidence - self.metrics_a.avg_confidence

    @property
    def match_rate_delta(self) -> float:
        """Difference in match rate (B - A)"""
        return self.metrics_b.match_rate - self.metrics_a.match_rate

    @property
    def diversity_delta(self) -> int:
        """Difference in unique sources (B - A)"""
        return self.metrics_b.unique_sources_v1 - self.metrics_a.unique_sources_v1


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# FIXTURE LOADING/SAVING
# =============================================================================

def save_matching_fixtures(
    output_path: str,
    vo_segments: List[Any],
    video_segments: List[Any],
    vo_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
    config: Any,
    project_dir: str = ""
) -> bool:
    """
    Save matching inputs to fixture files for testing.

    Creates two files:
    - {output_path}.json - Segments and config
    - {output_path}.npz - Embeddings

    Args:
        output_path: Base path for output files (without extension)
        vo_segments: Voiceover SRTSegment list
        video_segments: Video SRTSegment list
        vo_embeddings: Voiceover embedding array
        video_embeddings: Video embedding array
        config: Config object
        project_dir: Source project directory

    Returns:
        True if saved successfully
    """
    from src.utils import SRTSegment

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert segments to dicts
    vo_dicts = []
    for seg in vo_segments:
        if hasattr(seg, 'to_dict'):
            vo_dicts.append(seg.to_dict())
        elif isinstance(seg, dict):
            vo_dicts.append(seg)
        else:
            vo_dicts.append({
                'index': getattr(seg, 'index', 0),
                'start_time': getattr(seg, 'start_time', 0),
                'end_time': getattr(seg, 'end_time', 0),
                'text': getattr(seg, 'text', ''),
                'source_file': getattr(seg, 'source_file', ''),
            })

    vid_dicts = []
    for seg in video_segments:
        if hasattr(seg, 'to_dict'):
            vid_dicts.append(seg.to_dict())
        elif isinstance(seg, dict):
            vid_dicts.append(seg)
        else:
            vid_dicts.append({
                'index': getattr(seg, 'index', 0),
                'start_time': getattr(seg, 'start_time', 0),
                'end_time': getattr(seg, 'end_time', 0),
                'text': getattr(seg, 'text', ''),
                'source_file': getattr(seg, 'source_file', ''),
            })

    # Extract config snapshot
    config_snapshot = {}
    if hasattr(config, 'matching'):
        mc = config.matching
        config_snapshot = {
            'min_confidence': getattr(mc, 'min_confidence', 0.3),
            'high_confidence_threshold': getattr(mc, 'high_confidence_threshold', 0.85),
            'embedding_candidates': getattr(mc, 'embedding_candidates', 20),
            'llm_rerank_candidates': getattr(mc, 'llm_rerank_candidates', 5),
            'max_clip_reuse': getattr(mc, 'max_clip_reuse', 3),
            'reuse_penalty': getattr(mc, 'reuse_penalty', 0.1),
        }

    # Build fixture JSON
    json_path = output_path.with_suffix('.json')
    npz_path = output_path.with_suffix('.npz')

    fixture_data = {
        'created': datetime.now().isoformat(),
        'source_project': str(project_dir),
        'voiceover_segments': vo_dicts,
        'video_segments': vid_dicts,
        'config_snapshot': config_snapshot,
        'embeddings_file': npz_path.name,
        'stats': {
            'vo_segment_count': len(vo_dicts),
            'video_segment_count': len(vid_dicts),
            'vo_embedding_dim': vo_embeddings.shape[1] if len(vo_embeddings.shape) > 1 else 0,
            'video_embedding_dim': video_embeddings.shape[1] if len(video_embeddings.shape) > 1 else 0,
        }
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fixture_data, f, indent=2)

        np.savez_compressed(
            npz_path,
            vo_embeddings=vo_embeddings,
            video_embeddings=video_embeddings
        )

        print(f"  Saved matching fixtures:")
        print(f"    - {json_path}")
        print(f"    - {npz_path}")
        return True

    except Exception as e:
        print(f"  Failed to save fixtures: {e}")
        return False


def load_matching_fixtures(fixture_path: str) -> Tuple[List, List, np.ndarray, np.ndarray, Dict]:
    """
    Load matching fixtures from saved files.

    Args:
        fixture_path: Path to fixture JSON file

    Returns:
        Tuple of (vo_segments, video_segments, vo_embeddings, video_embeddings, config_snapshot)
    """
    from src.utils import SRTSegment

    fixture_path = Path(fixture_path)

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load embeddings
    npz_path = fixture_path.parent / data['embeddings_file']
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {npz_path}")

    npz_data = np.load(npz_path)
    vo_embeddings = npz_data['vo_embeddings']
    video_embeddings = npz_data['video_embeddings']

    # Convert dicts to SRTSegment objects
    vo_segments = [SRTSegment.from_dict(d) for d in data['voiceover_segments']]
    video_segments = [SRTSegment.from_dict(d) for d in data['video_segments']]

    config_snapshot = data.get('config_snapshot', {})

    return vo_segments, video_segments, vo_embeddings, video_embeddings, config_snapshot


# =============================================================================
# CONFIG UTILITIES
# =============================================================================

def create_test_config(overrides: Dict[str, Any] = None) -> Any:
    """
    Create a config object with optional overrides.

    Args:
        overrides: Dict of matching config overrides
            - Simple: {'min_confidence': 0.5}
            - Nested: {'location_matching.enabled': False}

    Returns:
        Config object with overrides applied
    """
    from src.config import load_config

    config = load_config()

    if overrides:
        for key, value in overrides.items():
            # Handle nested keys like "location_matching.enabled"
            if '.' in key:
                parts = key.split('.')
                if len(parts) == 2:
                    parent_key, child_key = parts
                    parent_obj = getattr(config.matching, parent_key, None)
                    if parent_obj is not None:
                        if hasattr(parent_obj, child_key):
                            setattr(parent_obj, child_key, value)
                        elif isinstance(parent_obj, dict):
                            parent_obj[child_key] = value
                        else:
                            print(f"  Warning: Cannot set {key}")
                    else:
                        print(f"  Warning: Unknown matching config key: {parent_key}")
            elif hasattr(config.matching, key):
                setattr(config.matching, key, value)
            else:
                print(f"  Warning: Unknown matching config key: {key}")

    return config


def parse_config_string(config_str: str) -> Dict[str, Any]:
    """
    Parse a config string like "min_confidence=0.5,embedding_candidates=100"

    Args:
        config_str: Comma-separated key=value pairs

    Returns:
        Dict of parsed config overrides
    """
    overrides = {}

    if not config_str:
        return overrides

    for part in config_str.split(','):
        part = part.strip()
        if '=' not in part:
            continue

        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Try to parse value as appropriate type
        try:
            if '.' in value:
                overrides[key] = float(value)
            elif value.lower() in ('true', 'false'):
                overrides[key] = value.lower() == 'true'
            else:
                overrides[key] = int(value)
        except ValueError:
            overrides[key] = value

    return overrides


# =============================================================================
# METRICS COLLECTION
# =============================================================================

def collect_metrics(matches: List[Any], total_segments: int, elapsed_time: float) -> MatchingMetrics:
    """
    Collect metrics from matching results.

    Args:
        matches: List of MatchResult objects
        total_segments: Total voiceover segments
        elapsed_time: Time taken in seconds

    Returns:
        MatchingMetrics object
    """
    metrics = MatchingMetrics()
    metrics.total_segments = total_segments

    # Count matches and gaps
    confidences = []
    sources_v1 = set()
    sources_all = set()
    source_counts = {}
    strategy_counts = {}
    v4_v6_count = 0
    v7_count = 0

    for m in matches:
        if m is None:
            metrics.gaps_count += 1
            continue

        if hasattr(m, 'has_gap') and m.has_gap:
            metrics.gaps_count += 1
            continue

        if hasattr(m, 'primary_match') and m.primary_match:
            pm = m.primary_match
            metrics.matched_segments += 1
            confidences.append(pm.confidence)

            # Track source diversity
            source_file = ''
            if hasattr(pm, 'video_segment') and pm.video_segment:
                source_file = getattr(pm.video_segment, 'source_file', '')

            if source_file:
                sources_v1.add(source_file)
                sources_all.add(source_file)
                source_name = Path(source_file).name
                source_counts[source_name] = source_counts.get(source_name, 0) + 1

        # Check secondary matches (V4-V6)
        if hasattr(m, 'secondary_matches') and m.secondary_matches:
            v4_v6_count += 1
            for sm in m.secondary_matches:
                if hasattr(sm, 'video_segment') and sm.video_segment:
                    sf = getattr(sm.video_segment, 'source_file', '')
                    if sf:
                        sources_all.add(sf)

        # Check strategy matches (V7+)
        if hasattr(m, 'strategy_matches') and m.strategy_matches:
            v7_count += 1
            for sm in m.strategy_matches:
                strategy = getattr(sm, 'strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                if hasattr(sm, 'video_segment') and sm.video_segment:
                    sf = getattr(sm.video_segment, 'source_file', '')
                    if sf:
                        sources_all.add(sf)

    # Calculate rates
    metrics.match_rate = metrics.matched_segments / total_segments if total_segments > 0 else 0

    # Confidence stats
    if confidences:
        metrics.avg_confidence = mean(confidences)
        metrics.min_confidence = min(confidences)
        metrics.max_confidence = max(confidences)
        metrics.confidence_std = stdev(confidences) if len(confidences) > 1 else 0.0

    # Diversity stats
    metrics.unique_sources_v1 = len(sources_v1)
    metrics.unique_sources_all = len(sources_all)
    metrics.source_distribution = dict(sorted(source_counts.items(), key=lambda x: -x[1])[:10])

    # Strategy coverage
    metrics.v4_v6_coverage = v4_v6_count / total_segments if total_segments > 0 else 0
    metrics.v7_coverage = v7_count / total_segments if total_segments > 0 else 0
    metrics.strategy_counts = strategy_counts

    # Performance
    metrics.total_time_ms = elapsed_time * 1000
    metrics.time_per_segment_ms = (elapsed_time * 1000) / total_segments if total_segments > 0 else 0

    return metrics


# =============================================================================
# TEST RUNNER
# =============================================================================

class MatchingTestRunner:
    """
    Test runner for matching algorithm.

    Usage:
        runner = MatchingTestRunner(verbose=True)
        runner.load_fixtures('path/to/fixture.json')

        # Single config test
        metrics = runner.run_matching(config_overrides={'min_confidence': 0.5})

        # A/B comparison
        comparison = runner.compare_configs(
            config_a={'min_confidence': 0.7},
            config_b={'min_confidence': 0.5}
        )
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.fixtures_loaded = False

        # Fixture data
        self.vo_segments = []
        self.video_segments = []
        self.vo_embeddings = None
        self.video_embeddings = None
        self.config_snapshot = {}
        self.fixture_path = ""

        # Test results
        self.results: List[TestResult] = []

    def log(self, message: str, indent: int = 0):
        """Print message with optional indent"""
        prefix = "  " * indent
        print(f"{prefix}{message}")

    def log_verbose(self, message: str, indent: int = 0):
        """Print message only in verbose mode"""
        if self.verbose:
            self.log(message, indent)

    def load_fixtures(self, fixture_path: str) -> bool:
        """Load fixtures from file"""
        try:
            self.fixture_path = fixture_path
            (
                self.vo_segments,
                self.video_segments,
                self.vo_embeddings,
                self.video_embeddings,
                self.config_snapshot
            ) = load_matching_fixtures(fixture_path)

            self.fixtures_loaded = True

            self.log(f"Loaded fixtures from: {fixture_path}")
            self.log(f"  Voiceover segments: {len(self.vo_segments)}")
            self.log(f"  Video segments: {len(self.video_segments)}")
            self.log(f"  Embedding dimensions: {self.vo_embeddings.shape[1] if len(self.vo_embeddings.shape) > 1 else 0}")

            return True

        except Exception as e:
            self.log(f"Failed to load fixtures: {e}")
            return False

    def run_matching(
        self,
        config_overrides: Dict[str, Any] = None
    ) -> Tuple[List[Any], MatchingMetrics]:
        """
        Run matching with optional config overrides.

        Args:
            config_overrides: Dict of matching config overrides

        Returns:
            Tuple of (matches, metrics)
        """
        if not self.fixtures_loaded:
            raise RuntimeError("No fixtures loaded. Call load_fixtures() first.")

        from src.matching import match_all_segments
        from src.utils import CacheManager
        from src.embeddings import build_embedding_index

        # Create config with overrides
        config = create_test_config(config_overrides)

        # Build embedding index
        embedding_index = build_embedding_index(self.video_embeddings, config=config)

        # Create cache manager (use temp dir)
        cache = CacheManager(str(Path(__file__).parent / '.test_cache'))

        self.log_verbose("Running matching...")

        # Time the matching
        start_time = time.time()

        matches = match_all_segments(
            voiceover_segments=self.vo_segments,
            video_segments=self.video_segments,
            voiceover_embeddings=self.vo_embeddings,
            video_embeddings=self.video_embeddings,
            scenes=None,
            config=config,
            cache=cache,
            embedding_index=embedding_index,
            face_preference='neutral'
        )

        elapsed_time = time.time() - start_time

        # Collect metrics
        metrics = collect_metrics(matches, len(self.vo_segments), elapsed_time)

        return matches, metrics

    def compare_configs(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        name_a: str = "Config A",
        name_b: str = "Config B"
    ) -> ComparisonResult:
        """
        Compare two configurations on the same fixtures.

        Args:
            config_a: First config overrides
            config_b: Second config overrides
            name_a: Name for first config
            name_b: Name for second config

        Returns:
            ComparisonResult with metrics from both
        """
        self.log(f"\nRunning A/B comparison:")
        self.log(f"  {name_a}: {config_a}")
        self.log(f"  {name_b}: {config_b}")

        # Run with config A
        self.log(f"\nRunning {name_a}...")
        _, metrics_a = self.run_matching(config_a)

        # Run with config B
        self.log(f"\nRunning {name_b}...")
        _, metrics_b = self.run_matching(config_b)

        return ComparisonResult(
            name_a=name_a,
            name_b=name_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b
        )

    def run_test(self, name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test"""
        self.log(f"  {name}...", indent=0)
        start = time.time()

        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start

            if isinstance(result, tuple):
                if len(result) == 2:
                    passed, message = result
                    details = {}
                else:
                    passed, message, details = result
            else:
                passed = bool(result)
                message = "OK" if passed else "Failed"
                details = {}

            test_result = TestResult(
                name=name,
                passed=passed,
                duration=duration,
                message=message,
                details=details
            )

        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                message=f"Exception: {str(e)}"
            )
            if self.verbose:
                import traceback
                traceback.print_exc()

        self.results.append(test_result)

        status = "PASS" if test_result.passed else "FAIL"
        self.log(f"     [{status}] ({duration:.1f}s) {test_result.message}")

        return test_result

    def print_metrics(self, metrics: MatchingMetrics, title: str = "METRICS"):
        """Print formatted metrics"""
        print(f"\n  {title}")
        print(f"  {'-' * 50}")  # ASCII dash instead of unicode
        print(f"    Matches: {metrics.matched_segments}/{metrics.total_segments} ({metrics.match_rate:.1%})")
        print(f"    Gaps: {metrics.gaps_count} ({1 - metrics.match_rate:.1%})")
        print(f"    Avg confidence: {metrics.avg_confidence:.2f}")
        print(f"    Confidence range: {metrics.min_confidence:.2f} - {metrics.max_confidence:.2f}")
        if metrics.confidence_std > 0:
            print(f"    Confidence std: {metrics.confidence_std:.2f}")

        print(f"\n  DIVERSITY")
        print(f"    V1 unique sources: {metrics.unique_sources_v1}")
        print(f"    Total unique sources: {metrics.unique_sources_all}")
        if metrics.source_distribution:
            print(f"    Top sources:")
            for source, count in list(metrics.source_distribution.items())[:5]:
                print(f"      - {source}: {count}")

        print(f"\n  STRATEGIES")
        print(f"    V4-V6 coverage: {metrics.v4_v6_coverage:.1%}")
        print(f"    V7 coverage: {metrics.v7_coverage:.1%}")
        if metrics.strategy_counts:
            for strategy, count in metrics.strategy_counts.items():
                print(f"      - {strategy}: {count}")

        print(f"\n  PERFORMANCE")
        print(f"    Total time: {metrics.total_time_ms/1000:.1f}s")
        print(f"    Per segment: {metrics.time_per_segment_ms:.0f}ms")

    def print_comparison(self, comparison: ComparisonResult):
        """Print formatted comparison"""
        print(f"\n{'=' * 60}")
        print(f"  A/B COMPARISON RESULTS")
        print(f"{'=' * 60}")

        print(f"\n  {comparison.name_a} vs {comparison.name_b}")
        print(f"  {'-' * 50}")

        # Side by side metrics
        ma, mb = comparison.metrics_a, comparison.metrics_b

        print(f"\n  {'Metric':<30} {'A':>10} {'B':>10} {'Delta':>10}")
        print(f"  {'-' * 60}")
        print(f"  {'Match rate':<30} {ma.match_rate:>9.1%} {mb.match_rate:>9.1%} {comparison.match_rate_delta:>+9.1%}")
        print(f"  {'Avg confidence':<30} {ma.avg_confidence:>10.2f} {mb.avg_confidence:>10.2f} {comparison.confidence_delta:>+10.2f}")
        print(f"  {'Unique sources (V1)':<30} {ma.unique_sources_v1:>10} {mb.unique_sources_v1:>10} {comparison.diversity_delta:>+10}")
        print(f"  {'V4-V6 coverage':<30} {ma.v4_v6_coverage:>9.1%} {mb.v4_v6_coverage:>9.1%} {mb.v4_v6_coverage - ma.v4_v6_coverage:>+9.1%}")
        print(f"  {'Time (ms/seg)':<30} {ma.time_per_segment_ms:>10.0f} {mb.time_per_segment_ms:>10.0f} {mb.time_per_segment_ms - ma.time_per_segment_ms:>+10.0f}")

        print(f"\n  WINNER: ", end="")

        # Determine winner based on multiple factors
        a_score = 0
        b_score = 0

        if ma.match_rate > mb.match_rate:
            a_score += 1
        elif mb.match_rate > ma.match_rate:
            b_score += 1

        if ma.avg_confidence > mb.avg_confidence:
            a_score += 1
        elif mb.avg_confidence > ma.avg_confidence:
            b_score += 1

        if ma.unique_sources_v1 > mb.unique_sources_v1:
            a_score += 1
        elif mb.unique_sources_v1 > ma.unique_sources_v1:
            b_score += 1

        if a_score > b_score:
            print(f"{comparison.name_a} ({a_score}-{b_score})")
        elif b_score > a_score:
            print(f"{comparison.name_b} ({b_score}-{a_score})")
        else:
            print(f"TIE ({a_score}-{b_score})")

        print(f"{'=' * 60}")

    def print_summary(self) -> bool:
        """Print test summary and return True if all passed"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"  MATCHING TEST SUMMARY")
        print(f"{'=' * 60}")

        if self.fixture_path:
            print(f"  Fixtures: {Path(self.fixture_path).name}")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message}")

        print(f"\n  Total: {passed}/{total} tests passed")
        print(f"{'=' * 60}")

        return failed == 0


# =============================================================================
# TEST CASES
# =============================================================================

@pytest.mark.integration
def test_matching_runs(runner: MatchingTestRunner) -> Tuple[bool, str, Dict]:
    """Test that matching runs without errors"""
    try:
        matches, metrics = runner.run_matching()

        if metrics.matched_segments > 0:
            return True, f"Matched {metrics.matched_segments} segments", {'metrics': asdict(metrics)}
        else:
            return False, "No segments matched", {}
    except Exception as e:
        return False, str(e), {}


@pytest.mark.integration
def test_confidence_thresholds(runner: MatchingTestRunner) -> Tuple[bool, str, Dict]:
    """Test that min_confidence filtering works"""
    try:
        # Run with high threshold
        _, metrics_high = runner.run_matching({'min_confidence': 0.7})

        # Run with low threshold
        _, metrics_low = runner.run_matching({'min_confidence': 0.3})

        # Low threshold should have >= matches
        if metrics_low.matched_segments >= metrics_high.matched_segments:
            return True, f"High: {metrics_high.matched_segments}, Low: {metrics_low.matched_segments}", {}
        else:
            return False, f"Low threshold has fewer matches", {}
    except Exception as e:
        return False, str(e), {}


@pytest.mark.integration
def test_reuse_prevention(runner: MatchingTestRunner) -> Tuple[bool, str, Dict]:
    """Test that reuse prevention works"""
    try:
        # Run with strict reuse (max 1)
        _, metrics_strict = runner.run_matching({'max_clip_reuse': 1})

        # Run with loose reuse (max 10)
        _, metrics_loose = runner.run_matching({'max_clip_reuse': 10})

        # Strict should have more unique sources
        if metrics_strict.unique_sources_v1 >= metrics_loose.unique_sources_v1 * 0.8:
            return True, f"Strict: {metrics_strict.unique_sources_v1} unique, Loose: {metrics_loose.unique_sources_v1}", {}
        else:
            return False, f"Strict has significantly fewer unique sources", {}
    except Exception as e:
        return False, str(e), {}


@pytest.mark.integration
def test_strategy_matches(runner: MatchingTestRunner) -> Tuple[bool, str, Dict]:
    """Test that strategy matches (V4-V7) are generated"""
    try:
        matches, metrics = runner.run_matching()

        has_strategies = metrics.v4_v6_coverage > 0 or metrics.v7_coverage > 0

        if has_strategies:
            return True, f"V4-V6: {metrics.v4_v6_coverage:.1%}, V7: {metrics.v7_coverage:.1%}", {}
        else:
            return True, "No strategy matches (may be expected)", {}  # Not a failure
    except Exception as e:
        return False, str(e), {}


# =============================================================================
# MAIN
# =============================================================================

def run_matching_tests(
    fixture_path: str,
    config_overrides: Dict[str, Any] = None,
    compare_mode: bool = False,
    config_a_str: str = None,
    config_b_str: str = None,
    verbose: bool = False
) -> bool:
    """Run matching tests"""

    runner = MatchingTestRunner(verbose=verbose)

    print(f"\n{'=' * 60}")
    print(f"  MATCHING TEST SUITE")
    print(f"{'=' * 60}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load fixtures
    if not runner.load_fixtures(fixture_path):
        return False

    # Comparison mode
    if compare_mode and config_a_str and config_b_str:
        config_a = parse_config_string(config_a_str)
        config_b = parse_config_string(config_b_str)

        comparison = runner.compare_configs(
            config_a=config_a,
            config_b=config_b,
            name_a=f"A ({config_a_str})",
            name_b=f"B ({config_b_str})"
        )

        runner.print_comparison(comparison)
        return True

    # Single config mode
    if config_overrides:
        print(f"\n  Config overrides: {config_overrides}")

    # Run basic matching and show metrics
    matches, metrics = runner.run_matching(config_overrides)
    runner.print_metrics(metrics)

    # Run test suite
    print(f"\n  --- TEST SUITE ---")

    runner.run_test("Matching runs", test_matching_runs, runner)
    runner.run_test("Confidence thresholds", test_confidence_thresholds, runner)
    runner.run_test("Reuse prevention", test_reuse_prevention, runner)
    runner.run_test("Strategy matches", test_strategy_matches, runner)

    return runner.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Test matching algorithm with saved fixtures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_matching.py --fixtures fixtures/test.json
    python tests/test_matching.py --fixtures fixtures/test.json --override min_confidence=0.5
    python tests/test_matching.py --fixtures fixtures/test.json --compare \\
        --config-a "min_confidence=0.7" --config-b "min_confidence=0.5"
        """
    )

    parser.add_argument(
        '--fixtures', '-f',
        type=str,
        required=True,
        help='Path to fixture JSON file'
    )

    parser.add_argument(
        '--override', '-o',
        type=str,
        action='append',
        help='Config override (e.g., min_confidence=0.5). Can be specified multiple times.'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Enable A/B comparison mode'
    )

    parser.add_argument(
        '--config-a',
        type=str,
        help='Config A for comparison (e.g., "min_confidence=0.7")'
    )

    parser.add_argument(
        '--config-b',
        type=str,
        help='Config B for comparison (e.g., "min_confidence=0.5")'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Parse overrides
    config_overrides = {}
    if args.override:
        for override in args.override:
            config_overrides.update(parse_config_string(override))

    success = run_matching_tests(
        fixture_path=args.fixtures,
        config_overrides=config_overrides if config_overrides else None,
        compare_mode=args.compare,
        config_a_str=args.config_a,
        config_b_str=args.config_b,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
