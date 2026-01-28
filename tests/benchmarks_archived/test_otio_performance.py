"""
Performance benchmarks for OTIO timeline generation.

Run with: pytest tests/benchmarks/test_otio_performance.py -v
"""

import pytest
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class MockMatch:
    """Mock match for OTIO benchmarks."""
    voiceover_index: int
    video_id: str
    start_time: float
    end_time: float
    score: float
    text: str
    confidence: str


@dataclass
class MockMatchResult:
    """Mock match result for OTIO benchmarks."""
    primary: MockMatch
    alternatives: List[MockMatch]
    secondaries: List[MockMatch]


class TestOTIOPerformance:
    """Benchmark OTIO timeline generation operations."""

    @pytest.fixture
    def mock_matches(self) -> List[MockMatchResult]:
        """Create mock match results for benchmarking."""
        matches = []
        for i in range(100):
            primary = MockMatch(
                voiceover_index=i,
                video_id=f"video_{i:04d}",
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                score=0.95,
                text=f"Match {i} text",
                confidence="high"
            )

            alternatives = [
                MockMatch(
                    voiceover_index=i,
                    video_id=f"alt_video_{i:04d}_{j}",
                    start_time=float(i * 5),
                    end_time=float((i + 1) * 5),
                    score=0.85 - (j * 0.05),
                    text=f"Alt match {i}-{j} text",
                    confidence="medium"
                )
                for j in range(3)
            ]

            secondaries = [
                MockMatch(
                    voiceover_index=i,
                    video_id=f"sec_video_{i:04d}_{k}",
                    start_time=float(i * 5),
                    end_time=float((i + 1) * 5),
                    score=0.75 - (k * 0.05),
                    text=f"Sec match {i}-{k} text",
                    confidence="low"
                )
                for k in range(3)
            ]

            matches.append(MockMatchResult(
                primary=primary,
                alternatives=alternatives,
                secondaries=secondaries
            ))

        return matches

    @pytest.mark.fast
    def test_timeline_creation_speed(self, mock_matches, benchmark):
        """Benchmark timeline creation speed."""
        from src.otio import create_timeline
        from src.config import load_config

        config = load_config()

        def create_otio_timeline():
            return create_timeline(
                matches=mock_matches,
                config=config,
                frame_rate=30.0
            )

        result = benchmark(create_otio_timeline)

        assert result is not None
        print(f"\nCreated timeline with {len(mock_matches)} segments")

        # Calculate throughput
        segments_per_second = len(mock_matches) / benchmark.stats['mean']
        print(f"Timeline generation: {segments_per_second:.1f} segments/second")

    @pytest.mark.fast
    def test_track_building_speed(self, mock_matches, benchmark):
        """Benchmark individual track building speed."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder()

        def build_track():
            clips = []
            for match_result in mock_matches:
                clip = builder._create_clip(
                    match=match_result.primary,
                    voiceover_index=match_result.primary.voiceover_index,
                    strategy_name="primary",
                    frame_rate=30.0
                )
                clips.append(clip)
            return clips

        result = benchmark(build_track)

        assert len(result) == len(mock_matches)
        print(f"\nBuilt track with {len(result)} clips")

    @pytest.mark.fast
    def test_otio_serialization_speed(self, mock_matches, temp_benchmark_dir, benchmark):
        """Benchmark OTIO serialization speed."""
        from src.otio import create_timeline, save_timeline
        from src.config import load_config

        config = load_config()
        timeline = create_timeline(mock_matches, config, frame_rate=30.0)
        output_file = temp_benchmark_dir / "benchmark_timeline.otio"

        def save_otio():
            save_timeline(timeline, str(output_file))

        benchmark(save_otio)

        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"\nOTIO file size: {file_size_mb:.2f} MB for {len(mock_matches)} segments")

    @pytest.mark.fast
    def test_edl_export_speed(self, mock_matches, temp_benchmark_dir, benchmark):
        """Benchmark EDL export speed."""
        from src.otio import create_timeline, save_timeline_as_edl
        from src.config import load_config

        config = load_config()
        timeline = create_timeline(mock_matches, config, frame_rate=30.0)
        output_file = temp_benchmark_dir / "benchmark_timeline.edl"

        def save_edl():
            save_timeline_as_edl(timeline, str(output_file))

        benchmark(save_edl)

        file_size_kb = output_file.stat().st_size / 1024
        print(f"\nEDL file size: {file_size_kb:.2f} KB")

    @pytest.mark.fast
    def test_xml_generation_speed(self, mock_matches, temp_benchmark_dir, benchmark):
        """Benchmark FCP7 XML generation speed."""
        from src.otio import create_timeline, generate_resolve_xml_with_bins
        from src.config import load_config

        config = load_config()
        timeline = create_timeline(mock_matches, config, frame_rate=30.0)
        output_file = temp_benchmark_dir / "benchmark_timeline.xml"

        def generate_xml():
            generate_resolve_xml_with_bins(timeline, str(output_file))

        benchmark(generate_xml)

        file_size_kb = output_file.stat().st_size / 1024
        print(f"\nXML file size: {file_size_kb:.2f} KB")

    @pytest.mark.fast
    def test_split_timeline_export_speed(self, mock_matches, temp_benchmark_dir, benchmark):
        """Benchmark split timeline export speed."""
        from src.otio import create_timeline, save_timeline_split
        from src.config import load_config

        config = load_config()
        timeline = create_timeline(mock_matches, config, frame_rate=30.0)
        output_prefix = str(temp_benchmark_dir / "split_timeline")

        def save_split():
            save_timeline_split(timeline, output_prefix)

        benchmark(save_split)

        # Count generated files
        split_files = list(temp_benchmark_dir.glob("split_timeline_*.otio"))
        print(f"\nGenerated {len(split_files)} split timeline files")

    @pytest.mark.fast
    def test_segment_map_generation_speed(self, mock_matches, benchmark):
        """Benchmark segment map generation speed."""
        from src.otio import create_timeline, generate_segment_map
        from src.config import load_config

        config = load_config()
        timeline = create_timeline(mock_matches, config, frame_rate=30.0)

        def generate_map():
            return generate_segment_map(timeline)

        result = benchmark(generate_map)

        assert result is not None
        print(f"\nGenerated segment map with {len(result)} entries")


class TestOTIOMemoryUsage:
    """Memory profiling for OTIO operations."""

    @pytest.mark.fast
    def test_timeline_memory_usage(self):
        """Profile memory usage of timeline creation."""
        import tracemalloc
        from src.otio import create_timeline
        from src.config import load_config

        # Create mock matches
        matches = []
        for i in range(1000):  # Large timeline
            primary = MockMatch(
                voiceover_index=i,
                video_id=f"video_{i:04d}",
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                score=0.95,
                text=f"Match {i} text",
                confidence="high"
            )
            matches.append(MockMatchResult(
                primary=primary,
                alternatives=[],
                secondaries=[]
            ))

        config = load_config()

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        timeline = create_timeline(matches, config, frame_rate=30.0)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        memory_per_segment_kb = (peak_memory - initial_memory) / 1024 / len(matches)

        print(f"\nTimeline memory: {memory_used_mb:.2f} MB for {len(matches)} segments")
        print(f"Memory per segment: {memory_per_segment_kb:.2f} KB")

        # Should use less than 500MB for 1000 segments
        assert memory_used_mb < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
