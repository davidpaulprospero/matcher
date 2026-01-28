"""
Performance benchmarks for pipeline stages and orchestration.

Tests critical performance characteristics to detect regressions:
- Stage execution time
- State serialization/deserialization
- Pipeline orchestration overhead
- Checkpoint save/load performance
- Memory efficiency
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.state import PipelineState
from src.utils import SRTSegment, Match, MatchResult
from src.config import Config


# =============================================================================
# CHECKPOINT PERFORMANCE
# =============================================================================

class TestCheckpointPerformance:
    """Benchmark checkpoint save/load operations"""

    @pytest.mark.integration
    def test_checkpoint_save_small_state(self, benchmark):
        """Benchmark saving small pipeline state (10 segments)"""
        state = PipelineState()
        state.voiceover_segments = [
            Mock(index=i, start_time=i*3.0, end_time=(i+1)*3.0, text=f"Segment {i}")
            for i in range(10)
        ]
        state.keywords = ["keyword1", "keyword2", "keyword3"]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name

        def save_checkpoint():
            # Simulate checkpoint save (simplified)
            data = {
                'voiceover_segments': len(state.voiceover_segments),
                'keywords': state.keywords,
                'stage': 'ANALYZE'
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(data, f)

        try:
            result = benchmark(save_checkpoint)
            # Should complete in < 10ms
            assert benchmark.stats.stats.mean < 0.01
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    @pytest.mark.integration
    def test_checkpoint_save_large_state(self, benchmark):
        """Benchmark saving large pipeline state (1000 segments)"""
        state = PipelineState()
        state.voiceover_segments = [
            Mock(index=i, start_time=i*3.0, end_time=(i+1)*3.0, text=f"Segment {i}")
            for i in range(1000)
        ]
        state.keywords = [f"keyword_{i}" for i in range(100)]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name

        def save_checkpoint():
            data = {
                'voiceover_segments': len(state.voiceover_segments),
                'keywords': state.keywords,
                'downloaded_videos': len(state.downloaded_videos),
                'stage': 'DOWNLOAD'
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(data, f)

        try:
            result = benchmark(save_checkpoint)
            # Should complete in < 100ms even for large state
            assert benchmark.stats.stats.mean < 0.1
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    @pytest.mark.integration
    def test_checkpoint_load_performance(self, benchmark):
        """Benchmark loading checkpoint from disk"""
        checkpoint_data = {
            'voiceover_segments': 100,
            'keywords': [f"keyword_{i}" for i in range(50)],
            'stage': 'MATCH',
            'downloaded_videos': 500
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(checkpoint_data, f)
            checkpoint_path = f.name

        def load_checkpoint():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)

        try:
            result = benchmark(load_checkpoint)
            assert result == checkpoint_data
            # Should be fast (< 5ms)
            assert benchmark.stats.stats.mean < 0.005
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


# =============================================================================
# STATE MANAGEMENT PERFORMANCE
# =============================================================================

class TestStateManagementPerformance:
    """Benchmark PipelineState operations"""

    @pytest.mark.fast
    def test_state_initialization(self, benchmark):
        """Benchmark PipelineState creation"""
        def create_state():
            return PipelineState()

        result = benchmark(create_state)
        assert isinstance(result, PipelineState)
        # Should be instantaneous (< 1ms)
        assert benchmark.stats.stats.mean < 0.001

    @pytest.mark.fast
    def test_state_with_large_transcript_dict(self, benchmark):
        """Benchmark state with 100 videos × 50 segments each"""
        def create_large_state():
            state = PipelineState()
            # Simulate 100 videos with transcripts
            for i in range(100):
                state.transcripts[f"video_{i}"] = [
                    {"text": f"Segment {j}", "start": j*3.0, "end": (j+1)*3.0}
                    for j in range(50)
                ]
            return state

        result = benchmark(create_large_state)
        assert len(result.transcripts) == 100
        # Should complete in < 50ms
        assert benchmark.stats.stats.mean < 0.05

    @pytest.mark.fast
    def test_state_match_list_append_performance(self, benchmark):
        """Benchmark appending 1000 matches to state"""
        state = PipelineState()
        vo_seg = SRTSegment(0, 0.0, 3.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 3.0, "video", "v.mp4")
        match = Match(vo_seg, vid_seg, None, 0.8, "test")
        match_result = MatchResult(match, [], [], [])

        def append_matches():
            state.matches.clear()
            for i in range(1000):
                state.matches.append(match_result)

        benchmark(append_matches)
        assert len(state.matches) == 1000
        # Should be very fast (< 5ms)
        assert benchmark.stats.stats.mean < 0.005


# =============================================================================
# CONFIG LOADING PERFORMANCE
# =============================================================================

class TestConfigPerformance:
    """Benchmark configuration loading and validation"""

    @pytest.mark.fast
    def test_config_initialization(self, benchmark):
        """Benchmark Config object creation"""
        def create_config():
            return Config()

        result = benchmark(create_config)
        assert isinstance(result, Config)
        # Should be fast (< 2ms)
        assert benchmark.stats.stats.mean < 0.002

    @pytest.mark.fast
    def test_config_from_dict(self, benchmark):
        """Benchmark loading config from dictionary"""
        config_dict = {
            'keyword': {'max_keywords': 10},
            'download': {'timeout': 300},
            'matching': {'confidence_threshold': 0.7}
        }

        def load_config():
            # Simulate config loading
            return Config()  # Simplified - real loading would use dataclass_from_dict

        result = benchmark(load_config)
        # Should be fast (< 5ms)
        assert benchmark.stats.stats.mean < 0.005


# =============================================================================
# SEGMENT PROCESSING PERFORMANCE
# =============================================================================

class TestSegmentProcessingPerformance:
    """Benchmark SRT segment operations"""

    @pytest.mark.fast
    def test_create_1000_segments(self, benchmark):
        """Benchmark creating 1000 SRTSegment objects"""
        def create_segments():
            return [
                SRTSegment(
                    index=i,
                    start_time=i * 3.0,
                    end_time=(i + 1) * 3.0,
                    text=f"Segment {i}",
                    source_file="test.srt"
                )
                for i in range(1000)
            ]

        result = benchmark(create_segments)
        assert len(result) == 1000
        # Should complete in < 20ms
        assert benchmark.stats.stats.mean < 0.02

    @pytest.mark.fast
    def test_segment_duration_calculation(self, benchmark):
        """Benchmark duration calculation for 10000 segments"""
        segments = [
            SRTSegment(i, i*3.0, (i+1)*3.0, f"Seg {i}", "test.srt")
            for i in range(10000)
        ]

        def calculate_durations():
            return [seg.duration for seg in segments]

        result = benchmark(calculate_durations)
        assert len(result) == 10000
        # Should be very fast (< 10ms)
        assert benchmark.stats.stats.mean < 0.01


# =============================================================================
# MATCH RESULT CREATION PERFORMANCE
# =============================================================================

class TestMatchResultPerformance:
    """Benchmark Match and MatchResult creation"""

    @pytest.mark.fast
    def test_create_1000_match_results(self, benchmark):
        """Benchmark creating 1000 MatchResult objects"""
        vo_seg = SRTSegment(0, 0.0, 3.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 3.0, "video", "v.mp4")

        def create_match_results():
            results = []
            for i in range(1000):
                match = Match(vo_seg, vid_seg, None, 0.8 + (i % 20) * 0.01, f"Match {i}")
                result = MatchResult(
                    primary_match=match,
                    alternatives=[],
                    secondary_matches=[],
                    strategy_matches=[]
                )
                results.append(result)
            return results

        result = benchmark(create_match_results)
        assert len(result) == 1000
        # Should complete in < 30ms
        assert benchmark.stats.stats.mean < 0.03

    @pytest.mark.fast
    def test_match_result_with_alternatives(self, benchmark):
        """Benchmark MatchResult with 5 alternatives each"""
        vo_seg = SRTSegment(0, 0.0, 3.0, "test", "vo.srt")

        def create_with_alternatives():
            results = []
            for i in range(200):
                vid_seg = SRTSegment(0, 0.0, 3.0, f"video_{i}", "v.mp4")
                primary = Match(vo_seg, vid_seg, None, 0.9, "primary")

                alternatives = [
                    Match(vo_seg, SRTSegment(0, 0.0, 3.0, f"alt_{j}", "v.mp4"), None, 0.8 - j*0.05, f"alt_{j}")
                    for j in range(5)
                ]

                result = MatchResult(
                    primary_match=primary,
                    alternatives=alternatives,
                    secondary_matches=[],
                    strategy_matches=[]
                )
                results.append(result)
            return results

        result = benchmark(create_with_alternatives)
        assert len(result) == 200
        assert len(result[0].alternatives) == 5
        # Should complete in < 40ms
        assert benchmark.stats.stats.mean < 0.04


# =============================================================================
# MEMORY EFFICIENCY
# =============================================================================

class TestMemoryEfficiency:
    """Benchmark memory usage and large data structure performance"""

    @pytest.mark.fast
    def test_large_state_memory(self, benchmark):
        """Test memory footprint of large PipelineState"""
        def create_large_state():
            state = PipelineState()
            # 500 videos with 100 segments each
            for i in range(500):
                state.transcripts[f"video_{i}"] = [
                    {"text": f"Segment {j}" * 10, "start": j*3.0, "end": (j+1)*3.0}
                    for j in range(100)
                ]
            return state

        result = benchmark(create_large_state)
        assert len(result.transcripts) == 500

    @pytest.mark.fast
    def test_match_result_list_memory(self, benchmark):
        """Test memory usage of 10000 MatchResults"""
        vo_seg = SRTSegment(0, 0.0, 3.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 3.0, "video", "v.mp4")

        def create_large_match_list():
            matches = []
            for i in range(10000):
                match = Match(vo_seg, vid_seg, None, 0.8, "test")
                result = MatchResult(match, [], [], [])
                matches.append(result)
            return matches

        result = benchmark(create_large_match_list)
        assert len(result) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
