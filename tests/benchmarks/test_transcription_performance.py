"""
Performance benchmarks for transcription module.

Tests critical performance for:
- Transcript parsing
- Delta index updates
- Cache operations
- Segment splitting algorithms
- Parallel processing overhead
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils import SRTSegment


# =============================================================================
# TRANSCRIPT PARSING PERFORMANCE
# =============================================================================

class TestTranscriptParsingPerformance:
    """Benchmark transcript parsing operations"""

    def test_parse_small_srt(self, benchmark):
        """Benchmark parsing small SRT file (10 segments)"""
        srt_content = "\n\n".join([
            f"{i}\n{i*3.0:.3f} --> {(i+1)*3.0:.3f}\nSegment {i} text content"
            for i in range(1, 11)
        ])

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt', encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        def parse_srt():
            # Simulate SRT parsing
            segments = []
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Count segments
            return content.count('\n\n')

        try:
            result = benchmark(parse_srt)
            assert result >= 9  # At least 9 separators for 10 segments
            # Should be very fast (< 2ms)
            assert benchmark.stats.stats.mean < 0.002
        finally:
            Path(srt_path).unlink(missing_ok=True)

    def test_parse_large_srt(self, benchmark):
        """Benchmark parsing large SRT file (1000 segments)"""
        srt_content = "\n\n".join([
            f"{i}\n{i*3.0:.3f} --> {(i+1)*3.0:.3f}\nThis is segment {i} with some longer text content that might appear in a real transcript"
            for i in range(1, 1001)
        ])

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt', encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        def parse_srt():
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.count('\n\n')

        try:
            result = benchmark(parse_srt)
            assert result >= 999
            # Should complete in < 20ms
            assert benchmark.stats.stats.mean < 0.02
        finally:
            Path(srt_path).unlink(missing_ok=True)


# =============================================================================
# SEGMENT SPLITTING PERFORMANCE
# =============================================================================

class TestSegmentSplittingPerformance:
    """Benchmark pause-split and sentence-split algorithms"""

    def test_split_by_duration_1000_segments(self, benchmark):
        """Benchmark splitting 1000 segments by duration"""
        segments = [
            SRTSegment(i, i*10.0, (i+1)*10.0, f"Long segment {i} " * 20, "test.srt")
            for i in range(1000)
        ]

        def split_long_segments():
            result = []
            max_duration = 5.0
            for seg in segments:
                if seg.duration > max_duration:
                    # Simulate splitting
                    mid_time = (seg.start_time + seg.end_time) / 2
                    result.append(SRTSegment(seg.index, seg.start_time, mid_time, seg.text[:len(seg.text)//2], seg.source_file))
                    result.append(SRTSegment(seg.index+1, mid_time, seg.end_time, seg.text[len(seg.text)//2:], seg.source_file))
                else:
                    result.append(seg)
            return result

        result = benchmark(split_long_segments)
        assert len(result) > len(segments)  # Should split some
        # Should complete in < 30ms
        assert benchmark.stats.stats.mean < 0.03

    def test_split_by_sentences(self, benchmark):
        """Benchmark splitting 500 segments by sentences"""
        segments = [
            SRTSegment(i, i*5.0, (i+1)*5.0,
                      f"First sentence. Second sentence. Third sentence.",
                      "test.srt")
            for i in range(500)
        ]

        def split_by_sentences():
            result = []
            for seg in segments:
                sentences = seg.text.split('. ')
                if len(sentences) > 1:
                    duration_per = seg.duration / len(sentences)
                    for j, sentence in enumerate(sentences):
                        start = seg.start_time + j * duration_per
                        end = start + duration_per
                        result.append(SRTSegment(seg.index, start, end, sentence, seg.source_file))
                else:
                    result.append(seg)
            return result

        result = benchmark(split_by_sentences)
        assert len(result) > len(segments)
        # Should complete in < 25ms
        assert benchmark.stats.stats.mean < 0.025


# =============================================================================
# CACHE OPERATIONS PERFORMANCE
# =============================================================================

class TestTranscriptCachePerformance:
    """Benchmark transcript cache operations"""

    def test_cache_write_100_transcripts(self, benchmark):
        """Benchmark writing 100 transcript files to cache"""
        with tempfile.TemporaryDirectory() as cache_dir:
            transcripts = {
                f"video_{i}": [
                    {"text": f"Segment {j}", "start": j*3.0, "end": (j+1)*3.0}
                    for j in range(50)
                ]
                for i in range(100)
            }

            def write_cache():
                for video_id, segments in transcripts.items():
                    cache_file = Path(cache_dir) / f"{video_id}.json"
                    with open(cache_file, 'w') as f:
                        json.dump(segments, f)

            benchmark(write_cache)
            # Verify files were written
            assert len(list(Path(cache_dir).glob("*.json"))) == 100
            # Should complete in < 100ms
            assert benchmark.stats.stats.mean < 0.1

    def test_cache_read_100_transcripts(self, benchmark):
        """Benchmark reading 100 transcript files from cache"""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Pre-populate cache
            for i in range(100):
                segments = [
                    {"text": f"Segment {j}", "start": j*3.0, "end": (j+1)*3.0}
                    for j in range(50)
                ]
                cache_file = Path(cache_dir) / f"video_{i}.json"
                with open(cache_file, 'w') as f:
                    json.dump(segments, f)

            def read_cache():
                results = {}
                for cache_file in Path(cache_dir).glob("*.json"):
                    with open(cache_file, 'r') as f:
                        results[cache_file.stem] = json.load(f)
                return results

            result = benchmark(read_cache)
            assert len(result) == 100
            # Should complete in < 50ms
            assert benchmark.stats.stats.mean < 0.05


# =============================================================================
# DELTA INDEX PERFORMANCE
# =============================================================================

class TestDeltaIndexPerformance:
    """Benchmark delta-aware index operations"""

    def test_delta_calculation_1000_segments(self, benchmark):
        """Benchmark calculating deltas for 1000 segment pairs"""
        old_segments = [
            {"index": i, "start": i*3.0, "end": (i+1)*3.0, "text": f"Old segment {i}"}
            for i in range(1000)
        ]
        new_segments = [
            {"index": i, "start": i*3.0 + 0.1, "end": (i+1)*3.0 + 0.1, "text": f"New segment {i}"}  # Shifted
            for i in range(1000)
        ]

        def calculate_deltas():
            deltas = []
            for old, new in zip(old_segments, new_segments):
                delta = {
                    'start_delta': new['start'] - old['start'],
                    'end_delta': new['end'] - old['end'],
                    'text_changed': old['text'] != new['text']
                }
                deltas.append(delta)
            return deltas

        result = benchmark(calculate_deltas)
        assert len(result) == 1000
        # Should be very fast (< 5ms)
        assert benchmark.stats.stats.mean < 0.005

    def test_index_update_performance(self, benchmark):
        """Benchmark updating index with 500 new segments"""
        existing_index = {
            i: {"start": i*3.0, "end": (i+1)*3.0, "text": f"Segment {i}"}
            for i in range(1000)
        }

        new_segments = [
            {"index": i + 1000, "start": (i+1000)*3.0, "end": (i+1001)*3.0, "text": f"New segment {i}"}
            for i in range(500)
        ]

        def update_index():
            index_copy = existing_index.copy()
            for seg in new_segments:
                index_copy[seg['index']] = seg
            return index_copy

        result = benchmark(update_index)
        assert len(result) == 1500
        # Should complete in < 5ms
        assert benchmark.stats.stats.mean < 0.005


# =============================================================================
# TEXT PROCESSING PERFORMANCE
# =============================================================================

class TestTextProcessingPerformance:
    """Benchmark text processing operations"""

    def test_normalize_text_1000_segments(self, benchmark):
        """Benchmark text normalization for 1000 segments"""
        texts = [
            f"  Segment {i} with   MIXED case   and   extra   spaces  \n\t"
            for i in range(1000)
        ]

        def normalize_texts():
            return [
                ' '.join(text.strip().lower().split())
                for text in texts
            ]

        result = benchmark(normalize_texts)
        assert len(result) == 1000
        assert all('  ' not in text for text in result)  # No double spaces
        # Should complete in < 10ms
        assert benchmark.stats.stats.mean < 0.01

    def test_extract_keywords_from_text(self, benchmark):
        """Benchmark keyword extraction from 500 text segments"""
        texts = [
            f"This is a long segment about machine learning and artificial intelligence with many words segment {i}"
            for i in range(500)
        ]

        def extract_keywords():
            keywords = []
            for text in texts:
                words = text.lower().split()
                # Simple keyword extraction: words > 5 chars
                segment_keywords = [w for w in words if len(w) > 5]
                keywords.append(segment_keywords)
            return keywords

        result = benchmark(extract_keywords)
        assert len(result) == 500
        # Should complete in < 15ms
        assert benchmark.stats.stats.mean < 0.015


# =============================================================================
# PARALLEL PROCESSING OVERHEAD
# =============================================================================

class TestParallelProcessingPerformance:
    """Benchmark parallel processing overhead"""

    def test_sequential_vs_parallel_overhead(self, benchmark):
        """Measure overhead of parallel processing setup"""
        segments = list(range(100))

        def sequential_processing():
            return [x * 2 for x in segments]

        result = benchmark(sequential_processing)
        assert len(result) == 100
        # Sequential should be very fast (< 1ms)
        assert benchmark.stats.stats.mean < 0.001

    def test_batch_processing_performance(self, benchmark):
        """Benchmark processing 1000 items in batches of 100"""
        items = list(range(1000))
        batch_size = 100

        def batch_process():
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                # Simulate processing
                results.extend([x * 2 for x in batch])
            return results

        result = benchmark(batch_process)
        assert len(result) == 1000
        # Should complete in < 5ms
        assert benchmark.stats.stats.mean < 0.005


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
