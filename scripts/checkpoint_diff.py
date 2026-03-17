#!/usr/bin/env python3
"""
Checkpoint Comparison Utility

Compares two checkpoint files between projects to understand differences in:
- Video IDs
- Segments
- Match quality (confidence scores)

Usage:
    python scripts/checkpoint_diff.py <checkpoint1> <checkpoint2>
    python scripts/checkpoint_diff.py project1/checkpoint.json project2/checkpoint.json --detailed
    python scripts/checkpoint_diff.py cp1.json cp2.json --json

Exit codes:
    0 - Comparison complete
    1 - Error (file not found, parse error, etc.)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load checkpoint from file, handling compression."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if p.suffix == '.gz':
        import gzip
# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_video_ids_from_stage(stage_data: Dict[str, Any], stage_name: str) -> Set[str]:
    """Extract video IDs from a checkpoint stage."""
    video_ids = set()

    if stage_name == 'video_search':
        # video_search has 'results' with video_id
        results = stage_data.get('results', [])
        for r in results:
            if isinstance(r, dict) and 'video_id' in r:
                video_ids.add(r['video_id'])

    elif stage_name == 'caption':
        # caption has 'caption_results' with video_id
        caption_results = stage_data.get('caption_results', {})
        if isinstance(caption_results, dict):
            video_ids.update(caption_results.keys())
        elif isinstance(caption_results, list):
            for r in caption_results:
                if isinstance(r, dict) and 'video_id' in r:
                    video_ids.add(r['video_id'])

    elif stage_name == 'match':
        # match has 'matches' with video_id
        matches = stage_data.get('matches', [])
        for m in matches:
            if isinstance(m, dict) and 'video_id' in m:
                video_ids.add(m['video_id'])

    elif stage_name == 'iterative_match':
        # iterative_match has 'matches' with video_id
        matches = stage_data.get('matches', [])
        for m in matches:
            if isinstance(m, dict) and 'video_id' in m:
                video_ids.add(m['video_id'])

    elif stage_name == 'download_segments':
        # download_segments has 'downloaded' with video_id
        downloaded = stage_data.get('downloaded', [])
        for d in downloaded:
            if isinstance(d, dict) and 'video_id' in d:
                video_ids.add(d['video_id'])

    return video_ids


def get_matches_from_stage(stage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract matches from a checkpoint stage."""
    if 'matches' in stage_data:
        return stage_data['matches']
    return []


def get_confidence_scores(matches: List[Dict[str, Any]]) -> List[float]:
    """Extract confidence scores from matches."""
    scores = []
    for m in matches:
        if isinstance(m, dict) and 'confidence' in m:
            scores.append(m['confidence'])
    return scores


def compare_video_ids(cp1: Dict, cp2: Dict) -> Dict[str, Any]:
    """Compare video IDs between two checkpoints."""
    stages = ['video_search', 'caption', 'match', 'iterative_match', 'download_segments']

    results = {}
    for stage in stages:
        stage1 = cp1.get(stage, {})
        stage2 = cp2.get(stage, {})

        ids1 = get_video_ids_from_stage(stage1, stage)
        ids2 = get_video_ids_from_stage(stage2, stage)

        added = ids2 - ids1
        removed = ids1 - ids2
        common = ids1 & ids2

        results[stage] = {
            'count1': len(ids1),
            'count2': len(ids2),
            'added': sorted(added),
            'removed': sorted(removed),
            'common_count': len(common),
            'change': len(ids2) - len(ids1)
        }

    return results


def compare_segments(cp1: Dict, cp2: Dict) -> Dict[str, Any]:
    """Compare matched segments count between checkpoints."""
    results = {}

    for stage in ['match', 'iterative_match']:
        matches1 = get_matches_from_stage(cp1.get(stage, {}))
        matches2 = get_matches_from_stage(cp2.get(stage, {}))

        # Count segments (each match is a segment)
        results[stage] = {
            'count1': len(matches1),
            'count2': len(matches2),
            'change': len(matches2) - len(matches1)
        }

    return results


def compare_confidence_scores(cp1: Dict, cp2: Dict) -> Dict[str, Any]:
    """Compare confidence score distributions between checkpoints."""
    results = {}

    for stage in ['match', 'iterative_match']:
        matches1 = get_matches_from_stage(cp1.get(stage, {}))
        matches2 = get_matches_from_stage(cp2.get(stage, {}))

        scores1 = get_confidence_scores(matches1)
        scores2 = get_confidence_scores(matches2)

        results[stage] = {
            'count1': len(scores1),
            'count2': len(scores2),
            'avg1': sum(scores1) / len(scores1) if scores1 else 0,
            'avg2': sum(scores2) / len(scores2) if scores2 else 0,
            'min1': min(scores1) if scores1 else 0,
            'min2': min(scores2) if scores2 else 0,
            'max1': max(scores1) if scores1 else 0,
            'max2': max(scores2) if scores2 else 0,
            'distribution1': _score_distribution(scores1),
            'distribution2': _score_distribution(scores2)
        }

    return results


def _score_distribution(scores: List[float]) -> Dict[str, int]:
    """Create a distribution histogram of confidence scores."""
    if not scores:
        return {}

    buckets = {
        '0.0-0.2': 0,
        '0.2-0.4': 0,
        '0.4-0.6': 0,
        '0.6-0.8': 0,
        '0.8-1.0': 0
    }

    for s in scores:
        if s < 0.2:
            buckets['0.0-0.2'] += 1
        elif s < 0.4:
            buckets['0.2-0.4'] += 1
        elif s < 0.6:
            buckets['0.4-0.6'] += 1
        elif s < 0.8:
            buckets['0.6-0.8'] += 1
        else:
            buckets['0.8-1.0'] += 1

    return buckets


def get_detailed_segments(cp1: Dict, cp2: Dict) -> List[Dict[str, Any]]:
    """Get per-segment differences between checkpoints."""
    detailed = []

    for stage in ['match', 'iterative_match']:
        matches1 = {m.get('segment_id'): m for m in get_matches_from_stage(cp1.get(stage, {})) if m.get('segment_id')}
        matches2 = {m.get('segment_id'): m for m in get_matches_from_stage(cp2.get(stage, {})) if m.get('segment_id')}

        all_segments = set(matches1.keys()) | set(matches2.keys())

        for seg_id in sorted(all_segments):
            m1 = matches1.get(seg_id)
            m2 = matches2.get(seg_id)

            if m1 and not m2:
                detailed.append({
                    'segment_id': seg_id,
                    'stage': stage,
                    'change_type': 'removed',
                    'old_video_id': m1.get('video_id'),
                    'new_video_id': None,
                    'old_confidence': m1.get('confidence'),
                    'new_confidence': None
                })
            elif m2 and not m1:
                detailed.append({
                    'segment_id': seg_id,
                    'stage': stage,
                    'change_type': 'added',
                    'old_video_id': None,
                    'new_video_id': m2.get('video_id'),
                    'old_confidence': None,
                    'new_confidence': m2.get('confidence')
                })
            elif m1 and m2:
                if m1.get('video_id') != m2.get('video_id') or m1.get('confidence') != m2.get('confidence'):
                    detailed.append({
                        'segment_id': seg_id,
                        'stage': stage,
                        'change_type': 'modified',
                        'old_video_id': m1.get('video_id'),
                        'new_video_id': m2.get('video_id'),
                        'old_confidence': m1.get('confidence'),
                        'new_confidence': m2.get('confidence')
                    })

    return detailed


def format_human(cp1_path: str, cp2_path: str, video_id_diff: Dict, segments_diff: Dict,
                 confidence_diff: Dict, detailed: List[Dict]) -> str:
    """Format comparison for human-readable output."""
    lines = []

    lines.append("=" * 70)
    lines.append(f"Checkpoint Comparison: {cp1_path} vs {cp2_path}")
    lines.append("=" * 70)

    # Video IDs section
    lines.append("\n--- Video ID Differences ---")
    total_change = 0
    for stage, data in video_id_diff.items():
        if data['change'] != 0 or data['added'] or data['removed']:
            lines.append(f"\n{stage}:")
            lines.append(f"  Count: {data['count1']} -> {data['count2']} ({data['change']:+d})")
            if data['added']:
                lines.append(f"  Added: {len(data['added'])} video(s)")
                for vid in data['added'][:5]:
                    lines.append(f"    + {vid}")
                if len(data['added']) > 5:
                    lines.append(f"    ... and {len(data['added']) - 5} more")
            if data['removed']:
                lines.append(f"  Removed: {len(data['removed'])} video(s)")
                for vid in data['removed'][:5]:
                    lines.append(f"    - {vid}")
                if len(data['removed']) > 5:
                    lines.append(f"    ... and {len(data['removed']) - 5} more")
        total_change += data['change']

    # Segments section
    lines.append("\n--- Matched Segments Count ---")
    for stage, data in segments_diff.items():
        lines.append(f"{stage}: {data['count1']} -> {data['count2']} ({data['change']:+d})")

    # Confidence scores section
    lines.append("\n--- Confidence Score Distribution ---")
    for stage, data in confidence_diff.items():
        lines.append(f"\n{stage}:")
        lines.append(f"  Count: {data['count1']} -> {data['count2']}")
        lines.append(f"  Average: {data['avg1']:.3f} -> {data['avg2']:.3f}")
        lines.append(f"  Range: [{data['min1']:.3f}, {data['max1']:.3f}] -> [{data['min2']:.3f}, {data['max2']:.3f}]")

        # Show distribution comparison
        lines.append("  Distribution:")
        d1 = data['distribution1']
        d2 = data['distribution2']
        for bucket in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
            c1 = d1.get(bucket, 0)
            c2 = d2.get(bucket, 0)
            lines.append(f"    {bucket}: {c1} -> {c2} ({c2-c1:+d})")

    # Detailed segment changes
    if detailed:
        lines.append("\n--- Detailed Segment Changes ---")
        lines.append(f"Total: {len(detailed)} segment(s) changed\n")
        for d in detailed[:20]:
            lines.append(f"[{d['stage']}] {d['segment_id']}: {d['change_type']}")
            if d['change_type'] == 'modified':
                lines.append(f"  Video: {d['old_video_id']} -> {d['new_video_id']}")
                lines.append(f"  Confidence: {d['old_confidence']} -> {d['new_confidence']}")
            elif d['change_type'] == 'added':
                lines.append(f"  Video: {d['new_video_id']} (confidence: {d['new_confidence']})")
            elif d['change_type'] == 'removed':
                lines.append(f"  Video: {d['old_video_id']} (confidence: {d['old_confidence']})")

        if len(detailed) > 20:
            lines.append(f"\n... and {len(detailed) - 20} more changes (use --detailed to see all)")

    return "\n".join(lines)


def format_json(cp1_path: str, cp2_path: str, video_id_diff: Dict, segments_diff: Dict,
                confidence_diff: Dict, detailed: List[Dict]) -> str:
    """Format comparison as JSON."""
    output = {
        'checkpoint1': cp1_path,
        'checkpoint2': cp2_path,
        'video_id_differences': video_id_diff,
        'segments_difference': segments_diff,
        'confidence_difference': confidence_diff,
        'detailed_changes': detailed if detailed else []
    }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Compare checkpoint files between projects'
    )
    parser.add_argument('checkpoint1', help='Path to first checkpoint file')
    parser.add_argument('checkpoint2', help='Path to second checkpoint file')
    parser.add_argument('--detailed', action='store_true',
                        help='Show per-segment differences')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    try:
        cp1 = load_checkpoint(args.checkpoint1)
        cp2 = load_checkpoint(args.checkpoint2)

        # Compute all comparisons
        video_id_diff = compare_video_ids(cp1, cp2)
        segments_diff = compare_segments(cp1, cp2)
        confidence_diff = compare_confidence_scores(cp1, cp2)

        detailed = []
        if args.detailed:
            detailed = get_detailed_segments(cp1, cp2)

        # Output
        if args.json:
            print(format_json(args.checkpoint1, args.checkpoint2, video_id_diff,
                            segments_diff, confidence_diff, detailed))
        else:
            print(format_human(args.checkpoint1, args.checkpoint2, video_id_diff,
                             segments_diff, confidence_diff, detailed))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in checkpoint: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
