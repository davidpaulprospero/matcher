"""Dedicated logging system for high matches mode.

Provides a persistent log file for tracking iterations, coverage analysis,
and recovery keyword generation in high matches mode.

Pattern: Based on healing_logger.py - standalone logger with JSON + text output,
atomic writes, and session tracking.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import IterativeMatchState
    from .coverage_analyzer import CoverageReport

logger = logging.getLogger(__name__)


@dataclass
class HighMatchesLogEntry:
    """Single log entry for high matches mode activity."""
    timestamp: datetime
    entry_type: str  # session_start, coverage_analysis, recovery_keywords, etc.
    iteration: int
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class HighMatchesLogger:
    """Dedicated logging for high matches mode iterations.

    Provides dual output:
    - Human-readable .log file with formatted text
    - Machine-parseable .json file for analysis

    Thread-safe and uses atomic writes for crash safety.
    """

    def __init__(self, log_dir: Path, session_id: str = None):
        """Initialize high matches logger.

        Args:
            log_dir: Directory for log files
            session_id: Optional session ID (auto-generated if not provided)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[HighMatchesLogEntry] = []
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)

        # Create log files
        self.log_file = self.log_dir / f"high_matches_{self.session_id}.log"
        self.json_file = self.log_dir / f"high_matches_{self.session_id}.json"

        # Store config for summary
        self._config: Dict[str, Any] = {}
        self._initial_state: Dict[str, Any] = {}

        # Initialize files
        self._write_header()

        logger.info(f"[high_matches_logger] Session ID: {self.session_id}")
        logger.info(f"[high_matches_logger] Log file: {self.log_file.name}")

    def _write_header(self):
        """Write log file header."""
        header = [
            "=" * 80,
            f"HIGH MATCHES MODE SESSION: {self.session_id}",
            "=" * 80,
            f"Started: {self._start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ]
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")

    def log_session_start(self, config: Any, state: Any):
        """Log session start with configuration and initial state.

        Args:
            config: Pipeline configuration object
            state: Pipeline state object
        """
        hmm_config = config.matching.high_matches_mode

        self._config = {
            "target_confidence": hmm_config.target_confidence,
            "coverage_target": hmm_config.coverage_target,
            "max_iterations": hmm_config.max_iterations,
            "videos_per_iteration": hmm_config.videos_per_iteration,
            "keyword_strategy": hmm_config.keyword_strategy,
        }

        self._initial_state = {
            "total_segments": len(state.voiceover_segments) if state.voiceover_segments else 0,
            "initial_matches": len(state.matches) if state.matches else 0,
            "initial_videos": len(state.downloaded_videos) if state.downloaded_videos else 0,
            "initial_keywords": len(state.keywords) if state.keywords else 0,
        }

        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="session_start",
            iteration=0,
            data={
                "config": self._config,
                "initial_state": self._initial_state,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        lines = [
            f"Config: target_confidence={self._config['target_confidence']:.0%}, "
            f"coverage_target={self._config['coverage_target']:.0%}, "
            f"max_iterations={self._config['max_iterations']}",
            f"Initial: {self._initial_state['total_segments']} segments, "
            f"{self._initial_state['initial_matches']} matches, "
            f"{self._initial_state['initial_videos']} videos",
            "",
        ]
        self._append_text("\n".join(lines))

    def log_coverage_analysis(self, report: 'CoverageReport', iteration: int = 0):
        """Log coverage analysis results.

        Args:
            report: CoverageReport from coverage analyzer
            iteration: Current iteration number (0 for initial)
        """
        data = {
            "total_segments": report.total_segments,
            "high_confidence": report.high_confidence,
            "medium_confidence": report.medium_confidence,
            "low_confidence": report.low_confidence,
            "coverage_ratio": report.coverage_ratio,
            "target_confidence": report.target_confidence,
            "weak_segment_count": len(report.weak_segments),
            "weak_segment_ids": [ws.segment_id for ws in report.weak_segments[:20]],  # First 20
        }

        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="coverage_analysis",
            iteration=iteration,
            data=data,
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"[{ts}] COVERAGE ANALYSIS (iteration {iteration})",
            f"  Total: {report.total_segments} | "
            f"High (>={report.target_confidence:.0%}): {report.high_confidence} | "
            f"Medium (70-{report.target_confidence:.0%}): {report.medium_confidence} | "
            f"Low (<70%): {report.low_confidence}",
            f"  Coverage: {report.coverage_ratio:.1%} (target: {self._config.get('coverage_target', 0.85):.0%})",
            f"  Weak segments: {len(report.weak_segments)}",
        ]
        if report.weak_segments:
            weakest = report.weak_segments[0]
            lines.append(f"  Weakest: {weakest.segment_id} ({weakest.current_confidence:.2f})")
        lines.append("")
        self._append_text("\n".join(lines))

    def log_recovery_keywords(self, keywords: List[str], strategy: str,
                             weak_count: int, iteration: int):
        """Log recovery keyword generation.

        Args:
            keywords: Generated keywords
            strategy: Strategy used (weak_segments, diversify, both)
            weak_count: Number of weak segments targeted
            iteration: Current iteration number
        """
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="recovery_keywords",
            iteration=iteration,
            data={
                "keywords": keywords,
                "strategy": strategy,
                "weak_segment_count": weak_count,
                "keyword_count": len(keywords),
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"[{ts}] RECOVERY KEYWORDS (strategy={strategy})",
            f"  Generated {len(keywords)} keywords from {weak_count} weak segments:",
        ]
        for i, kw in enumerate(keywords[:10], 1):
            lines.append(f"    {i}. {kw}")
        if len(keywords) > 10:
            lines.append(f"    ... and {len(keywords) - 10} more")
        lines.append("")
        self._append_text("\n".join(lines))

    def log_iteration_start(self, iteration: int, coverage: float, weak_count: int):
        """Log start of an iteration.

        Args:
            iteration: Iteration number
            coverage: Current coverage ratio
            weak_count: Number of weak segments
        """
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="iteration_start",
            iteration=iteration,
            data={
                "current_coverage": coverage,
                "weak_segment_count": weak_count,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"[{ts}] ITERATION {iteration} START",
            f"  Current coverage: {coverage:.1%}",
            f"  Targeting {weak_count} weak segments",
            "",
        ]
        self._append_text("\n".join(lines))

    def log_download_result(self, videos_added: int, total_videos: int,
                           success: bool, iteration: int, source: str = "youtube"):
        """Log download stage result.

        Args:
            videos_added: Number of new videos downloaded
            total_videos: Total videos after download
            success: Whether download succeeded
            iteration: Current iteration number
            source: Source of videos (youtube, pexels, etc.)
        """
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="download_result",
            iteration=iteration,
            data={
                "videos_added": videos_added,
                "total_videos": total_videos,
                "success": success,
                "source": source,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        status = "COMPLETE" if success else "FAILED"
        source_label = f" ({source})" if source != "youtube" else ""
        self._append_text(
            f"[{ts}] DOWNLOAD {status}{source_label}\n"
            f"  Videos: {total_videos - videos_added} -> {total_videos} (+{videos_added} new)\n\n"
        )

    def log_rematch_result(self, new_coverage: float, previous_coverage: float,
                          match_count: int, iteration: int):
        """Log rematch stage result.

        Args:
            new_coverage: Coverage after rematch
            previous_coverage: Coverage before rematch
            match_count: Total matches after rematch
            iteration: Current iteration number
        """
        improvement = new_coverage - previous_coverage

        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="rematch_result",
            iteration=iteration,
            data={
                "new_coverage": new_coverage,
                "previous_coverage": previous_coverage,
                "improvement": improvement,
                "match_count": match_count,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        sign = "+" if improvement >= 0 else ""
        self._append_text(
            f"[{ts}] REMATCH COMPLETE\n"
            f"  Matches: {match_count}\n"
            f"  Coverage: {previous_coverage:.1%} -> {new_coverage:.1%} ({sign}{improvement:.1%})\n\n"
        )

    def log_recovery_strategy(self, strategy: str, result: str, videos_added: int, iteration: int):
        """Log a recovery strategy attempt.

        Args:
            strategy: Name of recovery strategy
            result: "success", "failed", or "skipped"
            videos_added: Number of new videos found
            iteration: Current iteration number
        """
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="recovery_strategy",
            iteration=iteration,
            data={
                "strategy": strategy,
                "result": result,
                "videos_added": videos_added,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        if result == "success":
            marker = f"SUCCESS: +{videos_added} videos"
        else:
            marker = "no new videos"
        self._append_text(f"[{ts}]   Recovery strategy '{strategy}': {marker}\n")

    def log_iteration_end(self, iteration: int, coverage: float,
                         continue_iteration: bool, reason: str):
        """Log end of an iteration.

        Args:
            iteration: Iteration number
            coverage: Final coverage for this iteration
            continue_iteration: Whether to continue with another iteration
            reason: Reason for stopping or continuing
        """
        entry = HighMatchesLogEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="iteration_end",
            iteration=iteration,
            data={
                "coverage": coverage,
                "continue": continue_iteration,
                "reason": reason,
            },
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write to text log
        ts = datetime.now().strftime("%H:%M:%S")
        status = "continuing..." if continue_iteration else f"stopping: {reason}"
        self._append_text(f"[{ts}] ITERATION {iteration} END\n  {status}\n\n")

    def log_final_report(self, state: 'IterativeMatchState', coverage: 'CoverageReport'):
        """Log final session report.

        Args:
            state: Final iterative match state
            coverage: Final coverage report
        """
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self._start_time).total_seconds()

        summary = {
            "duration_seconds": duration,
            "iterations": state.iteration_count,
            "coverage_history": state.coverage_history,
            "videos_added_total": sum(state.videos_added_per_iteration),
            "final_coverage": state.final_coverage,
            "target_achieved": state.target_achieved,
            "weak_segments_remaining": len(coverage.weak_segments),
        }

        entry = HighMatchesLogEntry(
            timestamp=end_time,
            entry_type="session_end",
            iteration=state.iteration_count,
            data=summary,
            session_id=self.session_id,
        )
        self._write_entry(entry)

        # Write summary to text log
        lines = [
            "",
            "=" * 80,
            "SESSION SUMMARY",
            "=" * 80,
            f"Duration: {int(duration // 60)}m {int(duration % 60)}s",
            f"Iterations: {state.iteration_count}",
            f"Videos downloaded: {self._initial_state.get('initial_videos', 0)} -> "
            f"{self._initial_state.get('initial_videos', 0) + sum(state.videos_added_per_iteration)} "
            f"(+{sum(state.videos_added_per_iteration)})",
            "",
            "Coverage progression:",
        ]

        for i, cov in enumerate(state.coverage_history):
            label = "Initial" if i == 0 else f"Iteration {i}"
            target = self._config.get('coverage_target', 0.85)
            marker = " [TARGET MET]" if cov >= target else ""
            total = coverage.total_segments
            lines.append(f"  {label}: {cov:.1%} ({int(cov * total)}/{total}){marker}")

        lines.append("")
        if state.target_achieved:
            lines.append("Result: SUCCESS - Coverage target achieved")
        else:
            lines.append(f"Result: INCOMPLETE - {len(coverage.weak_segments)} weak segments remaining")
            if coverage.weak_segments:
                weakest = coverage.weak_segments[0]
                lines.append(f"  Lowest: {weakest.segment_id} ({weakest.current_confidence:.2f})")

        lines.append("=" * 80)
        self._append_text("\n".join(lines))

        # Write final JSON with summary
        self._write_final_json(summary)

        logger.info(f"[high_matches_logger] Session complete. Log: {self.log_file.name}")

    def _write_entry(self, entry: HighMatchesLogEntry):
        """Write entry to in-memory list (thread-safe)."""
        with self._lock:
            self.entries.append(entry)

    def _append_text(self, text: str):
        """Append text to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"[high_matches_logger] Failed to write log: {e}")

    def _write_final_json(self, summary: Dict[str, Any]):
        """Write final JSON log with atomic write."""
        temp_path = None
        try:
            output = {
                "session_id": self.session_id,
                "config": self._config,
                "initial_state": self._initial_state,
                "summary": summary,
                "entries": [e.to_dict() for e in self.entries],
            }

            # Atomic write: temp file then rename
            fd, temp_path = tempfile.mkstemp(
                dir=self.log_dir,
                prefix='.high_matches_',
                suffix='.json.tmp'
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2)
            except Exception:
                raise

            os.replace(temp_path, self.json_file)
            temp_path = None  # Successful, don't delete

        except Exception as e:
            logger.error(f"[high_matches_logger] Failed to write JSON log: {e}")
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def get_log_paths(self) -> Dict[str, Path]:
        """Get paths to log files.

        Returns:
            Dict with 'text' and 'json' keys
        """
        return {
            "text": self.log_file,
            "json": self.json_file,
        }
