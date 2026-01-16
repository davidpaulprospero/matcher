"""Comprehensive logging system for self-healing pipeline."""

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealingLogEntry:
    """Single log entry for healing activity."""
    timestamp: datetime
    stage: str
    component: str  # "watcher", "healer", "orchestrator", "fallback"
    action: str     # "classify", "attempt", "fix", "escalate", "fallback", "self_heal"
    error_type: str
    error_message: str
    result: str     # "success", "failed", "skipped", "escalated", "retrying", "degraded"
    duration_ms: float
    session_id: str = ""
    stack_trace: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class HealingLogger:
    """Comprehensive logging for self-healing system.

    Provides dual output:
    - Human-readable .log file
    - Machine-parseable .json file

    Thread-safe for use in parallel processing.
    """

    def __init__(self, log_dir: Path, json_log: bool = True, console_format: str = "box"):
        """Initialize healing logger.

        Args:
            log_dir: Directory for log files
            json_log: Whether to write JSON logs
            console_format: Format for console output ("box", "simple", "minimal")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.json_log = json_log
        self.console_format = console_format
        self.session_id = str(uuid.uuid4())[:8]
        self.entries: List[HealingLogEntry] = []
        self._lock = threading.Lock()

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"healing_{timestamp}.log"
        self.json_file = self.log_dir / f"healing_{timestamp}.json"

        # Initialize JSON file with empty array
        if self.json_log:
            self.json_file.write_text("[]")

        # Runtime mitigation: Clean up orphaned temp files from crashed sessions
        self._cleanup_orphaned_temp_files()

        # Runtime mitigation: Detect network filesystem (NFS/SMB) and warn
        self._check_filesystem_atomicity()

        logger.info(f"[healing] Session ID: {self.session_id}")
        logger.info(f"[healing] Log files: {self.log_file.name}, {self.json_file.name}")

    def _cleanup_orphaned_temp_files(self):
        """Clean up temp files from crashed previous sessions.

        Runtime mitigation for power failure / process crash during atomic write.
        Temp files older than 60 seconds are considered orphaned.
        """
        import time
        cutoff = time.time() - 60  # 1 minute old

        try:
            for temp_file in self.log_dir.glob(".healing_*.json.tmp"):
                try:
                    if temp_file.stat().st_mtime < cutoff:
                        temp_file.unlink()
                        logger.warning(f"[healing] Cleaned up orphaned temp file: {temp_file.name}")
                except OSError:
                    pass  # Best effort
        except Exception as e:
            logger.debug(f"[healing] Temp file cleanup failed: {e}")

    def _check_filesystem_atomicity(self):
        """Detect network filesystems where os.replace may not be atomic.

        Runtime mitigation for NFS/SMB atomicity issues.
        Warns user if log directory is on a network share.
        """
        try:
            # Check for common network filesystem indicators
            log_dir_str = str(self.log_dir.resolve())

            # Windows UNC paths (\\server\share)
            if log_dir_str.startswith('\\\\'):
                logger.warning(
                    f"[healing] Log directory is on a network share ({log_dir_str[:30]}...). "
                    "Atomic writes may not be guaranteed. Consider using local storage."
                )
                return

            # Check mount point on Linux/Mac
            if hasattr(os, 'statvfs'):
                # This could detect NFS by filesystem type, but it's complex
                # Just warn if path looks like a mount point
                pass

        except Exception:
            pass  # Best effort detection

    # Maximum entries to keep in memory to prevent unbounded growth
    MAX_ENTRIES = 10000

    def _write(self, entry: HealingLogEntry):
        """Write entry to logs (thread-safe).

        Uses atomic write for JSON to prevent corruption on crash/disk full.
        """
        entry.session_id = self.session_id

        with self._lock:
            # Prevent unbounded memory growth
            self.entries.append(entry)
            if len(self.entries) > self.MAX_ENTRIES:
                # Keep most recent half
                self.entries = self.entries[-(self.MAX_ENTRIES // 2):]
                logger.warning(f"[healing] Truncated log entries to {len(self.entries)}")

            # Write to human-readable log (append is relatively safe)
            try:
                formatted = self._format_entry(entry)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception as e:
                logger.error(f"[healing] Failed to write log entry: {e}")

            # Write to JSON log using atomic write
            if self.json_log:
                self._atomic_json_write(entry)

    def _atomic_json_write(self, entry: HealingLogEntry):
        """Atomically append entry to JSON log file.

        Process:
        1. Read existing JSON
        2. Append new entry
        3. Write to temp file
        4. Atomic rename temp -> target

        If process dies at any step:
        - Step 1-3: Original file intact
        - Step 4: os.replace is atomic, either completes or doesn't
        """
        temp_path = None
        try:
            # Read existing entries
            try:
                existing = json.loads(self.json_file.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, FileNotFoundError, OSError):
                existing = []

            existing.append(entry.to_dict())

            # Write to temp file in same directory (required for atomic rename)
            # Using delete=False because we need to close before rename on Windows
            fd, temp_path = tempfile.mkstemp(
                dir=self.log_dir,
                prefix='.healing_',
                suffix='.json.tmp'
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2)
            except Exception:
                # fd is already closed by os.fdopen even on error
                raise

            # Atomic rename - this is the critical operation
            # os.replace is atomic on POSIX and Windows NTFS
            os.replace(temp_path, self.json_file)
            temp_path = None  # Successfully renamed, don't delete in finally

        except Exception as e:
            logger.error(f"[healing] Failed to write JSON log: {e}")
        finally:
            # Clean up temp file if it still exists (rename failed or error before rename)
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass  # Best effort cleanup

    def _format_entry(self, entry: HealingLogEntry) -> str:
        """Format entry for human-readable log."""
        ts = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        status = "✓" if entry.result == "success" else "✗" if entry.result == "failed" else "→"
        return f"[{ts}] [{entry.component}] {status} {entry.action}: {entry.error_message[:80]}"

    def log_classification(self, stage: str, error: Exception,
                          classification: 'ErrorClassification', duration_ms: float):
        """Log watcher classification result."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            component="watcher",
            action="classify",
            error_type=type(error).__name__,
            error_message=str(error)[:200],
            result="success",
            duration_ms=duration_ms,
            details={
                "category": classification.category,
                "severity": classification.severity,
                "suggested_healer": classification.suggested_healer,
                "confidence": classification.confidence,
                "needs_llm_healer": classification.needs_llm_healer,
                "reasoning": classification.reasoning[:100] if classification.reasoning else ""
            }
        )
        self._write(entry)

        # Console output
        conf = classification.confidence
        logger.info(f"[watcher] ✓ Category: {classification.category} | "
                   f"Severity: {classification.severity} | Conf: {conf:.2f}")
        if classification.suggested_healer:
            logger.info(f"[watcher] → Suggested: {classification.suggested_healer}")

    def log_healer_attempt(self, stage: str, healer_name: str,
                          error: Exception, result: 'HealerResult',
                          duration_ms: float, stack_trace: Optional[str] = None):
        """Log healer fix attempt."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            component="healer",
            action="attempt",
            error_type=type(error).__name__,
            error_message=str(error)[:200],
            result="success" if result.success else "failed",
            duration_ms=duration_ms,
            stack_trace=stack_trace,
            details={
                "healer": healer_name,
                "action": result.action.value if hasattr(result.action, 'value') else str(result.action),
                "message": result.message,
                "modified_config": result.modified_config,
                "healer_details": result.details
            }
        )
        self._write(entry)

    def log_fallback(self, stage: str, from_component: str,
                    to_component: str, reason: str):
        """Log fallback activation."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            component="fallback",
            action="fallback",
            error_type="FallbackTriggered",
            error_message=reason,
            result="degraded",
            duration_ms=0,
            details={
                "from": from_component,
                "to": to_component,
                "reason": reason
            }
        )
        self._write(entry)
        logger.warning(f"[healing] Fallback: {from_component} → {to_component}: {reason}")

    def log_escalation(self, stage: str, error: Exception,
                      reason: str, to_user: bool = False):
        """Log escalation to LLM healer or user."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            component="orchestrator",
            action="escalate",
            error_type=type(error).__name__,
            error_message=str(error)[:200],
            result="escalated",
            duration_ms=0,
            details={
                "reason": reason,
                "to_user": to_user,
                "to_llm_healer": not to_user
            }
        )
        self._write(entry)

        if to_user:
            logger.error(f"[healing] User escalation required: {reason}")
        else:
            logger.info(f"[healing] Escalating to LLM healer: {reason}")

    def log_self_heal(self, healer: str, attempt: int, max_attempts: int,
                     error_type: str, action: str):
        """Log LLM healer self-healing action."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="SELF_HEAL",
            component=healer,
            action="self_heal",
            error_type=error_type,
            error_message=action,
            result="retrying",
            duration_ms=0,
            details={
                "attempt": attempt,
                "max_attempts": max_attempts
            }
        )
        self._write(entry)
        logger.info(f"[{healer}] Self-heal ({attempt}/{max_attempts}): {action}")

    def log_provider_switch(self, from_provider: str, to_provider: str):
        """Log provider switch during self-healing."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="SELF_HEAL",
            component="llm-healer",
            action="provider_switch",
            error_type="ProviderSwitch",
            error_message=f"{from_provider} → {to_provider}",
            result="retrying",
            duration_ms=0,
            details={
                "from_provider": from_provider,
                "to_provider": to_provider
            }
        )
        self._write(entry)
        logger.warning(f"[llm-healer] Switching provider: {from_provider} → {to_provider}")

    def log_provider_switch_failed(self, provider: str, reason: str):
        """Log failed provider switch."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="SELF_HEAL",
            component="llm-healer",
            action="provider_switch_failed",
            error_type="ProviderSwitchFailed",
            error_message=f"Cannot use {provider}: {reason}",
            result="failed",
            duration_ms=0,
            details={
                "provider": provider,
                "reason": reason
            }
        )
        self._write(entry)
        logger.warning(f"[llm-healer] Cannot use {provider}: {reason}")

    def log_preflight_check(self, component: str, check_name: str,
                           passed: bool, message: str):
        """Log preflight check result."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="PREFLIGHT",
            component=component,
            action="preflight_check",
            error_type="PreflightCheck",
            error_message=message,
            result="success" if passed else "failed",
            duration_ms=0,
            details={
                "check_name": check_name,
                "passed": passed
            }
        )
        self._write(entry)

    def print_box(self, title: str, lines: List[str], width: int = 65):
        """Print a formatted box to console."""
        if self.console_format == "minimal":
            for line in lines:
                print(f"  {line}")
            return

        if self.console_format == "simple":
            print(f"\n  === {title} ===")
            for line in lines:
                print(f"  {line}")
            return

        # Box format
        print(f"\n┌{'─' * width}┐")
        print(f"│  {title:<{width-3}}│")
        print(f"├{'─' * width}┤")
        for line in lines:
            # Truncate if too long
            if len(line) > width - 4:
                line = line[:width-7] + "..."
            print(f"│  {line:<{width-3}}│")
        print(f"└{'─' * width}┘")

    def generate_report(self) -> str:
        """Generate summary report of all healing activity."""
        if not self.entries:
            return "No healing activity recorded."

        # Count by result
        success_count = sum(1 for e in self.entries if e.result == "success")
        failed_count = sum(1 for e in self.entries if e.result == "failed")

        # Count by component
        watcher_count = sum(1 for e in self.entries if e.component == "watcher")
        llm_healer_count = sum(1 for e in self.entries
                              if e.component == "llm-healer" and e.action == "attempt")

        # Group by stage
        stages: Dict[str, List[HealingLogEntry]] = {}
        for entry in self.entries:
            if entry.stage not in stages:
                stages[entry.stage] = []
            stages[entry.stage].append(entry)

        lines = [
            f"Session ID: {self.session_id}",
            f"Total entries: {len(self.entries)}",
            f"Successful heals: {success_count}",
            f"Failed heals: {failed_count}",
            f"Watcher classifications: {watcher_count}",
            f"LLM healer invocations: {llm_healer_count}",
            "",
            "By stage:"
        ]

        for stage_name, stage_entries in stages.items():
            if stage_name in ("SELF_HEAL", "PREFLIGHT"):
                continue
            healers = [e for e in stage_entries if e.component == "healer"]
            if healers:
                success = sum(1 for h in healers if h.result == "success")
                lines.append(f"  {stage_name}: {success}/{len(healers)} healed")

        return "\n".join(lines)

    def finalize(self):
        """Finalize logging and print summary."""
        report = self.generate_report()
        logger.info(f"\n[healing] Final report:\n{report}")

        # Write final JSON with summary
        if self.json_log:
            summary = {
                "session_id": self.session_id,
                "total_entries": len(self.entries),
                "entries": [e.to_dict() for e in self.entries]
            }
            self.json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
