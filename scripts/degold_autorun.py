#!/usr/bin/env python3
"""Hourly Degold autorun loop for Trello-assigned pipeline work."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from script_utils import run_subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_SCRIPT = PROJECT_ROOT / "scripts" / "pipeline_queue_state.py"
DEFAULT_QUEUE_STATE_FILE = PROJECT_ROOT / "Degold" / "pipeline_queue_state.json"
DEFAULT_AUTORUN_STATE_FILE = PROJECT_ROOT / "Degold" / "degold_autorun_state.json"
DEFAULT_LOCK_FILE = PROJECT_ROOT / "Degold" / "degold_autorun.lock"
DEFAULT_STOP_FILE = PROJECT_ROOT / "Degold" / "degold_autorun.stop"
QUEUE_STOP_FILE = PROJECT_ROOT / "Degold" / "queue_stop.txt"
FORCE_CYCLE_FILE = PROJECT_ROOT / "Degold" / "degold_autorun.force_cycle"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "degold_autorun.log"
AUTORUN_STATE_SCHEMA_VERSION = "1.4.0"
DEFAULT_FAILURE_RETRY_MINUTES = 5.0
RECENT_CYCLE_HISTORY_LIMIT = 24
DEFAULT_LOG_ROTATE_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5
STATE_WRITE_RETRY_DELAYS_SECONDS = (0.15, 0.35, 0.75, 1.5, 3.0)
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_HEARTBEAT_STALE_SECONDS = 90.0
HEARTBEAT_THREAD_JOIN_SECONDS = 5.0
STARTUP_FAILURE_CONTINUE_SECONDS = 5.0
ACTIONABLE_QUEUE_FIELDS = (
    "pending_not_started_card_ids",
    "ready_card_ids",
    "blocked_card_ids",
)
QUEUE_COUNT_FIELDS = (
    "pending_not_started",
    "ready",
    "blocked",
    "completed",
    "not_required",
    "running",
    "pipeline_actionable",
    "needs_submission_workflow",
)
LIPSYNC_NONBLOCKING_STATUSES = {
    "lipsync_not_submitted",
    "lipsync_not_downloaded",
}
OPTIONAL_STEP_NAMES = {"discord-prepare"}
RETRYABLE_STEP_NAMES = {"discord-prepare"}
TRANSIENT_FAILURE_HINTS = (
    "timed out",
    "timeout",
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)
STEP_RETRY_ATTEMPTS = 2
STEP_RETRY_BACKOFF_SECONDS = 5.0
STOP_FILE_POLL_SECONDS = 5.0
JSON_LOAD_RETRY_ATTEMPTS = 3
JSON_LOAD_RETRY_DELAY_SECONDS = 0.2
STARTED_CARD_RE = re.compile(r"Started\s+([A-Za-z0-9_-]+)\s+\(")
RUNTIME_STATE_FIELDS = ("runtime", "last_runtime")
UNSET = object()
_ACTIVE_AUTORUN_LEASE: "AutorunLease | None" = None
_GLOBAL_STOP_EVENT: threading.Event | None = None


def _create_signal_handler(stop_event: threading.Event) -> Callable[[int, None], None]:
    """Create a signal handler that sets the stop event."""
    def handler(signum: int, frame: Any) -> None:
        stop_event.set()
    return handler


def _setup_signal_handlers(stop_event: threading.Event) -> None:
    """Register SIGINT and SIGTERM handlers for graceful shutdown."""
    if sys.platform == "win32":
        # On Windows, use a simple approach - just handle SIGINT
        # CTRL_C_EVENT handling is unreliable, so we rely on the stop file instead
        return

    signal.signal(signal.SIGINT, _create_signal_handler(stop_event))
    signal.signal(signal.SIGTERM, _create_signal_handler(stop_event))

# Structured error codes for cycle summaries
ERR_NONE = "none"
ERR_QUEUE_STATE_LOAD = "queue_state_load"
ERR_QUEUE_PREPARE = "queue_prepare"
ERR_QUEUE_ARCHIVE = "queue_archive"
ERR_QUEUE_DISCORD = "queue_discord"
ERR_CARD_VALIDATION = "card_validation"
ERR_STEP_TIMEOUT = "step_timeout"
ERR_STEP_TRANSIENT = "step_transient"
ERR_LAUNCH_EXTRACT = "launch_extract"
ERR_UNEXPECTED = "unexpected"
ERROR_CODES = (
    ERR_NONE,
    ERR_QUEUE_STATE_LOAD,
    ERR_QUEUE_PREPARE,
    ERR_QUEUE_ARCHIVE,
    ERR_QUEUE_DISCORD,
    ERR_CARD_VALIDATION,
    ERR_STEP_TIMEOUT,
    ERR_STEP_TRANSIENT,
    ERR_LAUNCH_EXTRACT,
    ERR_UNEXPECTED,
)


@dataclass(frozen=True)
class AutorunConfig:
    """Runtime configuration for the autorun loop."""

    queue_script: Path
    queue_state_file: Path
    autorun_state_file: Path
    lock_file: Path
    stop_file: Path
    log_file: Path
    python_executable: str
    interval_minutes: float
    failure_retry_minutes: float
    once: bool
    refresh_lipsync: bool
    command_timeout_seconds: int
    card_ids: tuple[str, ...]
    stop_when_idle: bool
    skip_discord_prepare: bool
    verify: bool
    verify_threshold: float
    auto_watch: bool
    clear_suppressed: bool
    validate_card_ids: bool
    status: bool = False
    dry_run: bool = False
    list_suppressed: bool = False
    verbose: bool = False
    pipeline_timeout_minutes: int = 480
    auto_lipsync: bool = False
    webhook_url: str | None = None


def now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path, default: Any) -> Any:
    """Load JSON if present, otherwise return default."""
    if not path.exists():
        return default

    for attempt in range(1, JSON_LOAD_RETRY_ATTEMPTS + 1):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            if attempt >= JSON_LOAD_RETRY_ATTEMPTS or not path.exists():
                append_log_line(
                    path.parent / "degold_autorun.log",
                    f"[{now_iso()}] JSON parse error in {path}: {exc} (attempt {attempt}/{JSON_LOAD_RETRY_ATTEMPTS})"
                )
                return default
            time.sleep(JSON_LOAD_RETRY_DELAY_SECONDS)
        except OSError:
            if attempt >= JSON_LOAD_RETRY_ATTEMPTS or not path.exists():
                return default
            time.sleep(JSON_LOAD_RETRY_DELAY_SECONDS)

    return default


def write_json_file(path: Path, payload: Any) -> None:
    """Persist JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    fallback_text = ""
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        fallback_text = temp_path.read_text(encoding="utf-8", errors="replace")
        for attempt_index, delay_seconds in enumerate((0.0, *STATE_WRITE_RETRY_DELAYS_SECONDS), start=1):
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            try:
                os.replace(temp_path, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32}:
                    raise
                if attempt_index >= len(STATE_WRITE_RETRY_DELAYS_SECONDS) + 1:
                    with open(path, "w", encoding="utf-8", errors="replace") as handle:
                        handle.write(fallback_text)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def parse_iso_datetime(value: Any) -> datetime | None:
    """Parse an ISO8601 timestamp into an aware datetime."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def heartbeat_age_seconds(value: Any) -> float | None:
    """Return heartbeat age in seconds, or None when unavailable."""
    timestamp = parse_iso_datetime(value)
    if timestamp is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


def is_stale_heartbeat(value: Any, *, stale_after_seconds: float = DEFAULT_HEARTBEAT_STALE_SECONDS) -> bool:
    """Return True when a heartbeat is absent or older than the stale threshold."""
    age_seconds = heartbeat_age_seconds(value)
    if age_seconds is None:
        return True
    return age_seconds > max(1.0, float(stale_after_seconds))


def is_process_running(pid: Any) -> bool:
    """Return True when the pid still points to a live process."""
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if normalized_pid <= 0:
        return False
    if normalized_pid == os.getpid():
        return True

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            access = 0x00100000 | 0x1000  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
            handle = kernel32.OpenProcess(access, False, normalized_pid)
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 0x00000102  # WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception as exc:
            # Log at debug level - process check failed but not critical
            return False

    try:
        os.kill(normalized_pid, 0)
    except OSError:
        return False
    return True


def build_runtime_payload(
    *,
    instance_id: str,
    pid: int,
    started_at: str,
    heartbeat_at: str,
    status: str,
    current_step: str,
    cycle_started_at: str | None,
    target_card_ids: Sequence[str],
    lock_file: Path | str | None = None,
    takeover_reason: str = "",
) -> dict[str, Any]:
    """Build the serialized autorun runtime lease payload."""
    payload = {
        "instance_id": str(instance_id).strip(),
        "pid": int(pid),
        "started_at": str(started_at or "").strip(),
        "heartbeat_at": str(heartbeat_at or "").strip(),
        "status": str(status or "").strip() or "idle",
        "current_step": str(current_step or "").strip(),
        "cycle_started_at": str(cycle_started_at or "").strip() or None,
        "target_card_ids": normalize_card_ids(target_card_ids),
        "lock_file": str(lock_file or DEFAULT_LOCK_FILE),
        "heartbeat_interval_seconds": float(DEFAULT_HEARTBEAT_INTERVAL_SECONDS),
        "heartbeat_stale_seconds": float(DEFAULT_HEARTBEAT_STALE_SECONDS),
    }
    if str(takeover_reason or "").strip():
        payload["takeover_reason"] = str(takeover_reason).strip()
    return payload


def preserve_runtime_state(source_state: dict[str, Any], target_state: dict[str, Any]) -> dict[str, Any]:
    """Preserve runtime metadata when cycle summaries rewrite the state file."""
    for field in RUNTIME_STATE_FIELDS:
        value = source_state.get(field)
        if isinstance(value, dict) and value:
            target_state[field] = value
    return target_state


def persist_runtime_state(config: AutorunConfig, payload: dict[str, Any]) -> None:
    """Persist the live autorun runtime snapshot for diagnostics."""
    state = load_json_file(config.autorun_state_file, {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("schema_version", AUTORUN_STATE_SCHEMA_VERSION)
    runtime_payload = dict(payload)
    runtime_payload["lock_file"] = str(config.lock_file)
    state["runtime"] = runtime_payload
    write_json_file(config.autorun_state_file, state)


def clear_runtime_state(config: AutorunConfig, instance_id: str, *, exit_reason: str) -> None:
    """Move the live runtime snapshot to last_runtime on clean lease release."""
    state = load_json_file(config.autorun_state_file, {})
    if not isinstance(state, dict):
        state = {}
    runtime = state.get("runtime")
    if not isinstance(runtime, dict):
        return
    runtime_instance_id = str(runtime.get("instance_id") or "").strip()
    if runtime_instance_id != str(instance_id).strip():
        return
    stopped_payload = dict(runtime)
    stopped_payload["status"] = "stopped"
    stopped_payload["heartbeat_at"] = now_iso()
    stopped_payload["exited_at"] = now_iso()
    stopped_payload["exit_reason"] = str(exit_reason or "").strip() or "unknown"
    state.pop("runtime", None)
    state["last_runtime"] = stopped_payload
    write_json_file(config.autorun_state_file, state)


def try_create_lock_file(path: Path, payload: dict[str, Any]) -> bool:
    """Create the single-instance lock atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= int(getattr(os, "O_BINARY", 0) or 0)
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return True


def remove_lock_file_if_owned(path: Path, instance_id: str) -> bool:
    """Remove the autorun lock only when the current process still owns it."""
    payload = load_json_file(path, {})
    if not isinstance(payload, dict):
        return False
    owner_instance_id = str(payload.get("instance_id") or "").strip()
    if owner_instance_id != str(instance_id).strip():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def stale_lock_reason(payload: dict[str, Any]) -> str | None:
    """Return takeover reason when an existing autorun lock is stale."""
    if not isinstance(payload, dict) or not payload:
        return "invalid-lock-payload"
    pid = int(payload.get("pid", 0) or 0)
    heartbeat_at = payload.get("heartbeat_at")
    if pid <= 0:
        return "missing-pid"
    if not is_process_running(pid):
        return f"dead-owner pid={pid}"
    if is_stale_heartbeat(heartbeat_at):
        age_seconds = heartbeat_age_seconds(heartbeat_at)
        if age_seconds is None:
            return f"stale-heartbeat pid={pid}"
        return f"stale-heartbeat pid={pid} age_seconds={int(age_seconds)}"
    return None


@dataclass
class AutorunLease:
    """Live autorun single-instance lease with background heartbeats."""

    config: AutorunConfig
    instance_id: str
    started_at: str
    pid: int
    takeover_reason: str = ""
    status: str = "starting"
    current_step: str = ""
    cycle_started_at: str | None = None
    target_card_ids: tuple[str, ...] = ()
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def snapshot(self, *, heartbeat_at: str | None = None) -> dict[str, Any]:
        """Serialize the current lease state."""
        with self._state_lock:
            return build_runtime_payload(
                instance_id=self.instance_id,
                pid=self.pid,
                started_at=self.started_at,
                heartbeat_at=heartbeat_at or now_iso(),
                status=self.status,
                current_step=self.current_step,
                cycle_started_at=self.cycle_started_at,
                target_card_ids=self.target_card_ids,
                lock_file=self.config.lock_file,
                takeover_reason=self.takeover_reason,
            )

    def refresh(
        self,
        *,
        status: str | None = None,
        current_step: str | None = None,
        cycle_started_at: str | None | object = UNSET,
        target_card_ids: Sequence[str] | object = UNSET,
        publish_state: bool = True,
    ) -> dict[str, Any]:
        """Update the lease state and publish it."""
        with self._state_lock:
            if status is not None:
                self.status = str(status or "").strip() or "idle"
            if current_step is not None:
                self.current_step = str(current_step or "").strip()
            if cycle_started_at is not UNSET:
                self.cycle_started_at = str(cycle_started_at or "").strip() or None
            if target_card_ids is not UNSET:
                self.target_card_ids = tuple(normalize_card_ids(target_card_ids or []))
            payload = build_runtime_payload(
                instance_id=self.instance_id,
                pid=self.pid,
                started_at=self.started_at,
                heartbeat_at=now_iso(),
                status=self.status,
                current_step=self.current_step,
                cycle_started_at=self.cycle_started_at,
                target_card_ids=self.target_card_ids,
                lock_file=self.config.lock_file,
                takeover_reason=self.takeover_reason,
            )
        write_json_file(self.config.lock_file, payload)
        if publish_state:
            persist_runtime_state(self.config, payload)
        return payload

    def _heartbeat_loop(self) -> None:
        """Refresh the lock file periodically while the autorun process is alive."""
        current_interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        idle_intervals = (60.0, 15.0)  # (idle, active)

        while not self._stop_event.wait(current_interval):
            # Adaptive heartbeat interval based on status
            try:
                status = self.status
                # Longer interval when idle/sleeping, shorter when running
                if status in ("idle", "sleeping", "sleeping_after_failure"):
                    current_interval = idle_intervals[0]
                else:
                    current_interval = idle_intervals[1]
                self.refresh(publish_state=False)
            except Exception as exc:
                append_log_line(
                    self.config.log_file,
                    f"[{now_iso()}] autorun-heartbeat-error error={str(exc).strip() or exc.__class__.__name__}",
                )

    def start(self) -> None:
        """Publish the initial lease and start the background heartbeat thread."""
        self.refresh(status="starting", current_step="", cycle_started_at=None, target_card_ids=())
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="degold-autorun-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def close(self, *, exit_reason: str) -> None:
        """Stop the heartbeat thread, persist final runtime state, and release the lock."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_THREAD_JOIN_SECONDS)
        payload = self.refresh(
            status="stopped",
            current_step="",
            cycle_started_at=None,
            target_card_ids=(),
            publish_state=False,
        )
        clear_runtime_state(self.config, self.instance_id, exit_reason=exit_reason)
        remove_lock_file_if_owned(self.config.lock_file, self.instance_id)
        append_log_line(
            self.config.log_file,
            (
                f"[{now_iso()}] autorun-lease-released "
                f"instance_id={self.instance_id} pid={self.pid} exit_reason={str(exit_reason or '').strip() or 'unknown'}"
            ),
        )


def update_active_autorun_lease(
    *,
    status: str | None = None,
    current_step: str | None = None,
    cycle_started_at: str | None | object = UNSET,
    target_card_ids: Sequence[str] | object = UNSET,
) -> None:
    """Publish state changes for the active autorun lease when one is running."""
    lease = _ACTIVE_AUTORUN_LEASE
    if lease is None:
        return
    lease.refresh(
        status=status,
        current_step=current_step,
        cycle_started_at=cycle_started_at,
        target_card_ids=target_card_ids,
    )


def acquire_autorun_lease(config: AutorunConfig) -> AutorunLease | None:
    """Acquire the autorun single-instance lease or return None when already owned."""
    instance_id = uuid.uuid4().hex[:12]
    pid = os.getpid()
    started_at = now_iso()
    takeover_reason = ""
    while True:
        payload = build_runtime_payload(
            instance_id=instance_id,
            pid=pid,
            started_at=started_at,
            heartbeat_at=started_at,
            status="starting",
            current_step="",
            cycle_started_at=None,
            target_card_ids=(),
            lock_file=config.lock_file,
            takeover_reason=takeover_reason,
        )
        if try_create_lock_file(config.lock_file, payload):
            lease = AutorunLease(
                config=config,
                instance_id=instance_id,
                started_at=started_at,
                pid=pid,
                takeover_reason=takeover_reason,
            )
            lease.start()
            append_log_line(
                config.log_file,
                (
                    f"[{now_iso()}] autorun-lease-acquired "
                    f"instance_id={instance_id} pid={pid}"
                    + (f" takeover_reason={takeover_reason}" if takeover_reason else "")
                ),
            )
            return lease

        existing_payload = load_json_file(config.lock_file, {})
        if not isinstance(existing_payload, dict):
            existing_payload = {}
        takeover_reason = stale_lock_reason(existing_payload) or ""
        if takeover_reason:
            try:
                config.lock_file.unlink()
            except OSError:
                existing_owner = str((existing_payload or {}).get("instance_id") or "").strip() or "unknown"
                append_log_line(
                    config.log_file,
                    (
                        f"[{now_iso()}] autorun-lease-takeover-blocked "
                        f"existing_instance_id={existing_owner} reason={takeover_reason}"
                    ),
                )
                return None
            append_log_line(
                config.log_file,
                (
                    f"[{now_iso()}] autorun-lease-takeover "
                    f"existing_instance_id={str((existing_payload or {}).get('instance_id') or '').strip() or 'unknown'} "
                    f"reason={takeover_reason}"
                ),
            )
            continue

        existing_pid = int((existing_payload or {}).get("pid", 0) or 0)
        existing_status = str((existing_payload or {}).get("status") or "").strip() or "unknown"
        heartbeat_at = str((existing_payload or {}).get("heartbeat_at") or "").strip() or "-"
        message = (
            f"Another autorun instance is active (pid={existing_pid}, status={existing_status}, "
            f"heartbeat_at={heartbeat_at})."
        )
        append_log_line(
            config.log_file,
            (
                f"[{now_iso()}] autorun-instance-active "
                f"existing_instance_id={str((existing_payload or {}).get('instance_id') or '').strip() or 'unknown'} "
                f"pid={existing_pid} status={existing_status} heartbeat_at={heartbeat_at}"
            ),
        )
        print(message)
        return None


# Global counter for log rotation optimization
_log_write_count = 0
LOG_ROTATE_CHECK_INTERVAL = 100


def append_log_line(path: Path, message: str) -> None:
    """Append a timestamped log line."""
    global _log_write_count
    path.parent.mkdir(parents=True, exist_ok=True)

    # Optimize: only check rotation every N writes
    _log_write_count += 1
    if _log_write_count % LOG_ROTATE_CHECK_INTERVAL == 0:
        rotate_log_file(path)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip())
        handle.write("\n")


def append_log_block(path: Path, lines: Sequence[str]) -> None:
    """Append a grouped multi-line log entry with a trailing separator line."""
    global _log_write_count
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Optimize: only check rotation every N writes
    _log_write_count += 1
    if _log_write_count % LOG_ROTATE_CHECK_INTERVAL == 0:
        rotate_log_file(path)

    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).rstrip())
            handle.write("\n")
        handle.write("\n")


def rotate_log_file(
    path: Path,
    *,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Rotate the autorun log when it grows beyond the configured cap."""
    max_bytes = DEFAULT_LOG_ROTATE_MAX_BYTES if max_bytes is None else int(max_bytes)
    backup_count = DEFAULT_LOG_BACKUP_COUNT if backup_count is None else int(backup_count)
    if max_bytes <= 0 or backup_count <= 0 or not path.exists():
        return

    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return

    oldest_backup = path.with_name(f"{path.name}.{backup_count}")
    if oldest_backup.exists():
        try:
            oldest_backup.unlink()
        except OSError:
            pass

    for index in range(backup_count - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        destination = path.with_name(f"{path.name}.{index + 1}")
        if source.exists():
            os.replace(source, destination)

    os.replace(path, path.with_name(f"{path.name}.1"))


def normalize_card_ids(values: Sequence[Any]) -> list[str]:
    """Normalize card IDs to sorted unique strings."""
    return sorted({str(value).strip() for value in values if str(value).strip()})


def truncate_history(entries: Sequence[dict[str, Any]], limit: int = RECENT_CYCLE_HISTORY_LIMIT) -> list[dict[str, Any]]:
    """Keep only the most recent bounded history entries."""
    if limit <= 0:
        return []
    normalized_entries = [entry for entry in entries if isinstance(entry, dict)]
    return normalized_entries[-limit:]


def build_queue_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the queue details the autorun loop needs."""
    pipelines = state.get("pipelines") or {}
    queue = state.get("queue") or {}
    summary = state.get("summary") or {}
    runtime_summary = state.get("runtime_summary") or {}
    submission_queue = state.get("submission_queue") or {}

    assigned_card_ids: list[str] = []
    lipsync_attention_card_ids: list[str] = []
    ready_without_lipsync_card_ids: list[str] = []
    for key, entry in pipelines.items():
        pipeline = entry or {}
        card_id = str(pipeline.get("card_id") or key).strip()
        if card_id:
            assigned_card_ids.append(card_id)
        checks = pipeline.get("start_checks") or {}
        nonblocking_statuses = {
            str(status).strip()
            for status in (checks.get("nonblocking_statuses") or [])
            if str(status).strip()
        }
        if nonblocking_statuses & LIPSYNC_NONBLOCKING_STATUSES:
            lipsync_attention_card_ids.append(card_id)
            pipeline_state = str(
                pipeline.get("pipeline_run_state")
                or pipeline.get("pipeline_state")
                or ""
            ).strip()
            if pipeline_state == "ready":
                ready_without_lipsync_card_ids.append(card_id)

    actionable_card_ids: list[str] = []
    for field in ACTIONABLE_QUEUE_FIELDS:
        actionable_card_ids.extend(queue.get(field, []) or [])

    return {
        "generated_at": state.get("generated_at"),
        "assigned_card_ids": normalize_card_ids(assigned_card_ids),
        "actionable_card_ids": normalize_card_ids(actionable_card_ids),
        "running_card_ids": normalize_card_ids(runtime_summary.get("running_card_ids", []) or []),
        "ready_card_ids": normalize_card_ids(queue.get("ready_card_ids", []) or []),
        "submission_pending_card_ids": normalize_card_ids(
            submission_queue.get("needs_submission_card_ids", []) or []
        ),
        "lipsync_attention_card_ids": normalize_card_ids(lipsync_attention_card_ids),
        "ready_without_lipsync_card_ids": normalize_card_ids(ready_without_lipsync_card_ids),
        "queue_counts": {
            field: int(summary.get(field, 0) or 0)
            for field in QUEUE_COUNT_FIELDS
        },
    }


def diff_new_card_ids(current_ids: Sequence[Any], previous_ids: Sequence[Any]) -> list[str]:
    """Return sorted IDs present now that were not seen in the previous cycle."""
    previous = {str(value).strip() for value in previous_ids if str(value).strip()}
    return sorted({str(value).strip() for value in current_ids if str(value).strip()} - previous)


def resolve_suppressed_card_ids(
    values: Sequence[Any],
    actionable_ids: Sequence[Any],
) -> list[str]:
    """Normalize suppressed IDs and keep only those still actionable."""
    actionable = {str(value).strip().lower() for value in actionable_ids if str(value).strip()}
    return [
        card_id
        for card_id in normalize_card_ids(values)
        if str(card_id).strip().lower() in actionable
    ]


def clear_suppressed_cards(config: AutorunConfig) -> bool:
    """Clear the suppressed card list at startup if --clear-suppressed is set."""
    if not config.clear_suppressed:
        return False
    state = load_json_file(config.autorun_state_file, {})
    if not isinstance(state, dict):
        state = {}
    current_suppressed = state.get("suppressed_card_ids", []) or []
    if not current_suppressed:
        return False
    state["suppressed_card_ids"] = []
    write_json_file(config.autorun_state_file, state)
    append_log_line(
        config.log_file,
        f"[{now_iso()}] suppressed-cards-cleared count={len(current_suppressed)}",
    )
    return True


def validate_card_ids(config: AutorunConfig, actionable_ids: list[str]) -> tuple[bool, list[str]]:
    """Warn if provided card IDs don't match the actionable queue."""
    if not config.card_ids or not config.validate_card_ids:
        return True, []
    actionable_lookup = {card_id.lower() for card_id in actionable_ids}
    missing_ids = [
        card_id for card_id in config.card_ids
        if card_id.lower() not in actionable_lookup
    ]
    return len(missing_ids) == 0, missing_ids


def should_apply_suppression(config: AutorunConfig) -> bool:
    """Allow explicit one-shot retries to bypass startup-failure suppression."""
    return not (config.once and bool(config.card_ids))


def resolve_cycle_target_card_ids(
    config: AutorunConfig,
    before_snapshot: dict[str, Any],
    previous_state: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Resolve the card IDs eligible for this autorun cycle."""
    actionable_ids = normalize_card_ids(before_snapshot.get("actionable_card_ids", []) or [])
    actionable_lookup = {card_id.lower() for card_id in actionable_ids}
    if config.card_ids:
        candidate_ids = [
            card_id
            for card_id in normalize_card_ids(config.card_ids)
            if card_id.lower() in actionable_lookup
        ]
    else:
        candidate_ids = list(actionable_ids)

    suppressed_ids = resolve_suppressed_card_ids(
        previous_state.get("suppressed_card_ids", []) or [],
        actionable_ids,
    )
    if should_apply_suppression(config) and suppressed_ids:
        suppressed_lookup = {card_id.lower() for card_id in suppressed_ids}
        unsuppressed_candidate_ids = [
            card_id for card_id in candidate_ids
            if card_id.lower() not in suppressed_lookup
        ]
        if unsuppressed_candidate_ids:
            candidate_ids = unsuppressed_candidate_ids
        else:
            candidate_ids = [
                card_id for card_id in candidate_ids
                if card_id.lower() in suppressed_lookup
            ]
    return candidate_ids, suppressed_ids


def remaining_unsuppressed_actionable_ids(summary: dict[str, Any]) -> list[str]:
    """Return actionable/running card IDs remaining after suppression for the cycle."""
    suppressed_ids = {
        str(card_id).strip().lower()
        for card_id in (summary.get("suppressed_card_ids", []) or [])
        if str(card_id).strip()
    }
    remaining_ids: list[str] = []
    for raw_card_id in list(summary.get("actionable_card_ids", []) or []) + list(summary.get("running_after_card_ids", []) or []):
        card_id = str(raw_card_id).strip()
        if not card_id:
            continue
        if card_id.lower() in suppressed_ids:
            continue
        remaining_ids.append(card_id)
    return normalize_card_ids(remaining_ids)


def remaining_suppressed_actionable_ids(summary: dict[str, Any]) -> list[str]:
    """Return suppressed actionable card IDs that are eligible for a retry cycle."""
    suppressed_ids = {
        str(card_id).strip().lower()
        for card_id in (summary.get("suppressed_card_ids", []) or [])
        if str(card_id).strip()
    }
    return normalize_card_ids(
        [
            str(card_id).strip()
            for card_id in (summary.get("actionable_card_ids", []) or [])
            if str(card_id).strip() and str(card_id).strip().lower() in suppressed_ids
        ]
    )


def tail_text(text: str, max_lines: int = 12) -> str:
    """Return the last few lines of command output."""
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def coerce_text(value: Any) -> str:
    """Convert subprocess outputs into safe text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def build_failed_process(
    command: Sequence[str],
    *,
    returncode: int,
    stdout: Any = "",
    stderr: Any = "",
) -> subprocess.CompletedProcess[str]:
    """Build a failed CompletedProcess for handled subprocess exceptions."""
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=int(returncode),
        stdout=coerce_text(stdout),
        stderr=coerce_text(stderr),
    )


def extract_launch_card_id(output: str) -> str | None:
    """Parse the launched card ID from prepare --run-ready output."""
    match = STARTED_CARD_RE.search(str(output or ""))
    if not match:
        return None
    return match.group(1)


def build_workflow_steps(
    config: AutorunConfig,
    target_card_ids: Sequence[str],
) -> list[tuple[str, list[str]]]:
    """Build the existing /pipeline-queue run workflow for one autorun cycle."""
    archive_args = ["--sync-first"]
    if config.refresh_lipsync:
        archive_args.append("--refresh-lipsync")
    prepare_args: list[str] = []
    if target_card_ids:
        prepare_args.append("--run-ready")
    for card_id in target_card_ids:
        prepare_args.extend(["--card-id", card_id])

    steps: list[tuple[str, list[str]]] = []
    if not config.skip_discord_prepare:
        steps.append(("discord-prepare", []))
    steps.append(("archive-completed", archive_args))
    steps.append(("prepare", prepare_args))
    return steps


def run_timing_verification(
    config: AutorunConfig,
    project_path: str,
) -> subprocess.CompletedProcess[str]:
    """Run timing verification for a completed pipeline project."""
    verify_script = PROJECT_ROOT / "scripts" / "regenerate_otio.py"
    command = [
        config.python_executable,
        str(verify_script),
        project_path,
        "--drift-threshold",
        str(config.verify_threshold),
    ]
    try:
        return run_subprocess(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return build_failed_process(
            command,
            returncode=124,
            stdout=coerce_text(exc.stdout),
            stderr=f"Verification timed out after 300s",
        )
    except OSError as exc:
        return build_failed_process(
            command,
            returncode=getattr(exc, "errno", 1) or 1,
            stderr=f"Failed to launch verification: {exc}",
        )


def spawn_watch_task(
    config: AutorunConfig,
    project_path: str,
) -> bool:
    """Spawn a background watch task for the active project."""
    watch_script = PROJECT_ROOT / "scripts" / "watch_auto.ps1"
    if not watch_script.exists():
        append_log_line(
            config.log_file,
            f"[{now_iso()}] watch-spawn-failed script-not-found path={watch_script}",
        )
        return False

    try:
        subprocess.Popen(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(watch_script), "-ProjectPath", project_path],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        append_log_line(
            config.log_file,
            f"[{now_iso()}] watch-spawned project_path={project_path}",
        )
        return True
    except Exception as exc:
        append_log_line(
            config.log_file,
            f"[{now_iso()}] watch-spawn-failed error={str(exc).strip() or exc.__class__.__name__}",
        )
        return False


def run_queue_command(
    python_executable: str,
    queue_script: Path,
    subcommand: str,
    extra_args: Sequence[str],
    *,
    state_file: Path | None = None,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run pipeline_queue_state.py with safe text decoding."""
    command = [python_executable, str(queue_script)]
    if state_file is not None:
        command.extend(["--state-file", str(state_file)])
    command.extend([subcommand, *extra_args])
    try:
        return run_subprocess(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_parts = [
            coerce_text(exc.stderr),
            f"Command timed out after {timeout_seconds}s",
        ]
        return build_failed_process(
            command,
            returncode=124,
            stdout=exc.stdout,
            stderr="\n".join(part for part in stderr_parts if part.strip()),
        )
    except OSError as exc:
        return build_failed_process(
            command,
            returncode=getattr(exc, "errno", 1) or 1,
            stderr=f"Failed to launch command: {exc}",
        )


def format_command_failure(result: subprocess.CompletedProcess[str], subcommand: str) -> str:
    """Build a compact error message from a failed subprocess."""
    stdout_tail = tail_text(result.stdout)
    stderr_tail = tail_text(result.stderr)
    parts = [f"{subcommand} exited with code {result.returncode}"]
    if stdout_tail:
        parts.append(f"stdout:\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"stderr:\n{stderr_tail}")
    return "\n".join(parts)


def is_transient_command_failure(result: subprocess.CompletedProcess[str]) -> bool:
    """Return True when a command failure looks transient and worth retrying."""
    if int(result.returncode) == 124:
        return True
    text = f"{result.stdout}\n{result.stderr}".lower()
    return any(hint in text for hint in TRANSIENT_FAILURE_HINTS)


def build_step_result(
    subcommand: str,
    args: Sequence[str],
    result: subprocess.CompletedProcess[str],
    *,
    status: str = "success",
    warning: str = "",
    retry_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Serialize one queue workflow step result into autorun state."""
    return {
        "name": subcommand,
        "status": status,
        "args": list(args),
        "retry_args": list(retry_args or []),
        "returncode": int(result.returncode),
        "stdout_tail": tail_text(result.stdout),
        "stderr_tail": tail_text(result.stderr),
        "warning": str(warning or "").strip(),
    }


def format_id_list(values: Sequence[Any]) -> str:
    """Format a card-id list for compact logs."""
    return ",".join(normalize_card_ids(values)) or "-"


def format_queue_counts(counts: dict[str, Any]) -> str:
    """Format queue counters in a stable field order."""
    return ",".join(f"{field}={int((counts or {}).get(field, 0) or 0)}" for field in QUEUE_COUNT_FIELDS)


def indent_log_text(value: str, *, prefix: str) -> list[str]:
    """Indent multi-line values for grouped log blocks."""
    lines = [line.rstrip() for line in str(value or "").splitlines() if line.strip()]
    return [f"{prefix}{line}" for line in lines]


def build_recent_cycle_entry(summary: dict[str, Any]) -> dict[str, Any]:
    """Persist a compact history entry for recent-cycle troubleshooting."""
    workflow_steps: list[dict[str, Any]] = []
    for step in summary.get("workflow_steps", []) or []:
        workflow_steps.append(
            {
                "name": str(step.get("name") or "").strip(),
                "status": str(step.get("status") or "").strip(),
                "returncode": int(step.get("returncode", 0) or 0),
                "warning": str(step.get("warning") or "").strip(),
            }
        )

    entry = {
        "status": str(summary.get("status") or "").strip(),
        "error_code": str(summary.get("error_code") or ERR_NONE).strip(),
        "started_at": summary.get("started_at"),
        "completed_at": summary.get("completed_at"),
        "duration_seconds": round(float(summary.get("duration_seconds", 0.0) or 0.0), 3),
        "target_card_ids": normalize_card_ids(summary.get("target_card_ids", []) or []),
        "suppressed_card_ids": normalize_card_ids(summary.get("suppressed_card_ids", []) or []),
        "new_assigned_card_ids": normalize_card_ids(summary.get("new_assigned_card_ids", []) or []),
        "new_actionable_card_ids": normalize_card_ids(summary.get("new_actionable_card_ids", []) or []),
        "launched_card_id": summary.get("launched_card_id"),
        "launch_outcome": str(summary.get("launch_outcome") or "").strip(),
        "launched_latest_log": str(summary.get("launched_latest_log") or "").strip(),
        "launched_project_dirs": normalize_card_ids(summary.get("launched_project_dirs", []) or []),
        "startup_failed_card_ids": normalize_card_ids(summary.get("startup_failed_card_ids", []) or []),
        "running_after_card_ids": normalize_card_ids(summary.get("running_after_card_ids", []) or []),
        "ready_after_card_ids": normalize_card_ids(summary.get("ready_after_card_ids", []) or []),
        "ready_without_lipsync_card_ids": normalize_card_ids(
            summary.get("ready_without_lipsync_card_ids", []) or []
        ),
        "submission_pending_card_ids": normalize_card_ids(summary.get("submission_pending_card_ids", []) or []),
        "warning_step_names": normalize_card_ids(summary.get("warning_step_names", []) or []),
        "workflow_steps": workflow_steps,
    }
    error_message = str(summary.get("error") or "").strip()
    if error_message:
        entry["error"] = error_message
    return entry


def build_cycle_log_lines(summary: dict[str, Any]) -> list[str]:
    """Build a detailed grouped log entry for one cycle."""
    lines = [format_cycle_log(summary)]
    lines.append(
        "  timing "
        f"started_at={summary.get('started_at') or '-'} "
        f"completed_at={summary.get('completed_at') or '-'} "
        f"duration_seconds={float(summary.get('duration_seconds', 0.0) or 0.0):.3f}"
    )
    lines.append(
        "  queue "
        f"generated_before={summary.get('queue_generated_at_before') or '-'} "
        f"generated_after={summary.get('queue_generated_at_after') or '-'}"
    )
    lines.append(f"  counts_before {format_queue_counts(summary.get('queue_counts_before') or {})}")
    lines.append(f"  counts_after  {format_queue_counts(summary.get('queue_counts_after') or {})}")
    lines.append(
        "  cards "
        f"targets={format_id_list(summary.get('target_card_ids', []) or [])} "
        f"suppressed={format_id_list(summary.get('suppressed_card_ids', []) or [])} "
        f"assigned={format_id_list(summary.get('assigned_card_ids', []) or [])} "
        f"actionable={format_id_list(summary.get('actionable_card_ids', []) or [])}"
    )
    lines.append(
        "  launch "
        f"card={summary.get('launched_card_id') or '-'} "
        f"outcome={summary.get('launch_outcome') or '-'} "
        f"startup_failed={format_id_list(summary.get('startup_failed_card_ids', []) or [])}"
    )
    launched_latest_log = str(summary.get("launched_latest_log") or "").strip()
    if launched_latest_log:
        lines.append(f"  launched_latest_log={launched_latest_log}")
    lines.append(
        "  lipsync "
        f"attention={format_id_list(summary.get('lipsync_attention_card_ids', []) or [])} "
        f"submission_pending={format_id_list(summary.get('submission_pending_card_ids', []) or [])}"
    )

    workflow_steps = summary.get("workflow_steps", []) or []
    if workflow_steps:
        lines.append("  workflow_steps")
        for step in workflow_steps:
            step_name = str(step.get("name") or "").strip() or "unknown"
            step_args = " ".join(str(arg) for arg in (step.get("args") or []))
            retry_args = " ".join(str(arg) for arg in (step.get("retry_args") or []))
            lines.append(
                f"    - name={step_name} "
                f"status={step.get('status') or '-'} "
                f"returncode={int(step.get('returncode', 0) or 0)} "
                f"args={step_args or '-'} "
                f"retry_args={retry_args or '-'}"
            )
            warning_text = str(step.get("warning") or "").strip()
            if warning_text:
                lines.append("      warning")
                lines.extend(indent_log_text(warning_text, prefix="        "))
            should_include_tails = warning_text or step_name == "prepare"
            stdout_tail = str(step.get("stdout_tail") or "").strip()
            stderr_tail = str(step.get("stderr_tail") or "").strip()
            if should_include_tails and stdout_tail:
                lines.append("      stdout_tail")
                lines.extend(indent_log_text(stdout_tail, prefix="        "))
            if should_include_tails and stderr_tail:
                lines.append("      stderr_tail")
                lines.extend(indent_log_text(stderr_tail, prefix="        "))

    return lines


def build_failure_log_lines(summary: dict[str, Any], consecutive_failures: int) -> list[str]:
    """Build a grouped failure log entry."""
    lines = [
        f"[{summary['completed_at']}] status=failure "
        f"started_at={summary['started_at']} "
        f"duration_seconds={float(summary.get('duration_seconds', 0.0) or 0.0):.3f} "
        f"consecutive_failures={consecutive_failures}"
    ]
    error_message = str(summary.get("error") or "").strip()
    if error_message:
        lines.append("  error")
        lines.extend(indent_log_text(error_message, prefix="    "))
    return lines


def run_workflow_step(
    config: AutorunConfig,
    subcommand: str,
    args: Sequence[str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run one /pipeline-queue step with autorunner-specific resilience."""
    result = run_queue_command(
        config.python_executable,
        config.queue_script,
        subcommand,
        args,
        state_file=config.queue_state_file,
        timeout_seconds=config.command_timeout_seconds,
    )
    if result.returncode == 0:
        return result, build_step_result(subcommand, args, result)

    if subcommand in RETRYABLE_STEP_NAMES and is_transient_command_failure(result):
        first_failure = format_command_failure(result, subcommand)
        for attempt in range(2, STEP_RETRY_ATTEMPTS + 1):
            time.sleep(STEP_RETRY_BACKOFF_SECONDS * (attempt - 1))
            retry_result = run_queue_command(
                config.python_executable,
                config.queue_script,
                subcommand,
                args,
                state_file=config.queue_state_file,
                timeout_seconds=config.command_timeout_seconds,
            )
            if retry_result.returncode == 0:
                return retry_result, build_step_result(
                    subcommand,
                    args,
                    retry_result,
                    status="success_after_retry",
                    warning=first_failure,
                )
            result = retry_result
            if not is_transient_command_failure(result):
                break

    if subcommand == "archive-completed" and "--refresh-lipsync" in args:
        retry_args = [arg for arg in args if arg != "--refresh-lipsync"]
        retry_result = run_queue_command(
            config.python_executable,
            config.queue_script,
            subcommand,
            retry_args,
            state_file=config.queue_state_file,
            timeout_seconds=config.command_timeout_seconds,
        )
        if retry_result.returncode == 0:
            return retry_result, build_step_result(
                subcommand,
                args,
                retry_result,
                status="success_with_refresh_fallback",
                warning=format_command_failure(result, f"{subcommand} (with --refresh-lipsync)"),
                retry_args=retry_args,
            )

    if subcommand in OPTIONAL_STEP_NAMES:
        return result, build_step_result(
            subcommand,
            args,
            result,
            status="warning",
            warning=format_command_failure(result, subcommand),
        )

    raise RuntimeError(format_command_failure(result, subcommand))


def is_usable_queue_snapshot(snapshot: dict[str, Any]) -> bool:
    """Return True when a queue snapshot contains enough signal to trust."""
    if str(snapshot.get("generated_at") or "").strip():
        return True
    if snapshot.get("assigned_card_ids") or snapshot.get("actionable_card_ids"):
        return True
    if snapshot.get("running_card_ids") or snapshot.get("ready_card_ids"):
        return True
    counts = snapshot.get("queue_counts") or {}
    return any(int(value or 0) > 0 for value in counts.values())


def recover_queue_snapshot(
    config: AutorunConfig,
    fallback_state: dict[str, Any],
    step_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ensure the autorunner has a usable queue snapshot after a workflow cycle."""
    after_state = load_json_file(config.queue_state_file, fallback_state)
    after_snapshot = build_queue_snapshot(after_state)
    if is_usable_queue_snapshot(after_snapshot):
        return after_state, after_snapshot

    recovery_result = run_queue_command(
        config.python_executable,
        config.queue_script,
        "sync",
        [],
        state_file=config.queue_state_file,
        timeout_seconds=config.command_timeout_seconds,
    )
    if recovery_result.returncode != 0:
        raise RuntimeError(
            "Queue state missing or unreadable after workflow\n"
            + format_command_failure(recovery_result, "sync (recovery)")
        )

    step_results.append(
        build_step_result(
            "sync-recovery",
            [],
            recovery_result,
            status="success_after_recovery",
            warning="Queue state was missing or unreadable after workflow; ran sync recovery",
        )
    )

    after_state = load_json_file(config.queue_state_file, after_state)
    after_snapshot = build_queue_snapshot(after_state)
    if is_usable_queue_snapshot(after_snapshot):
        return after_state, after_snapshot

    raise RuntimeError("Queue state missing or unreadable after sync recovery")


def format_cycle_log(summary: dict[str, Any]) -> str:
    """Format a single-line log message for one cycle."""
    return (
        f"[{summary['completed_at']}] status={summary['status']} "
        f"error_code={summary.get('error_code') or ERR_NONE} "
        f"duration_seconds={float(summary.get('duration_seconds', 0.0) or 0.0):.3f} "
        f"targets={','.join(summary.get('target_card_ids', [])) or '-'} "
        f"suppressed={','.join(summary.get('suppressed_card_ids', [])) or '-'} "
        f"new_assigned={','.join(summary['new_assigned_card_ids']) or '-'} "
        f"new_actionable={','.join(summary['new_actionable_card_ids']) or '-'} "
        f"launched={summary.get('launched_card_id') or '-'} "
        f"launch_outcome={summary.get('launch_outcome') or '-'} "
        f"startup_failed={','.join(summary.get('startup_failed_card_ids', [])) or '-'} "
        f"running_before={','.join(summary['running_before_card_ids']) or '-'} "
        f"running_after={','.join(summary['running_after_card_ids']) or '-'} "
        f"ready_after={','.join(summary.get('ready_after_card_ids', [])) or '-'} "
        f"ready_wo_lipsync={','.join(summary.get('ready_without_lipsync_card_ids', [])) or '-'} "
        f"submission_pending={','.join(summary.get('submission_pending_card_ids', [])) or '-'} "
        f"warnings={','.join(summary.get('warning_step_names', [])) or '-'}"
    )


def record_cycle_failure(
    config: AutorunConfig,
    started_at: str,
    error_message: str,
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    """Persist the last failure without clobbering previously seen cards."""
    state = load_json_file(config.autorun_state_file, {})
    completed_at = now_iso()
    consecutive_failures = int(state.get("consecutive_failures", 0) or 0) + 1
    failure_summary = {
        "status": "failure",
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": round(float(duration_seconds or 0.0), 3),
        "error": error_message,
    }
    updated_state = {
        "schema_version": AUTORUN_STATE_SCHEMA_VERSION,
        "updated_at": completed_at,
        "last_successful_cycle_at": state.get("last_successful_cycle_at"),
        "last_failed_cycle_at": completed_at,
        "last_seen_assigned_card_ids": state.get("last_seen_assigned_card_ids", []),
        "last_seen_actionable_card_ids": state.get("last_seen_actionable_card_ids", []),
        "last_launched_card_id": state.get("last_launched_card_id"),
        "suppressed_card_ids": normalize_card_ids(state.get("suppressed_card_ids", []) or []),
        "consecutive_failures": consecutive_failures,
        "recent_cycles": truncate_history(
            [
                *(state.get("recent_cycles", []) or []),
                build_recent_cycle_entry(failure_summary),
            ]
        ),
        "last_cycle": failure_summary,
    }
    latest_state = load_json_file(config.autorun_state_file, state)
    write_json_file(
        config.autorun_state_file,
        preserve_runtime_state(latest_state if isinstance(latest_state, dict) else {}, updated_state),
    )
    append_log_block(config.log_file, build_failure_log_lines(failure_summary, consecutive_failures))
    return updated_state


def get_pipeline_stage(project_path: Path) -> str | None:
    """Poll checkpoint for current pipeline stage."""
    checkpoint = project_path / "checkpoint.json"
    if not checkpoint.exists():
        return None
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8", errors="replace"))
        return data.get("last_completed_stage")
    except (json.JSONDecodeError, OSError):
        return None


def detect_incomplete_cycle(autorun_state: dict[str, Any]) -> bool:
    """Check for incomplete previous cycle (crash recovery detection)."""
    runtime = autorun_state.get("runtime", {})
    return runtime.get("status") == "running_cycle"


def run_cycle(config: AutorunConfig) -> dict[str, Any]:
    """Run one autorun cycle."""
    started_at = now_iso()
    started_monotonic = time.monotonic()

    # Check for crash recovery - incomplete previous cycle
    previous_state = load_json_file(config.autorun_state_file, {})
    if detect_incomplete_cycle(previous_state):
        append_log_line(
            config.log_file,
            f"[{now_iso()}] crash-recovery detected incomplete previous cycle, continuing..."
        )

    update_active_autorun_lease(
        status="running_cycle",
        current_step="",
        cycle_started_at=started_at,
        target_card_ids=config.card_ids,
    )
    before_state = load_json_file(config.queue_state_file, {})
    before_snapshot = build_queue_snapshot(before_state)
    previous_state = load_json_file(config.autorun_state_file, {})

    # Validate card IDs if requested
    is_valid, missing_ids = validate_card_ids(
        config,
        before_snapshot.get("actionable_card_ids", []) or [],
    )
    if not is_valid and config.card_ids:
        append_log_line(
            config.log_file,
            f"[{now_iso()}] card-validation-warning missing_ids={','.join(missing_ids)}",
        )

    previous_assigned = previous_state.get("last_seen_assigned_card_ids", []) or []
    previous_actionable = previous_state.get("last_seen_actionable_card_ids", []) or []
    cycle_target_card_ids, previous_suppressed_card_ids = resolve_cycle_target_card_ids(
        config,
        before_snapshot,
        previous_state,
    )
    step_results: list[dict[str, Any]] = []
    prepare_result: subprocess.CompletedProcess[str] | None = None

    for subcommand, args in build_workflow_steps(config, cycle_target_card_ids):
        update_active_autorun_lease(
            status="running_cycle",
            current_step=subcommand,
            cycle_started_at=started_at,
            target_card_ids=cycle_target_card_ids,
        )
        result, step_result = run_workflow_step(config, subcommand, args)
        step_results.append(step_result)
        if subcommand == "prepare":
            prepare_result = result
    update_active_autorun_lease(
        status="running_cycle",
        current_step="",
        cycle_started_at=started_at,
        target_card_ids=cycle_target_card_ids,
    )

    after_state, after_snapshot = recover_queue_snapshot(config, before_state, step_results)
    new_running_card_ids = diff_new_card_ids(
        after_snapshot["running_card_ids"],
        before_snapshot["running_card_ids"],
    )
    launched_card_id = extract_launch_card_id((prepare_result.stdout if prepare_result else ""))
    if launched_card_id is None and len(new_running_card_ids) == 1:
        launched_card_id = new_running_card_ids[0]
    after_pipelines = after_state.get("pipelines", {}) if isinstance(after_state.get("pipelines"), dict) else {}
    launched_entry = (
        after_pipelines.get(str(launched_card_id).lower(), {}) if launched_card_id else {}
    )
    launched_project = launched_entry.get("project", {}) if isinstance(launched_entry, dict) else {}
    launched_progress = launched_project.get("local_progress", {}) if isinstance(launched_project, dict) else {}
    running_after_ids = {
        str(card_id).strip().lower()
        for card_id in (after_snapshot["running_card_ids"] or [])
        if str(card_id).strip()
    }
    ready_after_ids = {
        str(card_id).strip().lower()
        for card_id in (after_snapshot["ready_card_ids"] or [])
        if str(card_id).strip()
    }
    launch_outcome = "not_launched"
    startup_failed_card_ids: list[str] = []
    if launched_card_id:
        launched_key = str(launched_card_id).strip().lower()
        if launched_key in running_after_ids:
            launch_outcome = "running"
        elif (
            launched_key in ready_after_ids
            and bool(launched_progress.get("has_run_logs"))
            and not bool(launched_progress.get("running_signal"))
        ):
            launch_outcome = "startup_failed"
            startup_failed_card_ids = [launched_card_id]
        else:
            launch_outcome = "launched_pending_confirmation"
    suppressed_card_ids = resolve_suppressed_card_ids(
        previous_suppressed_card_ids,
        after_snapshot["actionable_card_ids"],
    )
    if launch_outcome == "startup_failed" and launched_card_id:
        suppressed_lookup = {card_id.lower() for card_id in suppressed_card_ids}
        if launched_card_id.lower() not in suppressed_lookup:
            suppressed_card_ids.append(launched_card_id)
            suppressed_card_ids = normalize_card_ids(suppressed_card_ids)
    elif launch_outcome == "running" and launched_card_id:
        suppressed_card_ids = [
            card_id for card_id in suppressed_card_ids
            if card_id.lower() != launched_card_id.lower()
        ]
    warning_step_names = [
        str(step.get("name") or "")
        for step in step_results
        if str(step.get("status") or "") != "success"
    ]

    completed_at = now_iso()
    duration_seconds = round(time.monotonic() - started_monotonic, 3)

    # Compute error code from step results
    error_code = ERR_NONE
    for step_result in step_results:
        if step_result.get("status") != "success":
            step_name = step_result.get("name", "")
            returncode = step_result.get("returncode", 0)
            if step_name == "prepare":
                error_code = ERR_QUEUE_PREPARE
            elif step_name == "archive-completed":
                error_code = ERR_QUEUE_ARCHIVE
            elif step_name == "discord-prepare":
                error_code = ERR_QUEUE_DISCORD
            elif returncode == 124:
                error_code = ERR_STEP_TIMEOUT
            elif is_transient_command_failure(
                build_failed_process(
                    [],
                    returncode=returncode,
                    stdout=step_result.get("stdout_tail", ""),
                    stderr=step_result.get("stderr_tail", ""),
                )
            ):
                error_code = ERR_STEP_TRANSIENT
            else:
                error_code = ERR_UNEXPECTED
            break

    # Get pipeline stage if running
    pipeline_stage = None
    if launch_outcome == "running" and launched_project:
        project_path_str = str(launched_project.get("path") or "")
        if project_path_str:
            pipeline_stage = get_pipeline_stage(Path(project_path_str))

    summary = {
        "status": "success_with_warnings" if warning_step_names else "success",
        "error_code": error_code,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "target_card_ids": cycle_target_card_ids,
        "suppressed_card_ids": suppressed_card_ids,
        "queue_generated_at_before": before_snapshot.get("generated_at"),
        "queue_generated_at_after": after_snapshot.get("generated_at"),
        "assigned_card_ids": after_snapshot["assigned_card_ids"],
        "actionable_card_ids": after_snapshot["actionable_card_ids"],
        "new_assigned_card_ids": diff_new_card_ids(
            after_snapshot["assigned_card_ids"],
            previous_assigned,
        ),
        "new_actionable_card_ids": diff_new_card_ids(
            after_snapshot["actionable_card_ids"],
            previous_actionable,
        ),
        "running_before_card_ids": before_snapshot["running_card_ids"],
        "running_after_card_ids": after_snapshot["running_card_ids"],
        "ready_after_card_ids": after_snapshot["ready_card_ids"],
        "ready_without_lipsync_card_ids": after_snapshot["ready_without_lipsync_card_ids"],
        "lipsync_attention_card_ids": after_snapshot["lipsync_attention_card_ids"],
        "submission_pending_card_ids": after_snapshot["submission_pending_card_ids"],
        "queue_counts_before": before_snapshot["queue_counts"],
        "queue_counts_after": after_snapshot["queue_counts"],
        "launched_card_id": launched_card_id,
        "launch_outcome": launch_outcome,
        "pipeline_stage": pipeline_stage,
        "launched_latest_log": str(launched_progress.get("latest_log") or ""),
        "launched_project_dirs": list(launched_project.get("local_project_dirs", []) or []),
        "startup_failed_card_ids": startup_failed_card_ids,
        "workflow_steps": step_results,
        "warning_step_names": warning_step_names,
    }

    # Run timing verification if requested and launch was successful
    verify_result: subprocess.CompletedProcess[str] | None = None
    if config.verify and launch_outcome == "running" and launched_project:
        project_path = str(launched_project.get("path") or "")
        if project_path:
            verify_result = run_timing_verification(config, project_path)
            summary["verify_returncode"] = verify_result.returncode
            summary["verify_stdout_tail"] = tail_text(verify_result.stdout)
            summary["verify_stderr_tail"] = tail_text(verify_result.stderr)

    # Spawn watch task if requested and launch was successful
    if config.auto_watch and launch_outcome == "running" and launched_project:
        project_path = str(launched_project.get("path") or "")
        if project_path:
            watch_spawned = spawn_watch_task(config, project_path)
            summary["watch_spawned"] = watch_spawned

    autorun_state = {
        "schema_version": AUTORUN_STATE_SCHEMA_VERSION,
        "updated_at": completed_at,
        "last_successful_cycle_at": completed_at,
        "last_failed_cycle_at": previous_state.get("last_failed_cycle_at"),
        "last_seen_assigned_card_ids": after_snapshot["assigned_card_ids"],
        "last_seen_actionable_card_ids": after_snapshot["actionable_card_ids"],
        "last_launched_card_id": launched_card_id or previous_state.get("last_launched_card_id"),
        "suppressed_card_ids": suppressed_card_ids,
        "consecutive_failures": 0,
        "recent_cycles": truncate_history(
            [
                *(previous_state.get("recent_cycles", []) or []),
                build_recent_cycle_entry(summary),
            ]
        ),
        "last_cycle": summary,
    }
    latest_state = load_json_file(config.autorun_state_file, previous_state)
    write_json_file(
        config.autorun_state_file,
        preserve_runtime_state(latest_state if isinstance(latest_state, dict) else {}, autorun_state),
    )
    append_log_block(config.log_file, build_cycle_log_lines(summary))
    return summary


def create_stop_file(stop_file: Path, log_file: Path | None = None) -> None:
    """Create a stop marker that the loop will consume."""
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    stop_file.write_text(timestamp, encoding="utf-8")
    if log_file is not None:
        append_log_line(log_file, f"[{timestamp}] stop-file-created path={stop_file}")


def clear_startup_stop_file(stop_file: Path, log_file: Path | None = None) -> bool:
    """Remove any pre-existing stop marker before a fresh autorun invocation."""
    if not stop_file.exists():
        return False

    try:
        stop_file.unlink()
    except OSError:
        return False

    if log_file is not None:
        append_log_line(log_file, f"[{now_iso()}] startup-stop-file-cleared path={stop_file}")
    return True


def consume_stop_file(stop_file: Path, log_file: Path | None = None) -> bool:
    """Consume a stop marker if present."""
    if not stop_file.exists():
        return False

    try:
        stop_file.unlink()
    except OSError:
        pass

    if log_file is not None:
        append_log_line(log_file, f"[{now_iso()}] stop-file-consumed path={stop_file}")
    return True


def consume_queue_stop_file(log_file: Path | None = None) -> bool:
    """Check and consume the unified queue stop file (Degold/queue_stop.txt)."""
    stop_file = QUEUE_STOP_FILE
    if not stop_file.exists():
        return False

    try:
        content = stop_file.read_text(encoding="utf-8").strip().lower()
        # Only stop if the content is "true", "1", or "stop"
        if content in ("true", "1", "stop"):
            stop_file.unlink()
            if log_file is not None:
                append_log_line(log_file, f"[{now_iso()}] queue-stop-file-consumed path={stop_file}")
            return True
    except OSError:
        pass
    return False


def check_unified_stop(stop_file: Path, log_file: Path | None = None) -> bool:
    """Check both the local stop file and the unified queue stop file."""
    if consume_stop_file(stop_file, log_file):
        return True
    if consume_queue_stop_file(log_file):
        return True
    return False


def check_and_consume_force_cycle(log_file: Path) -> bool:
    """Check for and consume a force-cycle signal file. Returns True if found."""
    if FORCE_CYCLE_FILE.exists():
        try:
            FORCE_CYCLE_FILE.unlink(missing_ok=True)
            append_log_line(
                log_file,
                f"[{now_iso()}] force-cycle-consumed path={FORCE_CYCLE_FILE}",
            )
            return True
        except OSError:
            pass
    return False


def sleep_with_stop_checks(
    stop_file: Path,
    log_file: Path,
    total_seconds: float,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_seconds: float = STOP_FILE_POLL_SECONDS,
    stop_event: threading.Event | None = None,
) -> bool:
    """Sleep in small chunks so stop-file requests are honored promptly.

    Returns True if a stop signal was received, False otherwise.
    Also breaks early (returning False) if a force-cycle signal is detected.
    """
    remaining = max(0.0, float(total_seconds))
    poll_interval = max(0.1, float(poll_seconds))

    while remaining > 0:
        if check_unified_stop(stop_file, log_file):
            return True
        if stop_event is not None and stop_event.is_set():
            return True
        if check_and_consume_force_cycle(log_file):
            return False  # Not a stop — just skip the remaining sleep
        chunk = min(poll_interval, remaining)
        sleep_fn(chunk)
        remaining -= chunk

    return check_unified_stop(stop_file, log_file)


def compute_failure_sleep_seconds(config: AutorunConfig, consecutive_failures: int) -> float:
    """Return bounded retry sleep after a failed cycle."""
    normal_interval_seconds = max(0.0, float(config.interval_minutes) * 60.0)
    retry_base_seconds = max(0.0, float(config.failure_retry_minutes) * 60.0)
    if retry_base_seconds <= 0:
        return normal_interval_seconds

    exponent = max(0, int(consecutive_failures or 1) - 1)
    retry_seconds = retry_base_seconds * (2 ** exponent)
    if normal_interval_seconds <= 0:
        return retry_seconds
    return min(normal_interval_seconds, retry_seconds)


def build_loop_start_lines(config: AutorunConfig) -> list[str]:
    """Summarize runtime configuration at the start of a loop invocation."""
    return [
        (
            f"[{now_iso()}] autorun-start once={str(config.once).lower()} "
            f"interval_minutes={float(config.interval_minutes):.3f} "
            f"failure_retry_minutes={float(config.failure_retry_minutes):.3f} "
            f"refresh_lipsync={str(config.refresh_lipsync).lower()} "
            f"skip_discord_prepare={str(config.skip_discord_prepare).lower()} "
            f"command_timeout_seconds={int(config.command_timeout_seconds)} "
            f"stop_when_idle={str(config.stop_when_idle).lower()}"
        ),
        f"  queue_script={config.queue_script}",
        f"  queue_state_file={config.queue_state_file}",
        f"  autorun_state_file={config.autorun_state_file}",
        f"  lock_file={config.lock_file}",
        f"  log_file={config.log_file}",
        f"  stop_file={config.stop_file}",
        f"  target_card_ids={format_id_list(config.card_ids)}",
    ]


def run_loop(
    config: AutorunConfig,
    *,
    cycle_runner: Callable[[AutorunConfig], dict[str, Any]] = run_cycle,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Run the autorun loop until stopped or until --once completes."""
    global _ACTIVE_AUTORUN_LEASE, _GLOBAL_STOP_EVENT

    # Setup signal handlers for graceful shutdown
    stop_event = threading.Event()
    _GLOBAL_STOP_EVENT = stop_event
    _setup_signal_handlers(stop_event)

    lease = acquire_autorun_lease(config)
    if lease is None:
        return 0

    _ACTIVE_AUTORUN_LEASE = lease
    exit_reason = "unknown"
    try:
        append_log_block(config.log_file, build_loop_start_lines(config))
        lease.refresh(status="idle", current_step="", cycle_started_at=None, target_card_ids=())
        while True:
            if check_unified_stop(config.stop_file, config.log_file):
                exit_reason = "stop-file-before-cycle"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Stop signal received. Exiting.")
                return 0

            if stop_event.is_set():
                exit_reason = "signal-received"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Signal received. Exiting.")
                return 0

            sleep_seconds = max(0.0, config.interval_minutes * 60.0)
            sleep_reason = "interval"
            started_at = now_iso()
            started_monotonic = time.monotonic()
            lease.refresh(
                status="running_cycle",
                current_step="",
                cycle_started_at=started_at,
                target_card_ids=config.card_ids,
            )
            try:
                summary = cycle_runner(config)
            except Exception as exc:
                error_message = str(exc).strip() or exc.__class__.__name__
                failure_state = record_cycle_failure(
                    config,
                    started_at,
                    error_message,
                    duration_seconds=time.monotonic() - started_monotonic,
                )
                consecutive_failures = int(failure_state.get("consecutive_failures", 1) or 1)
                sleep_seconds = compute_failure_sleep_seconds(config, consecutive_failures)
                sleep_reason = f"failure_backoff consecutive_failures={consecutive_failures}"
                lease.refresh(status="sleeping_after_failure", current_step="", cycle_started_at=None, target_card_ids=())
                print(f"Autorun cycle failed: {error_message}")
                if config.once:
                    exit_reason = "once-failure"
                    return 1
            else:
                print(format_cycle_log(summary))
                lease.refresh(status="idle", current_step="", cycle_started_at=None, target_card_ids=())
                if summary.get("launch_outcome") == "startup_failed":
                    normal_interval_seconds = max(0.0, config.interval_minutes * 60.0)
                    remaining_ids = remaining_unsuppressed_actionable_ids(summary)
                    if remaining_ids:
                        sleep_seconds = min(normal_interval_seconds, STARTUP_FAILURE_CONTINUE_SECONDS)
                        sleep_reason = (
                            "startup_failed_continue "
                            f"remaining_targets={format_id_list(remaining_ids)}"
                        )
                    else:
                        suppressed_retry_ids = remaining_suppressed_actionable_ids(summary)
                        if suppressed_retry_ids:
                            retry_sleep_seconds = max(
                                STARTUP_FAILURE_CONTINUE_SECONDS,
                                float(config.failure_retry_minutes) * 60.0,
                            )
                            sleep_seconds = min(normal_interval_seconds, retry_sleep_seconds)
                            sleep_reason = (
                                "startup_failed_retry_suppressed "
                                f"targets={format_id_list(suppressed_retry_ids)}"
                            )
                if config.stop_when_idle and config.card_ids:
                    suppressed_ids = {
                        str(card_id).strip().lower()
                        for card_id in (summary.get("suppressed_card_ids", []) or [])
                        if str(card_id).strip()
                    }
                    target_ids = {
                        card_id.lower() for card_id in config.card_ids
                        if card_id.lower() not in suppressed_ids
                    }
                    actionable_or_running = {
                        str(card_id).strip().lower()
                        for card_id in (
                            list(summary.get("actionable_card_ids", []) or [])
                            + list(summary.get("running_after_card_ids", []) or [])
                        )
                        if str(card_id).strip()
                    }
                    if not (actionable_or_running & target_ids):
                        exit_reason = "target-set-drained"
                        append_log_line(
                            config.log_file,
                            f"[{now_iso()}] autorun-exit reason={exit_reason} targets={format_id_list(config.card_ids)}",
                        )
                        print("Target card set drained; exiting autorun loop.")
                        return 0
                if config.once:
                    exit_reason = "once-complete"
                    append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                    return 0

            if check_unified_stop(config.stop_file, config.log_file):
                exit_reason = "stop-file-after-cycle"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Stop signal received. Exiting.")
                return 0

            if stop_event.is_set():
                exit_reason = "signal-during-cycle"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Signal received. Exiting.")
                return 0

            lease.refresh(status="sleeping", current_step="", cycle_started_at=None, target_card_ids=())
            append_log_line(
                config.log_file,
                f"[{now_iso()}] sleeping reason={sleep_reason} interval_seconds={sleep_seconds:.0f}",
            )
            if sleep_with_stop_checks(
                config.stop_file,
                config.log_file,
                sleep_seconds,
                sleep_fn=sleep_fn,
            ):
                exit_reason = "stop-file-during-sleep"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Stop signal received. Exiting.")
                return 0

            if stop_event.is_set():
                exit_reason = "signal-during-sleep"
                append_log_line(config.log_file, f"[{now_iso()}] autorun-exit reason={exit_reason}")
                print(f"Signal received. Exiting.")
                return 0
    finally:
        _ACTIVE_AUTORUN_LEASE = None
        _GLOBAL_STOP_EVENT = None
        lease.close(exit_reason=exit_reason)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Hourly Degold pipeline autorun loop")
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=60.0,
        help="Polling interval in minutes (default: 60)",
    )
    parser.add_argument(
        "--failure-retry-minutes",
        type=float,
        default=DEFAULT_FAILURE_RETRY_MINUTES,
        help=(
            "Base retry interval in minutes after a failed cycle "
            f"(default: {DEFAULT_FAILURE_RETRY_MINUTES}, capped by --interval-minutes)"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle, then exit",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Create the stop file and exit",
    )
    parser.add_argument(
        "--force-cycle",
        action="store_true",
        help="Signal the running autorun to start a new cycle immediately (skip sleep)",
    )
    parser.add_argument(
        "--no-refresh-lipsync",
        action="store_true",
        help="Skip --refresh-lipsync during archive-completed",
    )
    parser.add_argument(
        "--skip-discord-prepare",
        action="store_true",
        help="Skip discord-prepare for manual ready-queue runs that do not need Discord ingestion",
    )
    parser.add_argument(
        "--card-id",
        action="append",
        help="Restrict sequential execution to these card IDs (can be provided multiple times)",
    )
    parser.add_argument(
        "--stop-when-idle",
        action="store_true",
        help="Exit after the targeted card IDs are no longer actionable or running",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=3600,
        help="Timeout per queue-state subprocess (default: 3600)",
    )
    parser.add_argument(
        "--queue-script",
        default=str(DEFAULT_QUEUE_SCRIPT),
        help=f"Path to pipeline_queue_state.py (default: {DEFAULT_QUEUE_SCRIPT})",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_QUEUE_STATE_FILE),
        help=f"Path to Degold queue state JSON (default: {DEFAULT_QUEUE_STATE_FILE})",
    )
    parser.add_argument(
        "--autorun-state-file",
        default=str(DEFAULT_AUTORUN_STATE_FILE),
        help=f"Path to autorun state JSON (default: {DEFAULT_AUTORUN_STATE_FILE})",
    )
    parser.add_argument(
        "--lock-file",
        default=str(DEFAULT_LOCK_FILE),
        help=f"Path to autorun lock/heartbeat JSON (default: {DEFAULT_LOCK_FILE})",
    )
    parser.add_argument(
        "--stop-file",
        default=str(DEFAULT_STOP_FILE),
        help=f"Path to stop marker file (default: {DEFAULT_STOP_FILE})",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help=f"Path to autorun log file (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run timing verification after each successful pipeline run",
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=0.5,
        help="Maximum allowed drift in seconds for timing verification (default: 0.5)",
    )
    parser.add_argument(
        "--auto-watch",
        action="store_true",
        help="Automatically spawn a watch task for the active project after launch",
    )
    parser.add_argument(
        "--clear-suppressed",
        action="store_true",
        help="Clear the suppressed card list at startup",
    )
    parser.add_argument(
        "--validate-card-ids",
        action="store_true",
        default=True,
        help="Warn if provided card IDs don't match actionable queue (default: True)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current queue status and exit (no execution)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cycle actions without executing them",
    )
    parser.add_argument(
        "--list-suppressed",
        action="store_true",
        help="List currently suppressed card IDs and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--pipeline-timeout-minutes",
        type=int,
        default=480,
        help="Timeout for launched pipeline (default: 480 = 8 hours)",
    )
    parser.add_argument(
        "--auto-lipsync",
        action="store_true",
        help="Trigger lipsync after pipeline completes",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        help="Discord webhook URL for notifications",
    )
    return parser


def show_queue_status(config: AutorunConfig) -> int:
    """Show current queue status and exit."""
    # Load autorun state
    autorun_state = load_json_file(config.autorun_state_file, {})

    # Load queue state
    queue_state = load_json_file(config.queue_state_file, {})

    # Determine status
    lock_data = load_json_file(config.lock_file, {})
    status = lock_data.get("status", "stopped")

    # Get counts - card IDs are nested under "queue" key
    queue_data = queue_state.get("queue", queue_state)
    ready_count = len(queue_data.get("ready_card_ids", []))
    blocked_count = len(queue_data.get("blocked_card_ids", []))
    completed_count = len(queue_data.get("completed_card_ids", []))

    # Get current card
    current_card = lock_data.get("target_card_ids", [])
    current_card_id = current_card[0] if current_card else "none"

    # Get last cycle info
    last_cycle = autorun_state.get("last_cycle", {})
    last_cycle_outcome = last_cycle.get("launch_outcome", "none")
    cycle_start = last_cycle.get("cycle_started_at")
    if cycle_start:
        cycle_time = parse_iso_datetime(cycle_start)
        if cycle_time:
            elapsed = (datetime.now(timezone.utc) - cycle_time).total_seconds()
            if elapsed < 60:
                last_cycle_time = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                last_cycle_time = f"{int(elapsed/60)}min ago"
            else:
                last_cycle_time = f"{int(elapsed/3600)}h ago"
        else:
            last_cycle_time = "unknown"
    else:
        last_cycle_time = "none"

    # Get uptime
    started_at = lock_data.get("started_at")
    if started_at:
        start_time = parse_iso_datetime(started_at)
        if start_time:
            uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            if uptime_seconds < 60:
                uptime = f"{int(uptime_seconds)}s"
            elif uptime_seconds < 3600:
                uptime = f"{int(uptime_seconds/60)}min"
            else:
                uptime = f"{int(uptime_seconds/3600)}h {int((uptime_seconds%3600)/60)}m"
        else:
            uptime = "unknown"
    else:
        uptime = "N/A"

    print(f"Autorun Status: {status.upper()}")
    print(f"Current Card: {current_card_id}")
    print(f"Queue: {ready_count} ready, {blocked_count} blocked, {completed_count} completed")
    print(f"Last Cycle: {last_cycle_time}, {last_cycle_outcome}")
    print(f"Uptime: {uptime}")

    return 0


def list_suppressed_cards(config: AutorunConfig) -> int:
    """List currently suppressed card IDs and exit."""
    autorun_state = load_json_file(config.autorun_state_file, {})
    suppressed = autorun_state.get("suppressed_card_ids", [])

    if suppressed:
        print("Suppressed card IDs:")
        for card_id in suppressed:
            print(f"  - {card_id}")
    else:
        print("No suppressed card IDs")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AutorunConfig(
        queue_script=Path(args.queue_script),
        queue_state_file=Path(args.state_file),
        autorun_state_file=Path(args.autorun_state_file),
        lock_file=Path(args.lock_file),
        stop_file=Path(args.stop_file),
        log_file=Path(args.log_file),
        python_executable=sys.executable,
        interval_minutes=float(args.interval_minutes),
        failure_retry_minutes=float(args.failure_retry_minutes),
        once=bool(args.once),
        refresh_lipsync=not bool(args.no_refresh_lipsync),
        command_timeout_seconds=int(args.command_timeout_seconds),
        card_ids=tuple(normalize_card_ids(args.card_id or [])),
        stop_when_idle=bool(args.stop_when_idle),
        skip_discord_prepare=bool(args.skip_discord_prepare),
        verify=bool(args.verify),
        verify_threshold=float(args.verify_threshold),
        auto_watch=bool(args.auto_watch),
        clear_suppressed=bool(args.clear_suppressed),
        validate_card_ids=bool(args.validate_card_ids),
        status=bool(args.status),
        dry_run=bool(args.dry_run),
        list_suppressed=bool(args.list_suppressed),
        verbose=bool(args.verbose),
        pipeline_timeout_minutes=int(args.pipeline_timeout_minutes),
        auto_lipsync=bool(args.auto_lipsync),
        webhook_url=args.webhook_url,
    )

    # Handle --status flag: show current queue status and exit
    if args.status:
        return show_queue_status(config)

    # Handle --list-suppressed flag: list suppressed cards and exit
    if args.list_suppressed:
        return list_suppressed_cards(config)

    if args.force_cycle:
        FORCE_CYCLE_FILE.write_text("force", encoding="utf-8")
        print(f"Created force-cycle signal: {FORCE_CYCLE_FILE}")
        return 0

    if args.stop:
        create_stop_file(config.stop_file, config.log_file)
        print(f"Created stop signal: {config.stop_file}")
        return 0

    cleared_stop = clear_startup_stop_file(config.stop_file, config.log_file)
    if cleared_stop:
        print(f"Cleared stale stop signal: {config.stop_file}")

    cleared_suppressed = clear_suppressed_cards(config)
    if cleared_suppressed:
        print(f"Cleared suppressed card list at startup")

    return run_loop(config)


if __name__ == "__main__":
    raise SystemExit(main())
