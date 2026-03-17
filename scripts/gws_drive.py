#!/usr/bin/env python3
"""
Shared Google Workspace CLI helpers for Drive operations.

Primary goal:
- Use `gws` for Drive file listing/downloads with structured errors.
- Let callers decide fallback behavior (for example, `gdown`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from shutil import which
import subprocess
import sys
from typing import Any

from script_utils import run_subprocess

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


GWS_AUTH_ERROR_HINTS = (
    "invalid_grant",
    "unauthorized",
    "permission denied",
    "accessnotconfigured",
    "insufficient",
    "forbidden",
    "401",
    "403",
)


@dataclass(frozen=True)
class GwsDriveContext:
    """Per-call auth context for `gws` execution."""

    token: str = ""
    credentials_file: str = ""
    impersonated_user: str = ""


class GwsDriveError(RuntimeError):
    """Typed exception for `gws` execution and parsing failures."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _resolve_gws_executable() -> str | None:
    """Resolve gws executable path (Windows-safe for .CMD shims)."""
    return which("gws")


def gws_is_available() -> bool:
    """Return True when `gws` binary exists in PATH."""
    return _resolve_gws_executable() is not None


def _is_auth_error(text: str) -> bool:
    blob = str(text or "").lower()
    return any(hint in blob for hint in GWS_AUTH_ERROR_HINTS)


def _command_env(context: GwsDriveContext) -> dict[str, str]:
    env = dict(os.environ)
    if context.token:
        env["GOOGLE_WORKSPACE_CLI_TOKEN"] = str(context.token).strip()
    else:
        env.pop("GOOGLE_WORKSPACE_CLI_TOKEN", None)
    if context.credentials_file:
        env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"] = str(context.credentials_file).strip()
    if context.impersonated_user:
        env["GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER"] = str(context.impersonated_user).strip()
    return env


def ensure_gws_ready(context: GwsDriveContext) -> None:
    """Validate minimum prerequisites before running `gws`."""
    if not gws_is_available():
        raise GwsDriveError("missing_binary", "gws CLI not found on PATH")
    if not (context.token or context.credentials_file):
        raise GwsDriveError(
            "missing_auth",
            "No gws auth configured (set GOOGLE_WORKSPACE_CLI_TOKEN or credentials file)",
        )


def run_gws(
    args: list[str],
    *,
    context: GwsDriveContext,
    timeout_seconds: int = 180,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a gws command and raise GwsDriveError on hard failures."""
    ensure_gws_ready(context)
    executable = _resolve_gws_executable()
    if not executable:
        raise GwsDriveError("missing_binary", "gws CLI not found on PATH")
    command = [executable, *args]
    result = run_subprocess(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_command_env(context),
        check=False,
        timeout=max(1, int(timeout_seconds)),
    )
    combined = "\n".join(part for part in ((result.stderr or "").strip(), (result.stdout or "").strip()) if part)
    if result.returncode != 0 and _is_auth_error(combined) and context.token:
        # If an injected bearer token is stale, retry once without it so gws can use local OAuth creds.
        retry_context = GwsDriveContext(
            token="",
            credentials_file=context.credentials_file,
            impersonated_user=context.impersonated_user,
        )
        result = run_subprocess(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_command_env(retry_context),
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
        combined = "\n".join(part for part in ((result.stderr or "").strip(), (result.stdout or "").strip()) if part)
    if result.returncode != 0:
        code = "auth_failed" if _is_auth_error(combined) else "command_failed"
        raise GwsDriveError(code, combined or f"gws exited with code {result.returncode}")
    return result


def parse_json_output(raw: str) -> Any:
    """Parse gws output as JSON or NDJSON."""
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GwsDriveError("parse_error", f"Failed to parse gws JSON output: {exc}") from exc
        return rows


def list_drive_folder_files(
    folder_id: str,
    *,
    context: GwsDriveContext,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """
    List non-trashed files inside a Drive folder.

    Returns flattened file rows with at least id/name/mimeType keys when available.
    """
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType,size,modifiedTime),nextPageToken",
        "pageSize": max(1, int(page_size)),
    }
    result = run_gws(
        ["drive", "files", "list", "--params", json.dumps(params), "--format", "json"],
        context=context,
    )
    payload = parse_json_output(result.stdout)
    if isinstance(payload, dict):
        files = payload.get("files", [])
        if isinstance(files, list):
            return [item for item in files if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        merged: list[dict[str, Any]] = []
        for page in payload:
            if isinstance(page, dict):
                files = page.get("files", [])
                if isinstance(files, list):
                    merged.extend(item for item in files if isinstance(item, dict))
        return merged
    return []


def download_drive_file(
    file_id: str,
    destination: Path,
    *,
    context: GwsDriveContext,
    timeout_seconds: int = 300,
) -> Path:
    """Download a binary Drive file to destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {"fileId": file_id, "alt": "media"}
    run_gws(
        [
            "drive",
            "files",
            "get",
            "--params",
            json.dumps(params),
            "-o",
            str(destination),
        ],
        context=context,
        timeout_seconds=timeout_seconds,
    )
    if not destination.exists() or destination.stat().st_size <= 0:
        raise GwsDriveError("empty_download", f"Downloaded file is missing/empty: {destination}")
    return destination


def export_drive_file(
    file_id: str,
    mime_type: str,
    destination: Path,
    *,
    context: GwsDriveContext,
    timeout_seconds: int = 300,
) -> Path:
    """Export a Google-native file (Docs/Sheets/etc.) to a binary format."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {"fileId": file_id, "mimeType": mime_type}
    run_gws(
        [
            "drive",
            "files",
            "export",
            "--params",
            json.dumps(params),
            "-o",
            str(destination),
        ],
        context=context,
        timeout_seconds=timeout_seconds,
    )
    if not destination.exists() or destination.stat().st_size <= 0:
        raise GwsDriveError("empty_download", f"Exported file is missing/empty: {destination}")
    return destination


def _context_from_env(
    *,
    token: str = "",
    credentials_file: str = "",
    impersonated_user: str = "",
) -> GwsDriveContext:
    return GwsDriveContext(
        token=str(token or os.getenv("GOOGLE_WORKSPACE_CLI_TOKEN") or "").strip(),
        credentials_file=str(credentials_file or os.getenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE") or "").strip(),
        impersonated_user=str(impersonated_user or os.getenv("GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER") or "").strip(),
    )


def _extract_folder_id(folder_id: str, folder_url: str) -> str:
    direct = str(folder_id or "").strip()
    if direct:
        return direct
    url = str(folder_url or "").strip()
    match = re.search(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise GwsDriveError("invalid_input", "Unable to extract folder id from --folder-url")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Google Drive helper wrapper around gws")
    parser.add_argument("--env-file", default="", help="Optional .env file to load before running (e.g. Degold/accounts/david.env)")
    parser.add_argument("--token", default="", help="Override GOOGLE_WORKSPACE_CLI_TOKEN")
    parser.add_argument("--credentials-file", default="", help="Override GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE")
    parser.add_argument("--impersonated-user", default="", help="Override GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-folder", help="List files in a Drive folder")
    list_group = list_parser.add_mutually_exclusive_group(required=True)
    list_group.add_argument("--folder-id", default="", help="Drive folder id")
    list_group.add_argument("--folder-url", default="", help="Drive folder URL")
    list_parser.add_argument("--page-size", type=int, default=200, help="Drive list page size (default: 200)")

    get_parser = subparsers.add_parser("get-file", help="Download file by Drive file id")
    get_parser.add_argument("--file-id", required=True, help="Drive file id")
    get_parser.add_argument("-o", "--output", required=True, help="Output path")

    export_parser = subparsers.add_parser("export-file", help="Export Google-native file by id")
    export_parser.add_argument("--file-id", required=True, help="Drive file id")
    export_parser.add_argument("--mime-type", required=True, help="Export MIME type")
    export_parser.add_argument("-o", "--output", required=True, help="Output path")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    if args.env_file:
        if load_dotenv is None:
            print("[ERROR] python-dotenv is required for --env-file", file=sys.stderr)
            return 4
        load_dotenv(args.env_file, override=True)

    context = _context_from_env(
        token=args.token,
        credentials_file=args.credentials_file,
        impersonated_user=args.impersonated_user,
    )
    try:
        if args.command == "list-folder":
            folder_id = _extract_folder_id(args.folder_id, args.folder_url)
            files = list_drive_folder_files(folder_id, context=context, page_size=args.page_size)
            payload = {"folder_id": folder_id, "count": len(files), "files": files}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        if args.command == "get-file":
            destination = download_drive_file(args.file_id, Path(args.output), context=context)
            print(str(destination))
            return 0
        if args.command == "export-file":
            destination = export_drive_file(args.file_id, args.mime_type, Path(args.output), context=context)
            print(str(destination))
            return 0
    except GwsDriveError as exc:
        print(f"[ERROR] {exc.code}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] unexpected: {exc}", file=sys.stderr)
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
