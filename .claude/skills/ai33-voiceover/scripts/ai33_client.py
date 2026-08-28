#!/usr/bin/env python3
"""
AI33 Text-to-Speech API client (self-contained, no EU-AUTO imports).

Adapted from EU-AUTO/EU-AUTO-main/clients/ai33_client.py but stripped of
project-specific dependencies (BaseAPIClient, TimeoutConfig, VoiceoverResult)
so it can be dropped into any repo as a standalone module.

All config is read from environment variables (load_dotenv from .env first):
  AI33_API_KEY          (required)
  AI33_API_URL          (default https://api.ai33.pro/v1)
  AI33_VOICE_ID         (default "default")
  AI33_SPEED            (default 1.04)
  AI33_STABILITY        (default 0.50)
  AI33_SIMILARITY       (default 0.60 - similarity_boost)
  AI33_STYLE            (default 0.75)
  AI33_SPEAKER_BOOST    (default true)
  AI33_WITH_TRANSCRIPT  (default false)
  AI33_POLL_INTERVAL    (seconds, default 2)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


DEFAULT_API_URL = "https://api.ai33.pro/v3"
DEFAULT_VOICE_ID = "default"
DEFAULT_POLL_INTERVAL = 2.0

# Documentary-narration defaults lifted from EU-AUTO project.
DEFAULT_VOICE_SETTINGS: Dict[str, Any] = {
    "speed": 1.04,
    "stability": 0.50,
    "similarity_boost": 0.60,
    "style": 0.75,
    "use_speaker_boost": True,
}


@dataclass
class VoiceoverResult:
    """Minimal result envelope for a single TTS job."""
    local_path: Optional[str] = None
    audio_url: Optional[str] = None
    transcript_url: Optional[str] = None
    duration: Optional[float] = None
    status: str = "unknown"
    task_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class AI33Error(RuntimeError):
    pass


class AI33TextToSpeech:
    """AI33 TTS client. Submit text, poll, download audio."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        base_url: Optional[str] = None,
        poll_interval: Optional[float] = None,
        with_transcript: Optional[bool] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        connect_timeout: float = 30.0,
        read_timeout: float = 300.0,
        download_timeout: float = 600.0,
    ) -> None:
        self.api_key = api_key or os.getenv("AI33_API_KEY")
        if not self.api_key:
            raise AI33Error(
                "AI33_API_KEY missing. Add it to .env or pass api_key= explicitly."
            )

        self.base_url = (base_url or os.getenv("AI33_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.voice_id = voice_id or os.getenv("AI33_VOICE_ID") or DEFAULT_VOICE_ID
        self.poll_interval = (
            poll_interval
            if poll_interval is not None
            else float(os.getenv("AI33_POLL_INTERVAL", DEFAULT_POLL_INTERVAL))
        )

        env_transcript = os.getenv("AI33_WITH_TRANSCRIPT", "false").lower()
        if with_transcript is None:
            self.default_with_transcript = env_transcript in ("true", "1", "yes")
        else:
            self.default_with_transcript = with_transcript

        self.voice_settings = self._load_voice_settings(voice_settings)
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.download_timeout = download_timeout

        self.headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

        logger.info(
            "AI33 client ready: voice=%s url=%s transcript=%s speed=%.2f stability=%.2f "
            "similarity=%.2f style=%.2f speaker_boost=%s",
            self.voice_id,
            self.base_url,
            self.default_with_transcript,
            self.voice_settings["speed"],
            self.voice_settings["stability"],
            self.voice_settings["similarity_boost"],
            self.voice_settings["style"],
            self.voice_settings["use_speaker_boost"],
        )

    # ------------------------------------------------------------------ env

    def _load_voice_settings(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        settings = DEFAULT_VOICE_SETTINGS.copy()
        for env_key, key in (
            ("AI33_SPEED", "speed"),
            ("AI33_STABILITY", "stability"),
            ("AI33_SIMILARITY", "similarity_boost"),
            ("AI33_STYLE", "style"),
        ):
            raw = os.getenv(env_key)
            if raw not in (None, ""):
                try:
                    settings[key] = float(raw)
                except ValueError:
                    logger.warning("Ignoring non-numeric %s=%r", env_key, raw)
        sb = os.getenv("AI33_SPEAKER_BOOST")
        if sb not in (None, ""):
            settings["use_speaker_boost"] = sb.lower() in ("true", "1", "yes")
        if override:
            settings.update(override)
        return settings

    # --------------------------------------------------------------- public

    def generate_voiceover(
        self,
        text: str,
        output_path: Optional[str] = None,
        output_format: str = "mp3_44100_128",
        with_transcript: Optional[bool] = None,
        voice_id: Optional[str] = None,
        wait_for_completion: bool = True,
    ) -> VoiceoverResult:
        if not text or not text.strip():
            raise AI33Error("generate_voiceover() requires non-empty text")

        voice_id = voice_id or self.voice_id
        if with_transcript is None:
            with_transcript = self.default_with_transcript

        logger.info("Submitting TTS: %d chars, voice=%s", len(text), voice_id)

        task_id = self._submit(
            text=text,
            voice_id=voice_id,
            output_format=output_format,
            with_transcript=with_transcript,
        )
        logger.info("Submitted task %s", task_id)

        if not wait_for_completion:
            return VoiceoverResult(task_id=task_id, status="processing")

        data = self._poll(task_id)
        meta = data.get("metadata") or {}
        audio_url = meta.get("audio_url")
        transcript_url = meta.get("transcript_url")
        duration = meta.get("duration")

        local_path: Optional[str] = None
        if audio_url and output_path:
            local_path = self.download_audio(audio_url, output_path)

        return VoiceoverResult(
            local_path=local_path,
            audio_url=audio_url,
            transcript_url=transcript_url,
            duration=duration,
            status=data.get("status", "done"),
            task_id=task_id,
            raw=data,
        )

    def list_voices(self, provider: str = "elevenlabs") -> Dict[str, Any]:
        url = f"{self.base_url}/voices"
        resp = requests.get(
            url,
            params={"provider": provider},
            headers=self.headers,
            timeout=self.read_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/task/{task_id}"
        resp = requests.get(url, headers=self.headers, timeout=self.read_timeout)
        resp.raise_for_status()
        return resp.json()

    def download_audio(self, audio_url: str, output_path: str) -> str:
        logger.info("Downloading audio -> %s", output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(audio_url, timeout=self.download_timeout, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        size = Path(output_path).stat().st_size
        logger.info("Saved %s (%d bytes)", output_path, size)
        return output_path

    # -------------------------------------------------------------- private

    def _submit(
        self,
        text: str,
        voice_id: str,
        output_format: str,
        with_transcript: bool,
    ) -> str:
        # v3 endpoint: voice_id and provider go in the JSON body, not the path.
        # Provider defaults to 'elevenlabs' since most custom voices are EL.
        provider = "elevenlabs" if voice_id.startswith("elevenlabs_") else "elevenlabs"
        url = f"{self.base_url}/text-to-speech"
        params = {"output_format": output_format}
        payload: Dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "provider": provider,
            "model_id": "eleven_multilingual_v2",
            "with_transcript": with_transcript,
            "voice_settings": {
                "stability": self.voice_settings["stability"],
                "similarity_boost": self.voice_settings["similarity_boost"],
                "style": self.voice_settings["style"],
                "use_speaker_boost": self.voice_settings["use_speaker_boost"],
            },
        }
        speed = self.voice_settings.get("speed", 1.0)
        if speed and speed != 1.0:
            payload["speed"] = speed

        resp = requests.post(
            url,
            params=params,
            headers=self.headers,
            json=payload,
            timeout=self.read_timeout,
        )
        if resp.status_code >= 400:
            raise AI33Error(f"AI33 submit HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if not data.get("success"):
            raise AI33Error(f"AI33 submit returned success=false: {data}")
        task_id = data.get("task_id")
        if not task_id:
            raise AI33Error(f"AI33 submit missing task_id: {data}")
        return task_id

    def _poll(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/task/{task_id}"
        attempt = 0
        last_log = time.time()
        while True:
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.read_timeout)
                resp.raise_for_status()
                raw = resp.json()
            except requests.RequestException as exc:
                logger.warning("Poll %d failed: %s (retrying)", attempt, exc)
                attempt += 1
                time.sleep(self.poll_interval)
                continue

            # v3 wraps payload under "data"; v1 returned the dict directly.
            data = raw.get("data", raw) if isinstance(raw, dict) else raw
            status = data.get("status")
            if status == "done":
                meta = data.get("metadata") or {}
                if meta.get("audio_url"):
                    return data
                logger.warning("Status=done but no audio_url yet, continuing")
            elif status in ("processing", "doing", "created", "pending"):
                if time.time() - last_log >= 30:
                    elapsed = (attempt * self.poll_interval) / 60
                    logger.info("Task %s... still %s (%.1f min)", task_id[:8], status, elapsed)
                    last_log = time.time()
            elif status == "failed":
                raise AI33Error(f"AI33 task failed: {data.get('error_message', data)}")
            else:
                logger.warning("Unknown status %r on attempt %d", status, attempt)

            attempt += 1
            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# CLI

def _cli() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate voiceover via AI33 TTS")
    parser.add_argument("--text", help="Inline script text (or use --input-file)")
    parser.add_argument("--input-file", help="Read script text from a .txt file")
    parser.add_argument("--output", required=True, help="Output MP3 path")
    parser.add_argument("--voice-id", help="Override AI33_VOICE_ID env var")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--stability", type=float)
    parser.add_argument("--similarity", type=float)
    parser.add_argument("--style", type=float)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--format", default="mp3_44100_128")
    parser.add_argument("--with-transcript", action="store_true")
    parser.add_argument("--no-poll", action="store_true", help="Submit only, do not wait")
    parser.add_argument(
        "--dotenv",
        default=str(Path(__file__).resolve().parents[4] / ".env"),
        help="Path to .env (default: repo root)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv(args.dotenv, override=False)
    except ImportError:
        logger.warning("python-dotenv not installed; relying on already-set env vars")

    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    if not text or not text.strip():
        parser.error("No text provided (use --text, --input-file, or pipe on stdin)")

    overrides: Dict[str, Any] = {}
    if args.speed is not None:
        overrides["speed"] = args.speed
    if args.stability is not None:
        overrides["stability"] = args.stability
    if args.similarity is not None:
        overrides["similarity_boost"] = args.similarity
    if args.style is not None:
        overrides["style"] = args.style
    if args.no_speaker_boost:
        overrides["use_speaker_boost"] = False

    client = AI33TextToSpeech(voice_settings=overrides or None, voice_id=args.voice_id)
    result = client.generate_voiceover(
        text=text,
        output_path=args.output,
        output_format=args.format,
        with_transcript=args.with_transcript or None,
        wait_for_completion=not args.no_poll,
    )
    print(f"status={result.status}")
    print(f"task_id={result.task_id}")
    print(f"audio_url={result.audio_url}")
    print(f"transcript_url={result.transcript_url}")
    print(f"duration={result.duration}")
    print(f"local_path={result.local_path}")
    return 0 if result.status in ("done", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())