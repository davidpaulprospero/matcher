#!/usr/bin/env python3
"""
WAN Facecam Generator — talking-head lip-synced video from image + audio.

Uses the DashScope SDK (Alibaba WAN API) to generate video chunks from a
presenter image and voiceover audio, then concatenates them with ffmpeg.

Primary model: wan2.5-i2v-preview (480P, audio lip-sync)
Fallback model: wan2.2-i2v-plus (silent video only, 5s fixed)

Usage:
    python scripts/wan_facecam.py --audio vo.mp3 --image presenter.jpg
    python scripts/wan_facecam.py --audio vo.mp3 --image presenter.jpg --max-duration 60
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# On Windows, prevent subprocess from spawning visible console windows
_SUBPROCESS_FLAGS: dict = (
    {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
)
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & billing
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BILLING_LOG = PROJECT_ROOT / "clients" / "stu" / "facecam_billing.jsonl"

# Cost per 10-second chunk by resolution (wan2.5-i2v-preview, international)
# Official: $0.05/s (480P), $0.10/s (720P), $0.15/s (1080P)
# Verified against Mar 2026 invoice: 172.3s generated = $8.00 pretax
COST_PER_10S = {"480P": 0.50, "720P": 1.00, "1080P": 1.50}

# ---------------------------------------------------------------------------
# Models & defaults
# ---------------------------------------------------------------------------
MODEL_PRIMARY = "wan2.5-i2v-preview"       # Supports audio lip-sync
MODEL_FALLBACK = "wan2.2-i2v-plus"         # Silent video only (more stable)
DEFAULT_RESOLUTION = "480P"
DEFAULT_CHUNK_DURATION = 10                 # wan2.5 supports 5 or 10
MIN_AUDIO_DURATION = 3                      # API requires >= 3s audio
DEFAULT_MAX_DURATION = 120
DEFAULT_PROMPT = (
    "Professional presenter speaking directly to camera. "
    "Subtle natural head movements, occasional blinking, confident expression. "
    "Neutral background, studio lighting."
)

REGION_URLS = {
    "international": "https://dashscope-intl.aliyuncs.com/api/v1",
    "beijing": "https://dashscope.aliyuncs.com/api/v1",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ChunkResult:
    chunk_index: int
    video_path: Optional[str]
    duration: float
    model_used: str
    status: str
    error: Optional[str] = None


@dataclass
class FacecamResult:
    video_paths: List[str]
    concatenated_path: Optional[str]
    total_duration: float
    chunks_generated: int
    chunks_failed: int
    resolution: str
    models_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_file_url(path: str) -> str:
    """Convert a local path to a file:// URL (Windows-safe)."""
    p = Path(path).resolve()
    return "file:///" + str(p).replace("\\", "/")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class WanFacecamService:
    """Generates facecam (talking-head) videos via Alibaba WAN API."""

    def __init__(
        self,
        api_key: str,
        model: str = MODEL_PRIMARY,
        resolution: str = DEFAULT_RESOLUTION,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
        region: str = "international",
        billing_log: Path = DEFAULT_BILLING_LOG,
    ):
        import dashscope
        self.api_key = api_key
        self.model = model
        self.resolution = resolution
        self.chunk_duration = chunk_duration
        self.billing_log = billing_log

        # Configure SDK
        dashscope.api_key = api_key
        base_url = REGION_URLS.get(region, REGION_URLS["international"])
        dashscope.base_http_api_url = base_url
        logger.info(
            f"WanFacecamService initialized (model={model}, "
            f"resolution={resolution}, region={region})"
        )

    # ------------------------------------------------------------------
    # Billing ledger — append-only JSONL, one line per API call
    # ------------------------------------------------------------------
    def _log_billing(
        self,
        model: str,
        chunk_index: int,
        status: str,
        task_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a billing event the instant an API call is made.

        Written immediately after VideoSynthesis.async_call() so that even
        if the process crashes during wait(), the billable submission is on
        record.  Non-billable failures (submit rejected / network error)
        are logged with cost 0 for debugging.
        """
        is_billable = task_id is not None
        chunk_dur = self.chunk_duration if "wan2.5" in model else 5
        cost = (
            COST_PER_10S.get(self.resolution, 0.14) * (chunk_dur / 10)
            if is_billable
            else 0
        )
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "model": model,
            "resolution": self.resolution,
            "chunk_duration_s": chunk_dur,
            "chunk_index": chunk_index,
            "status": status,
            "est_cost_usd": round(cost, 4),
        }
        if error:
            entry["error"] = error[:300]
        try:
            self.billing_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.billing_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Billing log write failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        audio_path: str,
        image_path: str,
        output_dir: str = "./output/facecam",
        prompt: str = DEFAULT_PROMPT,
        max_duration: int = DEFAULT_MAX_DURATION,
    ) -> FacecamResult:
        """Orchestrate: chunk audio -> generate per chunk -> concat."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="wan_chunks_") as tmp:
            chunks = self._chunk_audio(audio_path, self.chunk_duration, max_duration, tmp)
            logger.info(f"Audio split into {len(chunks)} chunk(s)")

            results: List[ChunkResult] = []
            models_used: set = set()

            for i, chunk_path in enumerate(chunks):
                logger.info(f"Generating chunk {i + 1}/{len(chunks)}")
                r = self._generate_chunk_with_fallback(
                    image_path, str(chunk_path), prompt, i, str(out)
                )
                results.append(r)
                if r.status == "SUCCEEDED":
                    models_used.add(r.model_used)

        successful = [r for r in results if r.status == "SUCCEEDED"]
        failed = [r for r in results if r.status != "SUCCEEDED"]
        video_paths = [r.video_path for r in successful if r.video_path]

        # Concatenate if multiple
        concat_path = None
        if len(video_paths) > 1:
            concat_path = str(out / "facecam_final.mp4")
            concat_path = str(self._concat_videos(video_paths, concat_path))
        elif len(video_paths) == 1:
            concat_path = video_paths[0]

        return FacecamResult(
            video_paths=video_paths,
            concatenated_path=concat_path,
            total_duration=sum(r.duration for r in successful),
            chunks_generated=len(successful),
            chunks_failed=len(failed),
            resolution=self.resolution,
            models_used=sorted(models_used),
            errors=[r.error for r in failed if r.error],
        )

    # ------------------------------------------------------------------
    # Audio chunking
    # ------------------------------------------------------------------
    def _chunk_audio(
        self, audio_path: str, chunk_dur: int, max_dur: int, temp_dir: str
    ) -> List[Path]:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        max_ms = max_dur * 1000

        if total_ms > max_ms:
            audio = audio[:max_ms]
            total_ms = max_ms
            logger.info(f"Trimmed audio to {max_dur}s")

        chunk_ms = chunk_dur * 1000
        chunks: List[Path] = []

        for start in range(0, total_ms, chunk_ms):
            end = min(start + chunk_ms, total_ms)
            segment = audio[start:end]

            # Skip silent chunks — don't waste money on motionless avatar
            if segment.dBFS < -40:
                logger.info(
                    f"Skipping chunk {len(chunks)} — silence "
                    f"({segment.dBFS:.1f} dBFS < -40 dBFS)"
                )
                continue

            # Pad short final chunk to minimum API requirement (3s)
            min_ms = MIN_AUDIO_DURATION * 1000
            if len(segment) < min_ms:
                silence = AudioSegment.silent(duration=min_ms - len(segment))
                segment = segment + silence
                logger.info(
                    f"Padded chunk {len(chunks)} to {MIN_AUDIO_DURATION}s minimum"
                )

            out_path = Path(temp_dir) / f"chunk_{len(chunks):03d}.mp3"
            segment.export(str(out_path), format="mp3")
            chunks.append(out_path)

        return chunks

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------
    def _generate_chunk(
        self,
        image_path: str,
        audio_path: str,
        prompt: str,
        chunk_index: int,
        output_dir: str,
        model: str = MODEL_PRIMARY,
    ) -> ChunkResult:
        """Generate a single video chunk via DashScope SDK."""
        from dashscope import VideoSynthesis
        import requests as req

        # Pass resolved local paths — SDK auto-uploads to OSS
        img_local = str(Path(image_path).resolve())
        is_wan25 = "wan2.5" in model
        audio_local = str(Path(audio_path).resolve()) if is_wan25 else None

        logger.info(f"  Submitting to {model} (resolution={self.resolution})")
        logger.debug(f"  img_url={img_local}")
        if audio_local:
            logger.debug(f"  audio_url={audio_local}")

        # async_call with direct keyword args (dashscope >= 1.25)
        # SDK's _get_input -> check_and_upload_local handles OSS upload
        call_kwargs = dict(
            model=model,
            prompt=prompt,
            img_url=img_local,
            extend_prompt=False,
            resolution=self.resolution,
            duration=self.chunk_duration if is_wan25 else 5,
        )
        if audio_local:
            call_kwargs["audio_url"] = audio_local

        try:
            response = VideoSynthesis.async_call(**call_kwargs)
        except Exception as exc:
            self._log_billing(model, chunk_index, "call_error", error=str(exc))
            raise

        if response.status_code != HTTPStatus.OK:
            err = f"Submit failed: {response.code} - {response.message}"
            self._log_billing(model, chunk_index, "rejected", error=err)
            return ChunkResult(
                chunk_index=chunk_index,
                video_path=None,
                duration=0,
                model_used=model,
                status="FAILED",
                error=err,
            )

        task_id = response.output.task_id
        # --- billable from this point: log immediately ---
        self._log_billing(model, chunk_index, "submitted", task_id=task_id)
        logger.info(f"  Task submitted: {task_id}")

        result = VideoSynthesis.wait(task=task_id)

        if result.status_code != HTTPStatus.OK:
            err = f"Task failed: {result.code} - {result.message}"
            self._log_billing(model, chunk_index, "generation_failed",
                              task_id=task_id, error=err)
            return ChunkResult(
                chunk_index=chunk_index,
                video_path=None,
                duration=0,
                model_used=model,
                status="FAILED",
                error=err,
            )

        # Extract video URL from result
        video_url = getattr(result.output, "video_url", None)
        if not video_url:
            # Some SDK versions nest under results
            results_list = getattr(result.output, "results", None)
            if results_list and len(results_list) > 0:
                video_url = getattr(results_list[0], "url", None) or results_list[0].get("url")

        if not video_url:
            err = f"No video URL in response: {result.output}"
            self._log_billing(model, chunk_index, "no_video_url",
                              task_id=task_id, error=err)
            return ChunkResult(
                chunk_index=chunk_index,
                video_path=None,
                duration=0,
                model_used=model,
                status="FAILED",
                error=err,
            )

        # Download video
        out_path = Path(output_dir) / f"facecam_{chunk_index:03d}.mp4"
        logger.info(f"  Downloading video to {out_path}")
        try:
            resp = req.get(video_url, stream=True, timeout=(30, 300))
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for data in resp.iter_content(chunk_size=8192):
                    f.write(data)
        except Exception as exc:
            self._log_billing(model, chunk_index, "download_failed",
                              task_id=task_id, error=str(exc))
            raise

        # Get duration from usage if available
        duration = 0.0
        if hasattr(result, "usage") and result.usage:
            duration = getattr(result.usage, "duration", 0) or getattr(
                result.usage, "video_duration", 0
            )
        if not duration:
            duration = float(self.chunk_duration if is_wan25 else 5)

        self._log_billing(model, chunk_index, "completed", task_id=task_id)
        logger.info(f"  Chunk {chunk_index} complete ({duration}s, model={model})")

        return ChunkResult(
            chunk_index=chunk_index,
            video_path=str(out_path),
            duration=duration,
            model_used=model,
            status="SUCCEEDED",
        )

    def _generate_chunk_with_fallback(
        self,
        image_path: str,
        audio_path: str,
        prompt: str,
        chunk_index: int,
        output_dir: str,
    ) -> ChunkResult:
        """Try primary model, fall back to wan2.2 on failure."""
        try:
            result = self._generate_chunk(
                image_path, audio_path, prompt, chunk_index, output_dir, self.model
            )
            if result.status == "SUCCEEDED":
                return result
            logger.warning(
                f"Primary model failed for chunk {chunk_index}: {result.error}"
            )
        except Exception as e:
            logger.warning(f"Primary model error for chunk {chunk_index}: {e}")

        # Fallback to wan2.2 (silent video only)
        if self.model != MODEL_FALLBACK:
            logger.info(f"Falling back to {MODEL_FALLBACK} for chunk {chunk_index}")
            try:
                return self._generate_chunk(
                    image_path, audio_path, prompt, chunk_index, output_dir,
                    MODEL_FALLBACK,
                )
            except Exception as e:
                logger.error(f"Fallback also failed for chunk {chunk_index}: {e}")

        return ChunkResult(
            chunk_index=chunk_index,
            video_path=None,
            duration=0,
            model_used=MODEL_FALLBACK,
            status="FAILED",
            error="Both primary and fallback models failed",
        )

    # ------------------------------------------------------------------
    # Video concatenation
    # ------------------------------------------------------------------
    def _concat_videos(self, video_paths: List[str], output_path: str) -> Path:
        """Concatenate video chunks with ffmpeg."""
        out = Path(output_path)
        # Use the directory of the first video as cwd for ffmpeg,
        # with relative filenames in the filelist. This avoids encoding
        # issues with non-ASCII chars (em dashes, etc.) in Windows paths.
        chunks_dir = Path(video_paths[0]).parent

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
            dir=str(chunks_dir),
        ) as f:
            for vp in video_paths:
                fname = Path(vp).name
                f.write(f"file '{fname}'\n")
            filelist = f.name

        try:
            out_name = out.name
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", Path(filelist).name,
                "-c", "copy",
                out_name,
            ]
            logger.info(f"Concatenating {len(video_paths)} videos -> {out}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                cwd=str(chunks_dir),
                **_SUBPROCESS_FLAGS,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg concat failed: {result.stderr[:500]}")
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr[:200]}")
            # Move output to intended path if different from chunks_dir
            actual_out = chunks_dir / out_name
            if actual_out != out:
                import shutil
                shutil.move(str(actual_out), str(out))
            logger.info(f"Concatenated video saved: {out}")
        finally:
            Path(filelist).unlink(missing_ok=True)

        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate facecam video from presenter image + voiceover audio"
    )
    parser.add_argument("--audio", required=True, help="Path to voiceover audio file")
    parser.add_argument("--image", required=True, help="Path to presenter image file")
    parser.add_argument("--output", default="./output/facecam", help="Output directory")
    parser.add_argument("--model", default=MODEL_PRIMARY, help="WAN model name")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="480P|720P|1080P")
    parser.add_argument("--chunk-duration", type=int, default=DEFAULT_CHUNK_DURATION,
                        help="Seconds per video chunk (5 or 10)")
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION,
                        help="Max audio duration to process (seconds)")
    parser.add_argument("--prompt", default=None, help="Custom generation prompt")
    parser.add_argument("--region", default="international",
                        choices=["international", "beijing"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)
    if not Path(args.image).exists():
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY environment variable not set")
        sys.exit(1)

    svc = WanFacecamService(
        api_key=api_key,
        model=args.model,
        resolution=args.resolution,
        chunk_duration=args.chunk_duration,
        region=args.region,
    )

    result = svc.generate(
        audio_path=args.audio,
        image_path=args.image,
        output_dir=args.output,
        prompt=args.prompt or DEFAULT_PROMPT,
        max_duration=args.max_duration,
    )

    # Report
    print(f"\n{'='*60}")
    print(f"Facecam Generation Complete")
    print(f"{'='*60}")
    print(f"  Chunks generated: {result.chunks_generated}")
    print(f"  Chunks failed:    {result.chunks_failed}")
    print(f"  Total duration:   {result.total_duration:.1f}s")
    print(f"  Resolution:       {result.resolution}")
    print(f"  Models used:      {', '.join(result.models_used)}")

    if result.concatenated_path:
        size_mb = Path(result.concatenated_path).stat().st_size / (1024 * 1024)
        print(f"  Output:           {result.concatenated_path} ({size_mb:.1f} MB)")
    else:
        print("  Output:           No video generated")

    if result.errors:
        print(f"\n  Errors:")
        for err in result.errors:
            print(f"    - {err}")

    print(f"{'='*60}")

    sys.exit(0 if result.chunks_generated > 0 else 1)


if __name__ == "__main__":
    main()
