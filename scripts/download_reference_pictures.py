"""
Download reference pictures of a person or entity from Google/Bing/Yandex/Wikipedia.

Uses pyimagedl 0.4.9 ImageClient directly (the repo's GoogleBingImageClient
wrapper is pinned to 0.2.1 API surface and needs a port — see note below).

Usage:
    python scripts/download_reference_pictures.py \\
        --query "Mario Moreno Reyes" \\
        --output "E:/Edit Job/Samples Projects/Fransisco_2026-07-09/Fransisco/reference_images/mario_moreno_reyes"

    python scripts/download_reference_pictures.py \\
        --query "Rodrigo Belmonte Aguilar" \\
        --context "alcalde Michoacán" \\
        --output "E:/Edit Job/Samples Projects/Fransisco_2026-07-10/reference_images/rodrigo_belmonte_aguilar" \\
        --max 30

Caveats:
- The wrappers in src/media_sources/images/google_bing.py pass kwargs from
  the 0.2.1 API; with pyimagedl 0.4.9 those are silently ignored and the
  wrapper returns 0 images. This script bypasses the wrapper.
- Search results are thumbnails; cannot programmatically verify they depict
  the named person. Each saved image has a .json sidecar with its source URL
  for visual audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download-reference-pictures")

from imagedl.imagedl import ImageClient
from imagedl.modules.utils.structure import ImageInfo

DEFAULT_SOURCES = ["GoogleImageClient", "BingImageClient", "YandexImageClient", "WikipediaImageClient"]


def slugify(name: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s[:max_len] or "subject"


def build_query_variants(name: str, context: str | None) -> list[str]:
    """Generate search variants from a name and optional context phrase.

    Context is split on commas so each fragment becomes its own paired query
    (e.g. "--context 'rural farming, simple living'" with name "Amish" yields
    "Amish rural farming", "Amish simple living", "rural farming Amish",
    "simple living Amish", plus the original full-phrase variant).
    """
    name = name.strip()
    fragments: list[str] = [name]
    if context:
        # Full-phrase variants first (existing behavior), then per-fragment
        # splits so each concept gets its own weighted search.
        ctx = context.strip()
        if ctx:
            fragments.append(f"{name} {ctx}")
            fragments.append(f"{ctx} {name}")
            for piece in (p.strip() for p in ctx.split(",")):
                if piece and piece.lower() != name.lower():
                    fragments.append(f"{name} {piece}")
                    fragments.append(f"{piece} {name}")
    fragments += [
        f"{name} portrait",
        f"{name} photo",
        f"{name} biography",
    ]
    # Dedupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for q in fragments:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download reference pictures of a person/entity using pyimagedl.",
    )
    p.add_argument("--query", required=True, help="Full name of the person/entity, e.g. 'Mario Moreno Reyes'")
    p.add_argument(
        "--context",
        default=None,
        help="Optional context phrase to scope the search, e.g. 'alcalde Michoacán' or 'Golden Age Mexican cinema'",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination directory. Will be created if missing. Images and .json sidecars are saved here.",
    )
    p.add_argument("--max", type=int, default=25, help="Target image count (default 25)")
    p.add_argument("--min-size-kb", type=int, default=8, help="Minimum image size in KB (default 8)")
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated imagedl sources. Default: {','.join(DEFAULT_SOURCES)}",
    )
    p.add_argument(
        "--per-source-limit",
        type=int,
        default=40,
        help="Max candidates per (query, source) pair (default 40)",
    )
    p.add_argument(
        "--exclude-domains",
        default="alamy.com,alamy.ae,alamy.de,alamy.es,alamy.fr,alamy.it,alamy-media.co.uk",
        help=(
            "Comma-separated domains to skip when picking candidates (default filters Alamy stock photos). "
            "Match is a substring on the URL. Pass an empty string to disable."
        ),
    )
    p.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep imagedl's intermediate work directory (default: removed after run)",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce logging to WARNING")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if not sources:
        logger.error("No valid sources in --sources")
        return 1

    for s in sources:
        if s not in {"GoogleImageClient", "BingImageClient", "YandexImageClient", "WikipediaImageClient",
                     "DuckduckgoImageClient", "FlickrImageClient", "PexelsImageClient", "UnsplashImageClient",
                     "PixabayImageClient", "YahooImageClient"}:
            logger.warning(f"Unknown source '{s}' — proceeding anyway (imagedl will skip unknown)")

    queries = build_query_variants(args.query, args.context)
    logger.info(f"Subject: {args.query!r}" + (f" (context: {args.context!r})" if args.context else ""))
    logger.info(f"Output:  {out_dir}")
    logger.info(f"Sources: {sources}")
    logger.info(f"Queries: {queries}")

    client = ImageClient(
        image_sources=sources,
        init_image_clients_cfg={
            s: {
                "work_dir": str(work_dir / s),
                "disable_print": True,
                "max_retries": 2,
                "maintain_session": True,
            }
            for s in sources
        },
        clients_threadings={s: 1 for s in sources},
        # pyimagedl's BaseImageClient.get() calls requests.Session.get(...) with
        # no timeout kwarg, which defaults to an indefinite wait. A single
        # unreachable Yandex/Bing image URL can hang the whole download pass
        # forever. Force a bounded timeout on every download request so a stuck
        # image retries against the next candidate URL (and eventually drops)
        # instead of stalling the subprocess indefinitely.
        requests_overrides={
            s: {
                "timeout": (10.0, 30.0),  # (connect_timeout, read_timeout) seconds
            }
            for s in sources
        },
        search_filters={s: {} for s in sources},
    )

    started = time.time()
    all_infos: list[ImageInfo] = []
    for query in queries:
        logger.info(f"Searching '{query}'")
        results_by_source = client.search(
            keyword=query,
            search_limits_per_source=args.per_source_limit,
        )
        n = 0
        for source, infos in results_by_source.items():
            if not infos:
                continue
            logger.info(f"  {source}: {len(infos)}")
            all_infos.extend(infos)
            n += len(infos)
        if n == 0:
            logger.warning(f"  No results for '{query}'")
    elapsed_search = time.time() - started
    logger.info(f"Total candidates: {len(all_infos)} (search {elapsed_search:.1f}s)")

    if not all_infos:
        logger.error("No candidates from any source.")
        return 1

    # Dedupe by URL, prefer round-robin across sources for diversity
    excluded = [d.strip().lower() for d in (args.exclude_domains or "").split(",") if d.strip()]
    if excluded:
        logger.info(f"Excluding domains: {excluded}")

    def _matches_excluded(url: str) -> bool:
        if not excluded:
            return False
        lu = url.lower()
        return any(d in lu for d in excluded)

    by_source: dict[str, list[ImageInfo]] = {}
    for info in all_infos:
        urls = info.candidate_download_urls or []
        url = urls[0] if urls else info.identifier or info.download_url
        if not url:
            continue
        info.download_url = url
        if _matches_excluded(url):
            continue
        by_source.setdefault(info.source, []).append(info)

    picked: list[ImageInfo] = []
    seen_urls: set[str] = set()
    # Round-robin across sources so we don't bias toward whichever source
    # returned first.
    src_iters = {s: iter(lst) for s, lst in by_source.items()}
    while len(picked) < args.max and src_iters:
        progressed = False
        for s in list(src_iters.keys()):
            try:
                info = next(src_iters[s])
            except StopIteration:
                del src_iters[s]
                continue
            progressed = True
            if _matches_excluded(info.download_url):
                continue
            if info.download_url in seen_urls:
                continue
            seen_urls.add(info.download_url)
            picked.append(info)
            if len(picked) >= args.max:
                break
        if not progressed:
            break

    logger.info(f"Picked {len(picked)} unique images. Downloading...")
    started_dl = time.time()
    downloaded = client.download(image_infos=picked)
    logger.info(f"Download phase: {time.time() - started_dl:.1f}s")

    slug = slugify(args.query)
    min_bytes = max(args.min_size_kb * 1024, 1024)

    saved: list[Path] = []
    for info in downloaded:
        try:
            src_path = Path(info.save_path)
            if not src_path.exists():
                continue
            if src_path.stat().st_size < min_bytes:
                continue
            ext = src_path.suffix.lower()
            if ext in {".jpeg", ".jpe"}:
                ext = ".jpg"
            idx = len(saved) + 1
            dest = out_dir / f"{slug}_{idx:02d}{ext}"
            shutil.move(str(src_path), str(dest))
            sidecar = dest.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "query": args.query,
                        "context": args.context,
                        "search_query": getattr(info, "_last_query", None),  # may be None
                        "source": info.source,
                        "candidate_download_urls": info.candidate_download_urls,
                        "description": info.description,
                        "ext": info.ext,
                        "identifier": info.identifier,
                        "saved_as": dest.name,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            saved.append(dest)
            logger.info(f"  ✓ {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            logger.warning(f"  skip {info.save_path}: {e}")

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(f"Done. {len(saved)} images saved to {out_dir}")
    if len(saved) < args.max:
        logger.warning(
            f"Requested {args.max}, saved only {len(saved)}. "
            f"Try lowering --min-size-kb, adding sources, or refining --query/--context."
        )
    return 0 if saved else 2


if __name__ == "__main__":
    sys.exit(main())
