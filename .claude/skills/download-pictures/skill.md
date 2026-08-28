---
name: download-pictures
description: Download reference pictures of a named person or entity from Google/Bing/Yandex/Wikipedia using pyimagedl. Use when the user says "download pictures of X", "find images of X", "reference photos of X", "get me photos of X", or wants reference imagery for a documentary subject, person, or place.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# Download Reference Pictures

Downloads up to N reference pictures of a named person or entity by hitting
Google Images, Bing, Yandex, and Wikipedia via `pyimagedl 0.4.9`. Saves images
plus `.json` sidecars (source URL, identifier, source engine) under a directory
you specify. Bypasses the repo's `src/media_sources/images/google_bing.py`
wrapper because that wrapper is pinned to the pyimagedl 0.2.1 API surface and
is silently broken on the installed 0.4.9.

## When to Use

- User says "download pictures of X", "find photos of X", "reference images
  of X", "get me pictures of X", or similar
- Scoped for a doc/film project: needs reference imagery of a single
  person/entity
- Caller can supply `--context` to scope the search (e.g. "alcalde
  Michoacán", "Golden Age Mexican cinema")

## Invocation

```
/download-pictures <name> <output_dir> [--context "<phrase>"] [--max N] [--min-size-kb N] [--sources ...]
```

## Direct Script Invocation (recommended — runs without prompting)

```bash
# Basic
python scripts/download_reference_pictures.py \
  --query "Mario Moreno Reyes" \
  --output "E:/Edit Job/Samples Projects/Fransisco_2026-07-09/Fransisco/reference_images/mario_moreno_reyes"

# With context to scope results
python scripts/download_reference_pictures.py \
  --query "Rodrigo Belmonte Aguilar" \
  --context "alcalde Michoacán" \
  --output "E:/Edit Job/Samples Projects/Fransisco_2026-07-10/reference_images/rodrigo_belmonte_aguilar" \
  --max 30

# Source subset, larger size floor
python scripts/download_reference_pictures.py \
  --query "Frida Kahlo" \
  --output ./frida \
  --sources GoogleImageClient,YandexImageClient \
  --min-size-kb 50
```

## What It Does

1. Auto-builds query variants from `--query` + optional `--context`:
   `<name>`, full `<name> <context>` and `<context> <name>` phrases,
   then one pair per context fragment split on commas (so `--context "rural
   farming, simple living, traditional work"` adds 6 more variants:
   `<name> rural farming`, `rural farming <name>`, etc.), plus
   `<name> portrait/photo/biography`.
2. Searches each variant across each source in parallel.
3. Round-robin picks unique images across sources (no single-source bias).
4. Filters candidates whose URL contains any substring in `--exclude-domains`
   (default: Alamy/Shutterstock/Getty hosts) before download.
5. Downloads into `<output>/_workdir/<source>/...`, then moves each to
   `<output>/<slug>_NN.ext` with a matching `<slug>_NN.json` sidecar.
6. Cleans up `_workdir` unless `--keep-workdir` is passed.

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--query` | required | Full name of the person/entity |
| `--output` | required | Destination directory (created if missing) |
| `--context` | _none_ | Scope phrase: "alcalde Michoacán", "1950s film", etc. |
| `--max` | 25 | Target image count |
| `--min-size-kb` | 8 | Lower-bound size, drops tiny thumbnails |
| `--sources` | Google+Bing+Yandex+Wikipedia | Comma-separated imagedl source IDs |
| `--per-source-limit` | 40 | Max candidates per (query, source) pair |
| `--exclude-domains` | `alamy.com,alamy.ae,alamy.de,alamy.es,alamy.fr,alamy.it,alamy-media.co.uk,shutterstock.com,gettyimages.com` | Comma-separated domains to skip (substring match on URL). Pass `""` to disable. Filters out stock-photo watermarks. |
| `--keep-workdir` | _off_ | Keep `_workdir/` after run for debugging |
| `--quiet` | _off_ | Reduce logging to WARNING |

## Caveats

- **Cannot verify identity programmatically.** Search results are thumbnails —
  some will be wrong-match, stock photos, or generic politician imagery.
  Always eyeball the saved files. Sidecar `.json` files include the source URL
  for audit.
- **Outputs are preview thumbnails**, not full-resolution originals. For
  archival hi-res photos of Mexican-cinema subjects, search Wikimedia Commons
  directly or use a paid/CC image source.
- The repo wrapper `src/media_sources/images/google_bing.py` is **still
  broken** under pyimagedl 0.4.9. This skill bypasses it. Porting the wrapper
  to the new API is a separate task — note to the user if they hit the
  pipeline's `entity_images.py` stage and get 0 results.

## Re-running

If the first run pulled wrong images, refine and re-run:

```bash
# Add context
--context "presidente municipal de Apatzingán"

# Limit sources to whichever worked
--sources GoogleImageClient,YandexImageClient

# Drop tiny thumbs more aggressively
--min-size-kb 30

# Add the original output dir to the same script call; the script
# silently overwrites existing <slug>_NN.* files.
```
