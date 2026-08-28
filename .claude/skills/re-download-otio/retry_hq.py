import subprocess, os, gzip, json, re

PROJECT_DIR = r"E:\Edit Job\Kyteq\Disney\OJNZ7wJZ-How Big is Disney World's Bus Fleet__2026-05-05"
HQ_CACHE = os.path.join(PROJECT_DIR, "v", "matcher-hq")
os.makedirs(HQ_CACHE, exist_ok=True)

FAILED = [
    ("zm2Cv4xuMf8", 186, 286),
    ("zm2Cv4xuMf8", 76, 182),
    ("LljC899Ocz4", 368, 383),
    ("fyW2ORMeLS4", 0, 38),
    ("AzwxpTf5kyM", 35, 97),
    ("AzwxpTf5kyM", 166, 206),
    ("7JFVPeXX4gk", 15, 78),
    ("7OwKVtZHFYY", 40, 65),
    ("5zxDFPPnIFE", 134, 147),
    ("aWUk_v4zTyM", 7, 59),
]

def find_youtube_url(video_id, project_dir):
    checkpoints = [
        os.path.join(project_dir, "checkpoint.json"),
        os.path.join(project_dir, "gap_fill", "checkpoint.json"),
    ]
    for cp_path in checkpoints:
        if not os.path.exists(cp_path):
            continue
        try:
            with gzip.open(cp_path, "rt", encoding="utf-8") as f:
                cp = json.load(f)
        except (gzip.BadGzipFile, OSError):
            try:
                with open(cp_path, "r", encoding="utf-8") as f:
                    cp = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                continue
        for r in cp.get("video_search", {}).get("search_results", []):
            if r.get("video_id") == video_id:
                return r.get("url") or r.get("youtube_url")
    return None

ok, fail = 0, []
for vid, ss, ee in FAILED:
    url = find_youtube_url(vid, PROJECT_DIR)
    if not url:
        print(f"SKIP {vid} ss={ss} ee={ee} — no URL found")
        fail.append((vid, ss, ee, "no_url"))
        continue
    output = os.path.join(HQ_CACHE, f"{vid}_1080p_{ss}_{ee}.mp4")
    # Remove height>=720 constraint, use best available up to 480p MP4
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio[ext=m4a]/best",
        "--download-sections", f"*{ss}-{ee}",
        "--merge-output-format", "mp4",
        "--no-part",
        "-o", output,
        url,
    ]
    print(f"{vid} ss={ss} ee={ee}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0 and os.path.exists(output):
        sz = os.path.getsize(output) // 1024
        print(f"OK ({sz}KB)")
        ok += 1
    else:
        err = result.stderr[-300:] if result.stderr else ""
        print(f"FAIL — {err[:150]}")
        fail.append((vid, ss, ee, err[:150]))

print(f"\n=== {ok}/{len(FAILED)} recovered, {len(fail)} still failing ===")
if fail:
    for vid, ss, ee, reason in fail:
        print(f"  {vid} ss={ss} ee={ee} — {reason}")
