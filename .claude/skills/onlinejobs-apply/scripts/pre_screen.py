"""Pre-screen OnlineJobs.ph editor listings with Python + regex.

Fetches all search-result pages (default: 10), extracts job URLs, dedupes by id,
skips ids already in history.json, then fetches each job's HTML and applies
hard rules (cap, primary-work, topic). Outputs a JSON list of qualified jobs
plus the per-job classification.
"""

import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

SEARCH_URL = (
    "https://www.onlinejobs.ph/jobseekers/jobsearch"
    "?jobkeyword=editor&skill_tags=&gig=on&partTime=on&fullTime=on&isFromJobsearchForm=1"
)
MAX_PAGES = 10
HISTORY = Path(__file__).resolve().parent.parent / "history.json"

JOB_URL_RE = re.compile(r"/jobseekers/job/([a-z0-9-]+-\d+)")
TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.I)
# Wage/salary text lives in <p class="fs-18"> directly after the <h3>WAGE / SALARY</h3> label.
SALARY_RE = re.compile(
    r'WAGE\s*/\s*SALARY\s*</h3>\s*<p[^>]*>\s*(.+?)\s*</p>',
    re.I | re.S,
)
# Job description lives in <p id="job-description" ...>...content...</p>.
OVERVIEW_RE = re.compile(
    r'<p[^>]*id="job-description"[^>]*>(.+?)</p>',
    re.I | re.S,
)
PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)(?:\s*[-–]\s*(\$?[\d,]+(?:\d+)?))?")
HOURS_RE = re.compile(r"(\d+)\s*HOURS PER WEEK", re.I)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

PRIMARY_WORK_KEYWORDS = (
    "video edit", "video editor", "youtube", "long-form", "long form",
    "short-form", "short form", "reels", "tiktok", "shorts",
    "podcast video", "social video", "premiere", "davinci resolve",
    "after effects", "final cut", "motion graphics", "b-roll",
    "pacing", "retention", "sound design", "color grading", "captions",
)
TOPIC_EXCLUSIONS = (
    "real estate", "realtor", "propertiesby", "property management",
    "mortgage", "housing market", "not youtube", "corporate documentary",
    "reaction content", "commentary on", "react to",
    "facebook ads", "meta ads", "ads creative", "ad creative",
    "tiktok ads", "youtube ads", "google ads", "ugc ads",
    "advertising creative", "marketing creative", "direct response ads",
    "performance ads",
)
NEGATIVE_TITLE_TOKENS = (
    "graphic designer", "social media manager", "virtual assistant",
    "copywriter", "seo specialist", "thumbnail designer", "uploader",
    "channel manager", "marketing manager", "creative director",
    "content strategist", "content manager", "proofreading", "photo editor",
)


def http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_job_ids(search_url: str, max_pages: int) -> list[str]:
    """Fetch search results. OnlineJobs.ph paginates by inserting an integer
    offset into the path: /jobseekers/jobsearch/30, /60, /90 ..."""
    seen: set[str] = set()
    ids: list[str] = []
    base, _, query = search_url.partition("?")
    for page in range(max_pages):
        offset = page * 30
        url = f"{base}/{offset}?{query}" if offset else f"{base}?{query}"
        try:
            html = http_get(url)
        except urllib.error.HTTPError as e:
            print(f"  offset {offset}: HTTP {e.code} - stop", file=sys.stderr)
            break
        except Exception as e:
            print(f"  offset {offset}: error {e} - stop", file=sys.stderr)
            break
        page_ids = JOB_URL_RE.findall(html)
        uniq_page = list(dict.fromkeys(page_ids))
        new = [i for i in uniq_page if i not in seen]
        if not new and page > 0:
            print(f"  offset {offset}: no new ids, stop paginating", file=sys.stderr)
            break
        seen.update(new)
        ids.extend(new)
        print(
            f"  offset {offset:3d}: {len(uniq_page)} unique ids, +{len(new)} new (total {len(ids)})",
            file=sys.stderr,
        )
    return ids


def cap_check(salary: str, hours: int | None) -> tuple[bool, str]:
    if not salary:
        return False, "no salary stated"
    text = salary.strip()
    if any(k in text.lower() for k in ["n/a", "tba", "open for discussion", "not given", "negotiable"]):
        return False, "needs salary review (ambiguous pay)"
    nums = PRICE_RE.findall(text)
    if not nums:
        return False, "needs salary review (no $ amount)"
    vals = []
    for a, b in nums:
        try:
            vals.append(float(a.replace(",", "")))
            if b:
                vals.append(float(b.replace(",", "").replace("$", "")))
        except ValueError:
            pass
    if not vals:
        return False, "needs salary review (unparseable)"
    upper = max(vals)
    lower = min(vals)
    low = text.lower()

    # Bare $-amount with no unit + a known HOURS PER WEEK. Heuristic:
    # small amounts (≤$30) are typical hourly rates; mid-range amounts
    # ($100-500) with full-time hours (30-40) are typical monthly salaries.
    has_unit = bool(re.search(r"(month|mo|/hr|hour|video|minute|week|ad|gig|creative)", low))
    if not has_unit and hours:
        if 100 <= upper <= 500 and 30 <= hours <= 45:
            verdict = upper < 500
            return verdict, f"USD {lower:g}-{upper:g}/mo (bare amount, inferred from range) ({'PASS' if verdict else 'over cap'})"
        weekly_hours = hours
        monthly = upper * weekly_hours * 4.33
        verdict = monthly < 500
        return verdict, (
            f"USD {lower:g}-{upper:g}/hr x {weekly_hours}h/wk = ~USD {monthly:.0f}/mo "
            f"({'PASS' if verdict else 'over cap'})"
        )

    if "per minute" in low or "per finished minute" in low or "per completed minute" in low:
        # Per-minute rates must be converted to per-video cost.
        # Assume a 15-min long-form finished video unless signals point to short-form.
        finished_min = 10 if any(k in low for k in ["short", "reel", "tiktok", "shorts"]) else 15
        per_video = upper * finished_min
        verdict = per_video < 20
        return verdict, (
            f"USD {lower:g}-{upper:g}/min x {finished_min}min = ~USD {per_video:.0f}/video "
            f"({'PASS' if verdict else 'over cap'})"
        )
    if "per video" in low or "/video" in low or "each video" in low or "per creative" in low or "/ad " in low or "per ad" in low or "per completed" in low or "per finished" in low or "per deliverable" in low:
        verdict = upper < 20
        return verdict, f"USD {lower:g}-{upper:g}/video ({'PASS' if verdict else 'over cap'})"
    if "per week" in low:
        # unknown total; treat as ambiguous
        return False, f"needs salary review (per week USD {lower:g}-{upper:g})"
    if "hour" in low or "/hr" in low or "/hour" in low:
        weekly_hours = hours if hours else 40
        monthly = upper * weekly_hours * 4.33
        verdict = monthly < 500
        return verdict, (
            f"USD {lower:g}-{upper:g}/hr x {weekly_hours}h/wk = ~USD {monthly:.0f}/mo "
            f"({'PASS' if verdict else 'over cap'})"
        )
    if "month" in low or "/mo" in low:
        verdict = upper < 500
        return verdict, f"USD {lower:g}-{upper:g}/mo ({'PASS' if verdict else 'over cap'})"

    # No unit anywhere — fall back to monthly interpretation per the skill rule
    # "WAGE / SALARY field without another unit = monthly". Bare amounts under
    # $30 are flagged as needs salary review (too low to be a monthly salary,
    # almost certainly per-hour / per-video). Bare amounts $30-$499 pass as
    # monthly. Anything >= $500 is rejected.
    if not hours:
        if upper < 30:
            return False, f"needs salary review (bare amount USD {lower:g}-{upper:g}, below monthly floor; likely per-hour or per-gig)"
        verdict = upper < 500
        return verdict, f"USD {lower:g}-{upper:g} (bare amount, assumed monthly) ({'PASS' if verdict else 'over cap'})"
    weekly_hours = hours
    monthly = upper * weekly_hours * 4.33
    verdict = monthly < 500
    return verdict, (
        f"USD {lower:g}-{upper:g}/hr x {weekly_hours}h/wk (inferred) = ~USD {monthly:.0f}/mo "
        f"({'PASS' if verdict else 'over cap'})"
    )


def primary_work_ok(title: str, overview: str) -> tuple[bool, str]:
    title_low = title.lower()
    if any(tok in title_low for tok in NEGATIVE_TITLE_TOKENS):
        return False, f"title matches non-video role ({[t for t in NEGATIVE_TITLE_TOKENS if t in title_low][0]})"
    blob = (title + " " + overview[:1500]).lower()
    hits = [k for k in PRIMARY_WORK_KEYWORDS if k in blob]
    if len(hits) < 2:
        return False, f"only {len(hits)} primary-work signal(s): {hits}"
    return True, f"primary-work signals: {hits[:4]}"


def topic_ok(title: str, overview: str) -> tuple[bool, str]:
    blob = (title + " " + overview[:1500]).lower()
    hits = [k for k in TOPIC_EXCLUSIONS if k in blob]
    if hits:
        return False, f"topic mismatch ({hits[0]})"
    return True, "topic OK"


def classify(job_id: str) -> dict:
    url = f"https://www.onlinejobs.ph/jobseekers/job/{job_id}"
    try:
        html = http_get(url)
    except Exception as e:
        return {"id": job_id, "url": url, "outcome": "blocked", "reason": f"fetch error {e}"}

    title_m = TITLE_RE.search(html)
    title = title_m.group(1).strip() if title_m else job_id
    # Strip site suffix
    title = re.sub(r"\s*-\s*OnlineJobs\.ph\s*$", "", title, flags=re.I)

    salary_m = SALARY_RE.search(html)
    salary_raw = salary_m.group(1).strip() if salary_m else ""
    salary = TAG_RE.sub("", salary_raw).strip()
    salary = WS_RE.sub(" ", salary)
    hours_m = HOURS_RE.search(html)
    hours = int(hours_m.group(1)) if hours_m else None

    overview_m = OVERVIEW_RE.search(html)
    overview_raw = overview_m.group(1).strip() if overview_m else ""
    overview = TAG_RE.sub(" ", overview_raw)
    overview = WS_RE.sub(" ", overview).strip()

    cap_pass, cap_msg = cap_check(salary, hours)
    primary_pass, primary_msg = primary_work_ok(title, overview)
    topic_pass, topic_msg = topic_ok(title, overview)

    if not cap_pass and "needs salary review" in cap_msg:
        outcome = "needs salary review"
    elif not cap_pass:
        outcome = "rejected"
    elif not topic_pass:
        outcome = "topic mismatch"
    elif not primary_pass:
        outcome = "rejected"
    else:
        outcome = "qualified"

    return {
        "id": job_id,
        "url": url,
        "title": title,
        "salary": salary,
        "hours_per_week": hours,
        "outcome": outcome,
        "cap": cap_msg,
        "primary": primary_msg,
        "topic": topic_msg,
        "overview_snippet": overview[:400],
    }


def load_history_skip_set() -> set[str]:
    if not HISTORY.exists():
        return set()
    try:
        data = json.loads(HISTORY.read_text(encoding="utf-8"))
    except Exception:
        return set()
    skip = set()
    for jid, rec in (data.get("applied") or {}).items():
        out = (rec.get("outcome") or "").lower()
        if (
            out.startswith("rejected")
            or out.startswith("draft ready")
            or out == "sent"
            or out.startswith("topic mismatch")
            or out.startswith("blocked")
        ):
            skip.add(jid)
    return skip


def main():
    skip = load_history_skip_set()
    print(f"history skip set: {len(skip)} ids", file=sys.stderr)

    ids = fetch_job_ids(SEARCH_URL, MAX_PAGES)
    print(f"collected {len(ids)} unique ids across pages", file=sys.stderr)

    new_ids = [i for i in ids if i not in skip]
    print(f"after dedupe vs history: {len(new_ids)} new ids", file=sys.stderr)

    results = []
    for jid in new_ids:
        r = classify(jid)
        results.append(r)
        print(
            f"  {r['outcome']:22s} {jid:60s} {r.get('cap','')}",
            file=sys.stderr,
        )

    qualified = [r for r in results if r["outcome"] == "qualified"]
    review = [r for r in results if r["outcome"] == "needs salary review"]

    out = {
        "total_collected": len(ids),
        "new_after_history": len(new_ids),
        "qualified": qualified,
        "needs_salary_review": review,
        "all": results,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
