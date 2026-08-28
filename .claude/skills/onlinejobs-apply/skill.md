---
name: onlinejobs-apply
description: Find affordable OnlineJobs.ph YouTube video-editor jobs, open each qualified post in its own Playwright tab, and prepare factual application letters for manual review. Never submits or sends applications.
allowed-tools:
  - AskUserQuestion
  - Read
  - Write
  - Bash
  - mcp__plugin_playwright_playwright__browser_navigate
  - mcp__plugin_playwright_playwright__browser_snapshot
  - mcp__plugin_playwright_playwright__browser_find
  - mcp__plugin_playwright_playwright__browser_click
  - mcp__plugin_playwright_playwright__browser_fill_form
  - mcp__plugin_playwright_playwright__browser_type
  - mcp__plugin_playwright_playwright__browser_wait_for
  - mcp__plugin_playwright_playwright__browser_evaluate
  - mcp__plugin_playwright_playwright__browser_take_screenshot
  - mcp__plugin_playwright_playwright__browser_tabs
---

# OnlineJobs.ph YouTube Editor Applications

Find low-cost YouTube editing jobs, open every selected job in a separate browser tab, and leave a tailored application letter in the application field for the user to review and send.

## Invocation

```text
/onlinejobs-apply
/onlinejobs-apply --max 5
/onlinejobs-apply <onlinejobs-search-url> --max 3
```

Default search URLs (paginated in priority order — see Step 3):

```text
https://www.onlinejobs.ph/jobseekers/jobsearch?jobkeyword=faceless&skill_tags=&gig=on&partTime=on&fullTime=on&isFromJobsearchForm=1
https://www.onlinejobs.ph/jobseekers/jobsearch?jobkeyword=editor&skill_tags=&gig=on&partTime=on&fullTime=on&isFromJobsearchForm=1
```

The `faceless` keyword returns mostly genuine faceless YouTube editor roles in the first 2 pages and is the primary source. The `editor` keyword still surfaces tier-1 long-form YouTube roles the `faceless` search misses, so run both — `faceless` first, `editor` second.

Default maximum: 5 jobs. Hard maximum: 10 jobs per run.

## Local History

Maintain `.claude/skills/onlinejobs-apply/history.json` with a per-job record. The file is the source of truth for which listings have already been drafted so repeated runs can skip them.

Schema:

```json
{
  "applied": {
    "<job_id>": {
      "url": "https://www.onlinejobs.ph/jobseekers/job/<slug>",
      "title": "...",
      "compensation": "...",
      "tier": 1,
      "drafted_on": "2026-08-02",
      "outcome": "draft ready | sent | rejected | needs salary review | topic mismatch | blocked | skipped"
    }
  }
}
```

The `job_id` is the trailing slug of the canonical job URL (for example `youtube-editor-long-form-1700815`).

Before any new run, Read `history.json`. In Step 3 and Step 4, skip any candidate whose `job_id` already appears under `applied` with `outcome` in `draft ready`, `sent`, `rejected`, `topic mismatch`, or `blocked`. Always show jobs that are `needs salary review` or `skipped` so the user can revisit them.

When a job reaches a final state in Step 6, Step 8, or Step 10, Read the file, update or insert the entry, and Write the file back atomically. Do not edit `applied` entries for jobs that were inspected but not drafted unless the user explicitly asks to record the rejection.

To re-draft a job, pass `--force <job_id>` or remove the entry from `history.json` by hand. Without `--force`, never open a tab for a recorded job.

## Non-Negotiable Safety Boundary

The skill prepares drafts only.

- Never click a final `Submit`, `Send`, `Send Application`, `Send Proposal`, `Apply Now`, `Confirm`, or equivalent control.
- Never invoke a final action through DOM evaluation, JavaScript, keyboard shortcuts, or form submission.
- A non-final link or button may be used only when its destination clearly opens or reveals the application form.
- If it is unclear whether a control opens the form or submits the application, stop on that tab and report `blocked: ambiguous apply control`.
- Never enter, read, store, or export login credentials or cookies.
- Never bypass a CAPTCHA, verification prompt, rate limit, or access control.
- Keep every completed application tab open for manual review.

## Qualification Rules

A post qualifies only when its primary work, compensation, and topic fit meet these rules.

### 1. The Primary Job Must Be Video Editing

Do not qualify a post merely because it mentions video editing, YouTube, reels, or editing software.

A post qualifies as a primary video-editing role when either:

1. The title clearly names a video-editing role; or
2. The description contains at least two strong primary-work signals:
   - recurring edited-video deliverables or a videos-per-week cadence;
   - editing raw footage into completed videos is a main responsibility;
   - required editing software or a required video portfolio;
   - detailed editing duties such as pacing, story structure, B-roll, captions, sound design, color, motion graphics, or retention editing.

It must also contain at least one target-format signal: YouTube, long-form video, Spanish video content, faceless video, Shorts, Reels, TikTok, podcast video, or social video.

Reject posts where video is incidental to a different primary role, including virtual assistant, social media manager, graphic designer, writer, copywriter, SEO specialist, thumbnail designer, photo editor, uploader, channel manager, or marketing manager.

Reject reaction-channel work. Posts that center on editing reaction videos, commentary on other channels, or other people's TV shows and movies do not qualify. The reviewer flagged `youtube-video-editor-reaction-content-1700947` as a poor qualifier for this reason; treat it as `rejected: reaction-channel work` rather than a Tier 1 YouTube editor.

Reject text/book/proofreading/photo-only editor roles even if the search result contains the word `editor`.

### 2. Compensation Must Be Below the Cap

Accepted compensation models:

- Monthly salary: strictly less than USD 500 per month.
- Per-video rate: strictly less than USD 20 per video.

Interpret the OnlineJobs.ph `WAGE / SALARY` field as monthly when it shows a dollar amount or range without another unit. Description text such as `per video`, `/video`, `each video`, `hourly`, or `per week` overrides that default.

For ranges, compare the upper bound with the cap. The entire range must qualify.

| Listed compensation | Result |
|---|---|
| `USD 315-415` under WAGE / SALARY | Qualifies as monthly; 415 < 500 |
| `USD 499/month` | Qualifies |
| `USD 500/month` | Reject |
| `USD 15/video` | Qualifies |
| `USD 20/video` | Qualified (at cap; flag in report) |
| `USD 15-20/video` | Qualified (upper bound at cap; flag in report) |
| `USD 10-25/video` | Reject because the upper bound exceeds the cap |
| Monthly and per-video rates both listed | Every disclosed applicable rate must be below its cap |
| `Paid per video, then salary` / `TBD` / `N/A` | `qualified (TBD/N/A)`; still draft, but flag N/A salary in the report and confirm the rate in the letter |
| Hourly, weekly, negotiable, missing, or unclear | `needs salary review` only when no rate is given at all and the post is not faceless-context; otherwise qualified |

Normalize explicit PHP or other currencies to USD only when the post gives an unambiguous conversion. Otherwise mark `needs salary review` rather than guessing an exchange rate.

### 3. Topic Fit

Prefer straightforward long-form YouTube production: recurring 8+ minute videos, source-footage or image research, basic motion graphics, B-roll, pacing, retention editing, sound design, and faceless/documentary workflows.

Real estate is a poor-fit topic. Posts centered on realtors, properties, housing markets, mortgages, neighborhood tours, or real-estate lead generation should be marked `topic mismatch` and excluded from automatic selection. Show them separately only when they pass every other rule so the user can override the topic preference.

Do not reject a post merely for talking-head footage or educational content; reject it when real estate is the central subject.

Calibration examples:

- `youtube-video-editor-1700757`: do not select. It is real-estate content and lists `USD 100 per video`, which exceeds the per-video cap even though it also says `USD 400 a month`. Its screening word is `suburbs`, which must still be detected during inspection.
- `youtube-editor-long-form-1700815`: Tier 1 target. It requires one 12–20 minute YouTube video daily, clip/image sourcing, basic motion graphics, and lists `USD 315-415` monthly. It also says no AI-generated editing or content; treat that as a job instruction, not a screening word.
- `long-form-video-editor-1706897`: Tier 3 target. Header says `350` but body clarifies `10 dollars per finished video`, paid weekly (1–2 per week). Pre-screen should detect the per-video unit and route to the per-video cap. Screening word: `hi Joona` (post addresses Joona directly, no formal screening token).
- `faceless-youtube-video-editor-ai-voice-over-for-history-documentary-channel-1708055`: Tier 3 target. `WAGE / SALARY` is `N/A`, body lists `USD 15-20 per completed video`. Mark as `qualified (at cap)`. Screening word: `ZMB` at the start of the message.
- `experienced-youtube-video-editor-long-form-faceless-content-1704316`: Tier 3 target. `WAGE / SALARY` is `Paid per video first, then salary` with no rate. Mark as `qualified (TBD/N/A)` and ask the agency's range in the letter. No screening word.
- `video-editor-for-youtube-channels-1680967` and `video-editor-for-youtube-faceless-channels-1702589`: same employer template (Contact Name: Not Given, Member since: May 20, 2026, Total Job Posts: 3, body text 90%+ overlapping). Draftable, but the duplicate-template rule should keep only the most recent. Both ask for content the text apply form cannot deliver (1-minute video intro, internet speed screenshot) — answer in text with a placeholder note and offer to follow up in chat.

### 4. Priority Ranking

Rank qualified posts in this order:

1. **YouTube Editor** — prioritize long-form videos, especially recurring channel work.
2. **YouTube Spanish Editor** — YouTube editing where Spanish fluency or Spanish-language content is central.
3. **Faceless Editor** — faceless YouTube/documentary/listicle/cash-cow style editing.
4. **Short-form Editor** — Shorts, Reels, or TikTok as the main output.

Within each tier, rank long-form work above short-form work, then newer posts above older posts. A mixed long-form and short-form job stays in its appropriate YouTube tier when long-form is the main responsibility.

## Workflow

### Step 1: Use the Saved Applicant Profile

Use these saved facts by default. Do not ask for them again unless the user asks to update them.

- Signature: David
- Experience types: long-form YouTube and faceless videos
- Software: DaVinci Resolve, CapCut, and Adobe Premiere Pro
- Do not claim Spanish editing or Spanish fluency unless David explicitly confirms it.

Faceless portfolios suitable for any relevant YouTube application:

- Disney: https://www.youtube.com/watch?v=PmffQaO4wDk&list=PLM-XBQXMzweHT44V1ozrKqD0OFcagoG-v
- War News AI Facecam: https://www.youtube.com/watch?v=-N1rEnGpt54&list=PLM-XBQXMzweGp-Fv2Y_JXyjdkdQGfIE8P

AI-oriented portfolios; use only when AI work is relevant and the post permits it:

- AI Health Shorts: https://www.youtube.com/watch?v=5w8iq-F9u3E&list=PLFeq__4053Ak
- AI Slideshow: https://www.youtube.com/watch?v=Bi-CicxZU5M&list=PLM-XBQXMzweFyd4dzrB_IZ7xhQpFvmvlz

Saved applicant artifacts (use only when the post asks for them — do not paste into letters that don't request):

- 1-minute video introduction: https://youtu.be/ZmXt8ypNaoM
- Internet speed screenshot: https://imgur.com/gallery/06-11-25-ocQt3oC

Prefer one or two highly relevant links rather than pasting every portfolio. Never use the AI-oriented portfolios when a post prohibits AI-generated editing or content.

Ask only for missing facts required by a specific post, such as timezone, availability, turnaround, rate, or paid-test preference. Keep new answers in the conversation unless the user explicitly asks to update this saved profile. Never invent missing experience, clients, results, metrics, tools, languages, or portfolio links. Use a visible `[ADD ...]` placeholder when a required fact is missing.

### Step 2: Verify the Browser Session

1. Open the default or supplied search URL in Playwright.
2. Take a snapshot and determine whether the site requires login.
3. If logged out, ask the user to sign in manually in the browser and resume only after they confirm.
4. Stop if a CAPTCHA, identity check, account warning, or rate-limit message appears.

### Step 3: Collect Search Results (Python + regex, paginate all pages)

1. Use `urllib.request` (or `requests` if available) via Bash to fetch every page of search results. The default search returns 30 of 300 jobs; without pagination the run only sees the newest 30 listings.
2. Pagination targets, in order:
   - `jobkeyword=faceless` → 2 pages (60 listings). The first 2 pages are mostly genuine faceless YouTube editor roles; pages 3+ are noisy.
   - `jobkeyword=editor` → 2 pages (60 listings). Surfaces tier-1 long-form YouTube roles the `faceless` search misses.
   - If the user passes a custom search URL, use that in place of `editor` and keep the `faceless` first.
   Try the URL pattern `&page=N` first. If that returns the same HTML as page 1, fall back to scraping pagination links (`?page=2`, `?page=3`, ...) from the rendered HTML. Stop when a page returns no new job IDs or when the per-keyword page budget is exhausted.
3. From every fetched page, extract every `/jobseekers/job/<slug>-<id>` URL with a single regex pass: `r'/jobseekers/job/([a-z0-9-]+-\d+)'`.
4. Deduplicate by `<id>`. Drop any `<id>` already present in `history.json` with `outcome` in `draft ready | sent | rejected | topic mismatch | blocked`. Keep entries marked `needs salary review` or `skipped` so the user can revisit them.
5. Cache each fetched search page's HTML under `.playwright-mcp/cache/oj-<keyword>-<page>.html` (or `/tmp/oj-<keyword>-<page>.html` on this machine) so re-runs are cheap. Skip cache if older than 1 hour.
6. Only open Playwright after the Python pre-screen (Step 3.5) has narrowed candidates. Use Playwright DOM evaluation only to read page data — never click controls or submit forms during inspection.

### Step 3.5: Python Pre-Screen (regex on each candidate post)

Before opening any browser tab, fetch each candidate job URL with `urllib.request` and extract the title, `WAGE / SALARY`, and `JOB OVERVIEW` via regex:

```python
import re, urllib.request

JOB_URL_RE  = re.compile(r'/jobseekers/job/([a-z0-9-]+-\d+)')
TITLE_RE    = re.compile(r'<title>([^<]+)</title>', re.I)
SALARY_RE   = re.compile(r'WAGE\s*/\s*SALARY\s*([^|]{0,200})', re.I)
OVERVIEW_RE = re.compile(r'JOB OVERVIEW(.{0,3500}?)(?:SKILL REQUIREMENT|ABOUT THE EMPLOYER|$)', re.S | re.I)
PRICE_RE    = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)(?:\s*[-–]\s*([\d,]+(?:\d+)?))?')

def cap_ok(salary_text: str) -> tuple[bool, str]:
    if not salary_text: return (False, "no salary")
    nums = PRICE_RE.findall(salary_text)
    if not nums: return (False, "needs salary review")
    upper = max(float(n[1] or n[0]) for n in nums)
    if 'month' in salary_text.lower() or 'mo' in salary_text.lower():
        return (upper < 500, f"USD {upper}/mo")
    if any(k in salary_text.lower() for k in ['per video', '/video', 'each video', 'per week']):
        # per-week needs hours conversion (default 40)
        return (upper < 20, f"USD {upper}/video")
    if 'hour' in salary_text.lower() or 'hr' in salary_text.lower():
        # assume 40 hrs/week for monthly equivalent
        return (upper * 40 < 500, f"USD {upper}/hr (~USD {int(upper*40*4.33)}/mo)")
    return (False, "needs salary review")
```

Apply hard rules in Python before opening any tab:

1. **Cap rule** — `cap_ok(salary_text)` must return `(True, ...)`. Otherwise mark `needs salary review` or `rejected: compensation over cap`. For the faceless path, N/A, TBD, and "Paid per video, then salary" qualify as `qualified (TBD/N/A)` and proceed to Step 4 with a flag.
2. **Primary-work rule** — the first 600 chars of `JOB OVERVIEW` (case-insensitive) must contain at least two of: `video edit`, `video editor`, `youtube`, `long-form`, `short-form`, `reels`, `tiktok`, `shorts`, `podcast video`, `social video`, `premiere`, `davinci`, `after effects`, `motion graphics`, `b-roll`, `pacing`, `retention`, `sound design`, `color grading`, `captions`, `faceless`, `documentary`, `voice-over`, `voiceover`, `elevenlabs`, `b-roll sourcing`. When the search keyword is `faceless`, `faceless` in the title alone counts as one signal. Otherwise mark `rejected: not primarily video editing`.
3. **Duplicate-template detection** — capture `Business or Contact Name` + `Member since` from the about-the-employer block. If two candidates share the same templated body text (first 800 chars of `JOB OVERVIEW` are ≥90% overlapping) and the same employer signature, mark the later one `rejected: duplicate template`. The reviewer flagged `video-editor-for-youtube-channels-1680967` and `video-editor-for-youtube-faceless-channels-1702589` as the same employer template (Contact Name: Not Given, Member since: May 20, 2026, Total Job Posts: 3) — keep only the most recent of the two.
4. **Topic exclusions** — reject (mark `topic mismatch`) if any of these appear in the title or first 600 chars of overview: `real estate`, `realtor`, `propertiesby`, `property management`, `mortgage`, `housing market`, `not youtube`, `corporate documentary`. Reject reaction-channel work (overview contains `reaction content`, `commentary on`, `react to`). Do not auto-reject listings that say "no experience needed", "AI-based", or "template-driven" — those are still draftable when the rest of the post describes a recurring video editor role; flag the low-quality profile in the report instead.
5. **Outcome shortlist** — keep only candidates whose outcomes are `qualified`, `qualified (N/A salary)`, `qualified (TBD/N/A)`, `qualified (at cap)`, or `needs salary review`. Pass those to Step 4 for browser inspection of the full post and screening-word capture.

Cache each fetched job page's HTML under `/tmp/oj-job-<id>.html` for re-use.

### Step 4: Inspect and Classify Posts

Use one temporary inspection tab and process candidates sequentially.

For each candidate:

1. Navigate the inspection tab to the job URL.
2. Read the entire post, not only the title or search-card summary. Extract the title, full description, `WAGE / SALARY`, employer/name when visible, posting age, topic, responsibilities, required tools, output format, cadence, language, application instructions, and any screening word or phrase.
3. Apply the primary-work test, compensation test, topic-fit test, and priority ranking exactly as defined above.
4. Record one of:
   - `qualified`;
   - `qualified (N/A salary)` — passed every other rule but compensation is missing; flag in the report and ask the user to confirm in the letter;
   - `qualified (TBD/N/A)` — paid-per-video model with no published rate, or "paid per video, then salary";
   - `qualified (at cap)` — per-video rate at exactly $20 (or per-video range whose upper bound is $20);
   - `rejected: not primarily video editing`;
   - `rejected: compensation over cap`;
   - `rejected: duplicate template` — same employer + ≥90% overlapping body as another candidate;
   - `topic mismatch`;
   - `needs salary review`;
   - `blocked or unavailable`.
5. Stop the entire run if the site presents anti-bot or account verification.

Do not leave rejected posts open. Close only the skill-created temporary inspection tab after classification.

### Step 5: Capture Screening Words and Instructions

Most posts contain a proof-of-reading instruction. Search the full description for wording such as:

- `put/include/write/mention the word ... in your application`;
- `start/end your application with ...`;
- `use ... as the subject line`;
- `answer these questions` or `include the following`;
- specific portfolio, sample, availability, trial, or software instructions.

Record the exact instruction and exact token, preserving spelling, capitalization, punctuation, and requested location. Do not infer a token when none is present.

Apply it exactly:

- If the post says to mention a word anywhere, include it naturally and exactly once in the letter.
- If it says to start or end with a word or phrase, place it at that exact boundary.
- If it requests a subject line, fill the subject field only when one exists; otherwise mark `needs user input`.
- If it asks questions, answer only from the supplied applicant profile and use visible placeholders for missing facts.
- If it bans AI-generated editing or content, do not claim or suggest AI use; this is a compliance instruction, not a screening token.

Before marking a draft ready, compare the filled form against the recorded instruction and verify that every required word, answer, link, and placement is present.

### Step 6: Present the Qualified List

Show a compact table before drafting:

```text
#  Tier  Job title  Compensation  Long-form  Posted  Status
1  1     ...        USD 415/month   Yes        2d      qualified
2  3     ...        USD 15/video    Yes        1d      qualified (at cap)
3  3     ...        USD 10/video    Yes        3d      qualified
4  3     ...        N/A             Yes        1d      qualified (N/A salary)
5  3     ...        paid per video  Yes        5d      qualified (TBD/N/A)
```

Flag any `qualified (N/A salary)`, `qualified (TBD/N/A)`, or `qualified (at cap)` rows explicitly so the user can decide whether to include them. Show rejected duplicates (`rejected: duplicate template`) under a separate "Filtered" section. Ask the user to choose specific jobs or `all qualified`. Never exceed the requested maximum or the hard cap of 10.

### Step 7: Open One Tab Per Selected Post

For each selected qualified post, sequentially:

1. Create a new tab.
2. Navigate to the canonical job URL.
3. Verify the title and description match the classified post.
4. Find the application section by snapshot and accessible text.
5. **Click the `APPLY FOR THIS JOB` button** to navigate to the apply form page at `/apply`. This is a navigation, NOT a submission — it consumes 1 apply-point and opens the actual editable form on a separate URL. The page will load with subject + message-body fields available.
6. If the form cannot be reached (e.g., site error, expired post), leave the tab open and mark it blocked.

Never reuse a completed application tab for another job.

### Step 8: Draft the Application Letter

Write a distinct 140-220 word letter for each post.

The letter must:

- include every exact screening word, phrase, requested answer, and placement instruction found in Step 5;
- directly reference the exact YouTube format and one or two real deliverables from the post;
- emphasize long-form experience first when relevant;
- emphasize Spanish editing only when the applicant profile truthfully supports it;
- emphasize faceless storytelling, B-roll, pacing, retention, and sound design only when supported;
- mention short-form secondarily when the post includes both formats;
- include only supplied portfolio links and factual experience;
- be concise, natural, and ready for the user to edit or send;
- avoid generic praise, fake familiarity, unsupported metrics, and claims that the job's low budget is attractive.

Suggested structure:

```text
Hi [name or team],

[One sentence naming the exact role and format.]

[Two short paragraphs connecting truthful experience/tools to the post's main deliverables, with long-form YouTube first.]

[Portfolio and availability/turnaround.]

[Optional paid-test sentence.]

[Signature]
```

### Step 9: Fill but Never Send

1. Fill only the cover-letter, message, or application textarea using Playwright form tools.
2. Do not fill optional fields with guessed information.
3. Take a snapshot or screenshot and confirm the full draft remains visible in the field.
4. Do not focus, click, or activate the final submission control.
5. Leave the tab open.

### Step 9.5: OnlineJobs.ph Apply-Form Page (separate URL, not the job page)

The OnlineJobs.ph apply form is a **separate page** that loads after the user clicks `APPLY FOR THIS JOB` on the job listing. The job page itself only contains hidden inputs and the submit button — it does NOT have subject or message-body fields. The actual editable form lives at `/apply` after the button is clicked.

**Workflow:**

1. On the job page, find the `APPLY FOR THIS JOB` button (the visible submit button in `<form action="/apply">`).
2. Click the button. This navigates to the apply form page at `https://www.onlinejobs.ph/apply` (consumes 1 apply-point). This is NOT a final submission — it is a navigation.
3. The apply form page renders the subject field and message-body textarea. Fill both with the prepared draft.
4. **Never click the final submit button** (`Send Application` or equivalent) on the apply form page. Leave the tab open for the user to review and submit.

**Common confusion to avoid:**

- The textareas `#noteText` and the placeholder `"Kindly elaborate more on the reason here"` on the **job page** live inside a hidden Bootstrap modal `<div id="myModal-log" class="modal fade" style="display: none">`. **This is the BOOKMARK modal, NOT the apply form.** Do not fill it; do not assume its fields are the application letter.
- The job page's `<form action="/apply">` has only hidden inputs (`csrf-token`, `contact_email`, `back_id`, `job_id`) and the `APPLY FOR THIS JOB` submit button. It does not have subject or message fields. Clicking the button navigates to the actual apply form page.
- After clicking APPLY FOR THIS JOB, the page URL changes to `https://www.onlinejobs.ph/apply`. That page is where the subject + message-body fields live and where the draft should be filled.

### Step 10: Report Results

End with:

```text
Tab  Job title  Compensation  Status
2    ...        USD 415/month   draft ready
3    ...        USD 15/video    draft ready
4    ...        unclear      needs salary review
```

Allowed statuses: `draft ready`, `draft ready (N/A salary)`, `draft ready (TBD)`, `draft ready (at cap)`, `needs user input`, `needs salary review`, `blocked`, or `skipped`.

## Failure Handling

- Expired or closed post: mark `blocked or unavailable` and continue.
- Missing application textarea: leave the job tab open and mark `blocked`.
- **0 apply-points quota**: apply-points are consumed by the APPLY FOR THIS JOB click (not by SEND EMAIL). The quota counter is informational on the /apply page; the editable form fields are reachable from any /apply load. If you genuinely cannot reach /apply (e.g., APPLY FOR THIS JOB returns an error or redirects to a quota-exhausted message), open tabs anyway, save drafts to `pending_drafts.json`, do NOT click APPLY FOR THIS JOB, and mark each tab `blocked: 0 apply-points` so the user can revisit after the daily reset.
- **Bookmark modal `#myModal-log` is not the apply form.** Do not assume its textareas are the application letter fields; they are for saving notes about the job to the user's bookmarks.
- Navigation failure: report it once; do not retry repeatedly.
- Session expires: stop and ask the user to log in manually.
- More than 10 requested jobs: refuse the excess and process at most 10.
- Any sign that a click could send an application: do not click it.
