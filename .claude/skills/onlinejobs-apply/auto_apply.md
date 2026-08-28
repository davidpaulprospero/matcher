# onlinejobs-apply — Auto-Apply Cron Workflow

Recurring cron workflow for the onlinejobs-apply skill. Runs every 3 hours (durable, .claude/scheduled_tasks.json). Picks up to 3 qualified candidates, opens each in a browser tab, fills the /apply form, leaves SEND EMAIL for manual click.

This workflow is invoked by a CronCreate prompt — do NOT ask for confirmation when the cron fires. Treat the fire as an automatic user invocation.

## Goal

For each run: open up to 3 /apply tabs (one per qualified candidate), pre-filled and ready for manual SEND EMAIL.

## Hard Constraints (NON-NEGOTIABLE)

These override any "fully automated" intent. Violating any of these is a hard failure.

1. **NEVER click** `SEND EMAIL`, `Apply Now`, `Submit`, `Confirm`, `Send Application`, `Send Proposal`, or any equivalent final-submit control. The OJ /apply form's final action is the user's, not ours.
2. **NEVER auto-apply** to jobs that fail hard rules (cap, primary-work, topic).
3. **NEVER read, store, or export** login credentials or cookies.
4. **NEVER bypass** CAPTCHA, verification, rate limits, or access controls.
5. **CAP at 3 candidates per run.** Do not exceed this even if more qualify.
6. **FILL via browser_evaluate** using the React native setter (`HTMLInputElement.prototype.value` setter) + `dispatchEvent('input')`. Do NOT use `browser_type` or `browser_fill_form` — both fire keyboard events that can trigger OJ's auto-submit (per memory `feedback_oj_form_autosubmit.md`).

## Workflow Per Run

### Step 1: Pre-screen via Python

Run from the skill directory:

```bash
cd "D:/_Projects/voiceover-matcher-dev/.claude/skills/onlinejobs-apply" && python scripts/pre_screen.py 2> /tmp/oj-auto.log > /tmp/oj-auto.json
```

Read `/tmp/oj-auto.json`. Extract `qualified` and `needs_salary_review` lists.

If pre-screen errors or returns 0 qualified, stop and report the issue. Do not open any tabs.

### Step 2: Apply Hard Rules (Browser Inspection Required)

Python pre-screen alone is insufficient — its primary-work signals are coarse, and it does not check screening-word specificity, topic fit in detail, or duplicate-template overlap. For each candidate that survives the cap rule:

1. Navigate to the job URL in a Playwright inspection tab.
2. Read the full description (not just title/search-card).
3. Apply the same rules used in `onlinejobs-apply` skill Steps 3.5–5:
   - **Cap rule:** per-video rate must be `<$20`, or monthly rate must be `<$500`. Per-minute rates convert to per-video at 15-min long-form (10-min short-form). Per-hour rates convert at 40h/wk × 4.33.
   - **Primary-work rule:** title must name a video-editing role, OR description must contain ≥2 strong primary-work signals + ≥1 target-format signal (YouTube, long-form, faceless, Shorts, Reels, TikTok, podcast video, social video).
   - **Topic exclusions (REJECT):** real estate, realtor, propertiesby, mortgage, housing market, ads/marketing creative (Facebook/Meta/TikTok/Youtube ads, ad creatives, performance ads), reaction content, commentary on, react to.
   - **Format exclusions (REJECT per memory `feedback_oj_no_shortform_no_podcasts.md`):** short-form-only or podcast-only roles.
   - **Duplicate-template detection:** same employer (Contact Name + Member since) with ≥90% overlapping body as another candidate → keep the most recent only.
4. Capture any screening word (e.g., "STORY BEFORE EFFECTS", "favorite food", "ZMB"). Preserve exact spelling/capitalization/punctuation.

### Step 3: Pick Top 3 Qualified Candidates

Sort by tier:
- Tier 1: Long-form YouTube (recurring channel work, ≥8 min videos)
- Tier 2: YouTube Spanish / Faceless documentary
- Tier 3: Faceless / Short-form

Within tier, sort by `job_id` descending (newest first — `job_id` ends in a numeric suffix; descending picks highest = newest).

Take top 3 (or fewer if <3 qualify). Skip any `job_id` already in `history.json` with `outcome` starting with `draft ready`, `sent`, `rejected`, `topic mismatch`, or `blocked` (the patched `load_history_skip_set` already does this).

### Step 4: Open Tabs and Fill Forms

For each of the (up to) 3 selected candidates, in sequence:

1. **Open a new browser tab** via `mcp__plugin_playwright_playwright__browser_tabs action=new url=<job_url>`.
2. **Verify the job page loaded.** Take a snapshot.
3. **Click APPLY FOR THIS JOB** (the button is the only form-submission control on the job page; this navigates to `/apply` and consumes 1 apply-point). Verify the new URL is `https://www.onlinejobs.ph/apply` and the page title is `Contact | OnlineJobs.ph`.
4. **Verify form fields exist:**
   - `document.querySelector('input[name="info[subject]"]')` must be visible.
   - `document.querySelector('textarea[name="info[message]"]')` must be visible.
   - If either is missing, mark `blocked: form fields not visible` in history.json, close the tab, and skip.
5. **Fill subject + message via browser_evaluate** using the React native setter pattern (see Hard Constraints #6). Do NOT use `browser_type` or `browser_fill_form` on this page.
6. **Leave SEND EMAIL** unclicked. The user reviews and sends manually.

For each tab, the message body MUST:
- Include any captured screening word at the exact location specified by the post (e.g., "STORY BEFORE EFFECTS" at start, "CALM MOTION" at end).
- Use the saved applicant profile: signature **David**, software **DaVinci Resolve (primary), CapCut, Adobe Premiere Pro**, software comfort with **After Effects** for motion-design roles only, time zone **PHT (UTC+8), full-time**.
- Embed 1–2 most-relevant portfolio playlists:
  - Disney: https://www.youtube.com/watch?v=PmffQaO4wDk&list=PLM-XBQXMzweHT44V1ozrKqD0OFcagoG-v
  - War News AI Facecam: https://www.youtube.com/watch?v=-N1rEnGpt54&list=PLM-XBQXMzweGp-Fv2Y_JXyjdkdQGfIE8P
  - AI Health Shorts: https://www.youtube.com/watch?v=5w8iq-F9u3E&list=PLFeq__4053Ak
  - AI Slideshow: https://www.youtube.com/watch?v=Bi-CicxZU5M&list=PLM-XBQXMzweFyd4dzrB_IZ7xhQpFvmvlz
  - Pick Disney + War News for long-form faceless YouTube; AI Health Shorts + AI Slideshow for motion-design/short-form; never use AI-oriented portfolios when a post prohibits AI.
- Quote turnaround as **<24 hours per finished piece** (per memory `feedback_oj_turnaround_under_24h.md`).
- Use `[ADD: <thing>]` placeholder for any required fact David hasn't provided (e.g., favorite food, timezone-specific availability, paid-test preference). Do NOT invent facts.
- Never claim Spanish editing or Spanish fluency unless David explicitly confirms it.

### Step 5: Update history.json

For each tab opened, add an entry to `.claude/skills/onlinejobs-apply/history.json` under `applied`:

```json
{
  "<job_id>": {
    "url": "https://www.onlinejobs.ph/jobseekers/job/<slug>",
    "title": "<title>",
    "compensation": "<salary>",
    "tier": <1|2|3>,
    "drafted_on": "<YYYY-MM-DD>",
    "outcome": "draft ready"
  }
}
```

For each candidate inspected but NOT drafted (rejected), add an entry with `outcome` like:

```json
"outcome": "rejected: <one-line reason>"
```

Common rejection reasons (use the closest match):
- `rejected: compensation over cap`
- `rejected: short-form only`
- `rejected: podcast only`
- `rejected: ads/marketing creative role`
- `rejected: real estate topic`
- `rejected: reaction content`
- `rejected: duplicate template`
- `rejected: insufficient detail`
- `topic mismatch`
- `needs salary review`
- `blocked: <reason>`

Write `history.json` back atomically after all updates.

### Step 6: Update pending_drafts.json

For each tab opened, add an entry to `.claude/skills/onlinejobs-apply/pending_drafts.json`:

```json
{
  "job_id": "<job_id>",
  "url": "<url>",
  "tier": <1|2|3>,
  "compensation": "<salary>",
  "screening_word": "<word or null>",
  "apply_via": "OJ /apply form (filled on tab N - awaiting manual SEND EMAIL)",
  "subject": "<subject line filled>",
  "message_body_excerpt": "<first 200 chars of message>",
  "status": "filled in /apply form (tab N), awaiting manual submit"
}
```

Overwrite `pending_drafts.json` with the new state.

### Step 7: Log the Run

Append a line to `.claude/skills/onlinejobs-apply/auto_apply_log.json`:

```json
{
  "runs": [
    {
      "timestamp": "<ISO 8601>",
      "qualified_count": <int>,
      "drafted_count": <int>,
      "drafted_ids": ["<job_id>", ...],
      "rejected_count": <int>,
      "rejected_ids": ["<job_id>", ...],
      "errors": ["<error or empty>"]
    }
  ]
}
```

If `auto_apply_log.json` does not exist, create it with `{"runs": []}` and append.

### Step 8: Report to User

At the end of the run, print a concise summary (no need to ask questions):

```
Auto-apply run <timestamp>:
- Pre-screen: <N> qualified → <M> passed hard rules
- Drafted: <count> tabs opened (job_id1, job_id2, job_id3) — awaiting manual SEND EMAIL
- Rejected: <count> (reasons)
- Apply-points consumed (this run): <count>
- Tabs still open: <list>

Manual action: review each tab, fill any [ADD: ...] placeholders, click SEND EMAIL on the ones you want to send.
```

If running silently during a cron fire (no user prompt to respond to), write the summary to `/tmp/oj-auto-summary.txt` instead of printing.

## Failure Handling

- **Pre-screen network errors** (HTTP 403, 429, etc.): write error to log, exit silently. Next cron fire will retry.
- **Apply-points quota hit** (OJ shows quota=0 / "Daily limit reached"): STOP opening new tabs. Mark all remaining candidates as `skipped: apply-points quota exhausted`. Note quota exhaustion in the log.
- **Already-applied status** ("Applied" button instead of "APPLY FOR THIS JOB"): mark `already applied` in history.json, close the tab, skip.
- **Bookmark modal confusion** (`#myModal-log` with `#noteText` "Kindly elaborate more on the reason here"): this is the bookmark feature, NOT the apply form. Never paste drafts here.
- **Auto-submit detected** (after filling, the page shows "Applied" or success message): STOP. Note in log. Use OJ Messages to fix any issues. Do NOT continue opening tabs for this session — wait for next cron fire.
- **CAPTCHA / verification prompt**: STOP. Write error. Do NOT attempt to bypass.

## Files Referenced

- `.claude/skills/onlinejobs-apply/scripts/pre_screen.py` — pre-screen
- `.claude/skills/onlinejobs-apply/history.json` — past runs + skip set
- `.claude/skills/onlinejobs-apply/pending_drafts.json` — current drafts awaiting manual submit
- `.claude/skills/onlinejobs-apply/auto_apply_log.json` — per-run history (new)
- `/tmp/oj-auto.log`, `/tmp/oj-auto.json`, `/tmp/oj-auto-summary.txt` — temporary run artifacts

## Cron Setup

Created by the user via `CronCreate` with:
- cron: `17 */3 * * *` (every 3 hours, offset 17 min past the hour to avoid the :00 thundering herd)
- recurring: true
- durable: true
- prompt: "Run the onlinejobs-apply auto-apply workflow per the instructions in `.claude/skills/onlinejobs-apply/auto_apply.md`. Do not ask for confirmation — this is an automatic cron fire."

**Auto-expire note:** CronCreate recurring tasks auto-expire after 7 days. Re-create the cron weekly (or after 7 days) to keep the workflow running indefinitely.