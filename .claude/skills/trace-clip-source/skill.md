# Skill: trace-clip-source

Trace a viral or reposts clip back to its earliest/most original known source.
Builds a chain-of-custody through repost layers (raw → first repost → news coverage
→ later viral reposts).

## When to Use

Use `trace-clip-source` when:
- You have a viral video (YT Shorts, TikTok, IG Reel) and need to find the original
- A clip is circulating without source attribution and you need to identify the creator
- You suspect a clip was reposted by a compilation channel and want the raw trucker/creator version
- You need a chain-of-custody for licensing, legal, or editorial reasons

**NOT for:**
- Verifying a video's factual claims (use web research / Perplexity)
- Downloading a video (use yt-dlp / gws_drive)
- Identifying people in a video from their face (limited tools, often not possible)

## Invocation

```
/trace-clip-source <starting-url-or-id> [--investigation-dir <path>]
```

Examples:
```bash
# Start with a YT Shorts
/trace-clip-source "https://www.youtube.com/shorts/uu6-WsmO2VU"

# With investigation dir for saving findings
/trace-clip-source "https://www.youtube.com/shorts/uu6-WsmO2VU" --investigation-dir "D:/projects/clip_b/"
```

---

## Core Methodology: Trace Through Repost Layers

Viral clips almost never originate at the viral post. They flow through layers:

```
[Raw original creator] → [Early repost/compilation] → [News/blog coverage] → [Viral repost]
```

The job is to walk **backwards** from the viral post to find the earliest reachable
source, then identify the layer above it.

### Phase 0 — Scan for Source Credit (Early-Out)

**Before doing anything else, read the viral upload's description, pinned comment,
and author bio.** If the uploader names the source, you can skip Phases 2–5 and
jump straight to Phase 6 verification.

Search the description text (and pinned comment if any) for credit strings matching:
```regex
(?i)(courtesy|credit|source|original|via|repost|from)\s*[:\-]?\s*\S+
```

Common patterns:
- `"Video Courtesy of @username (on instagram)"`
- `"Credit: @username on TikTok"`
- `"Original: <url>"`
- `"DM for credit / removal"` (indicates they're acting as a republisher)

**If a credit is found, record it and jump to Phase 6** — your job becomes
loading the credited account, finding the matching Reel, and confirming its
post date precedes the viral upload.

This phase exists because branded republishers (dashcam brands, news outlets,
licensed-content accounts) preserve credits by name, while anonymous viral
reposters strip them. The presence of a credit collapses the investigation
from 6 phases to 1.

### Phase 1 — Lock Down the Viral Post

Confirm what you have:
1. Capture full metadata via yt-dlp:
   ```bash
   yt-dlp --write-info-json --skip-download -o "%(id)s" "<url>"
   ```
2. Save to `clip_<x>_investigation/`:
   - `<id>.info.json` — full yt-dlp metadata
   - Comments JSON (`--write-comments` if available, or fetch separately)
   - Frames at 1fps and 8fps (for plate crops, face crops)
3. Record: uploader channel, upload date, duration, view count, engagement signals
   (likes/dislikes ratio, follower count). Compilation channels typically have
   5K-50K followers; viral accounts have 100K+; original creators usually have <5K.

**Two kinds of republishers — distinguish early:**

| Republisher type | Credit behavior | Detection signal |
|---|---|---|
| **Branded republisher** (dashcam brands, news outlets, licensed-content networks) | Preserves credits by name — has legal/ethical skin in the game | Channel bio describes a brand/product. Watermark on the video. YT description names the source. |
| **Anonymous viral reposter** (compilation channels) | Strips credits. "All rights belong to respective owners" disclaimer. | Disclaimer text. Cropped-out watermarks. Generic channel name ("X Videos", "Road Rage Daily"). |

If the closest repost layer is a **branded republisher**, Phase 0 likely caught
the credit. If it's **anonymous**, you need full Phases 2–6.

### Phase 2 — Mine Comment Threads (highest-signal step)

Comment threads on viral posts contain hyper-local knowledge that never appears
in news or web search. People often volunteer:
- Exact location ("I KNOW WHERE THIS IS! 291 and I-70 interchange")
- Identifications ("that lady lives in my town")
- Source attribution ("original: <link>")
- Witness accounts ("I saw her damaged car near...")

**Always run this BEFORE web search.** Use Python + regex on the comments JSON:

```python
import json, re
with open('clip_comments.info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
comments = data['comments']

# Find URLs in comments — usually only 0-3 exist even in 5000+ comment threads
url_re = re.compile(r'https?://|www\.', re.I)
for c in comments:
    if url_re.search(c.get('text', '') or ''):
        print(c.get('like_count'), c.get('author'), c['text'][:300])

# Find location / recognition / attribution
for c in sorted(comments, key=lambda x: -x.get('like_count', 0))[:500]:
    text = (c.get('text') or '').lower()
    if any(k in text for k in ['my town', 'i know', 'recognize', 'i saw', 'source', 'original', 'tiktok', 'facebook', 'instagram', 'reddit', 'credit']):
        print(c.get('like_count'), c.get('author'), c['text'][:300])
```

**Important:** Even low-engagement comments (1-2 likes) can be the most informative
because only people who actually recognize the location bother commenting that way.

### Phase 3 — Reverse Image Search Key Visual Elements

If the clip has distinctive visual identifiers (license plate, face, badge, sign),
extract crops and reverse-search them. License plates are the single best signal —
they're unique and survive re-encoding better than faces.

**Tools (in order of preference):**
1. **Yandex Images** (best for license plates, faces, similar-looking duplicates)
2. **Google Lens** via images.google.com (best for finding pages the image appears on)
3. **TinEye** (best for finding oldest indexed version)

**Browser workflow (Playwright MCP):**
```javascript
// Navigate to images.google.com
// Click "Search by image" button
// Upload the plate crop
// Extract structured results from the page
```

**Expect zero results for original-source on the plate alone.** That's a useful
negative — it means the clip hasn't been indexed elsewhere with that plate visible.
Compile accounts usually crop the plate out, so a reverse search returning only the
repost page confirms the chain.

**Cross-validate:** If Google AI Overview claims plate state ("Pennsylvania", etc.),
treat it as unverified. AI Overview hallucinates plate states routinely. Trust only
the structured visual matches, not the AI summary.

### Phase 4 — Direct Plate / Identifier Text Search

Run Google search for the exact plate string with quotes:
```
"LC6 XOG" — zero hits means plate is not indexed anywhere
"LC6 XOG" site:reddit.com — try with site: filter per platform
"LC6 XOG" missouri — combine with location
```

If a search returns zero results, that's informative: the original isn't on the
indexed public web. Document the negative result and move on.

### Phase 5 — Search News Coverage of the Same Event

Use targeted queries combining event-specific terms:
```
"Lee's Summit" OR "Independence" missouri dashcam truck malibu merge bumper
"I-70" "291" Independence Missouri truck semi crash
"Peterbilt" "Chevy Malibu" bumper
```

**Verify the news story matches the viral clip:**
- Same vehicle types (silver Chevy Malibu, Peterbilt semi)
- Same incident type (rear bumper stripped off during merge)
- Same date range (±1 month of the viral upload)
- Same location (use the location from Phase 2 commenter intel)

If you find a matching news article, the embedded video URL or "source" credit is
your next-best lead. News articles often embed the original clip directly.

**GM Authority, Jalopnik, TweakTown, Carscoops** are common embedders of viral
car-incident videos. They typically embed via an `<iframe>` from the source
(Instagram, YouTube, Twitter). Extract the iframe `src` — that's the next link
in the chain.

### Phase 6 — Trace Through Compilation Chains

Most viral clips pass through 2-4 repost layers:

```
Raw creator → Compilation account (IG/FB/YT) → News/blog coverage → Viral repost
```

Each layer may or may not credit the layer above it. Compilation accounts almost
never do. News coverage usually embeds the compilation version. Viral reposts
usually don't credit anything.

**To trace:** For each repost you find, look for:
1. Embedded video iframes (source = next layer back)
2. Caption text (rarely has links)
3. Comments asking for source (the most reliable signal)
4. Bio "Submit Video" links (compilation accounts often have these — write to them)

When you find a compilation IG Reel ID like `DH6_bNhqyv-`, the URL is:
```
https://www.instagram.com/reel/<id>/
```

The OG metadata on that page contains:
- `og:title` — account name
- `og:description` — "X likes, Y comments - <account> on <date>: '<caption>'"
- `og:image` — thumbnail

This gives you the posting account and original post date even without logging in.

---

## Tool Effectiveness Matrix

| Tool | Use For | Effectiveness |
|------|---------|---------------|
| **Playwright browser** (MCP) | Reading any specific page (YT Shorts, IG Reel, news article, Reddit) | **HIGH** — works where WebFetch fails |
| **yt-dlp** | Capturing metadata, comments, frames from YT | **HIGH** — most reliable extraction |
| **Python + JSON** | Comment analysis, bulk filtering | **HIGH** — handles 5K+ comments in seconds |
| **Yandex reverse image** | License plates, faces, similar-looking duplicates | **HIGH** — best for plates |
| **Google Lens** | Finding pages where an image appears | **MEDIUM** — good for viral images, weak for plates |
| **WebSearch** | **Verification tool** once you have a candidate username + scene keywords. Useless for discovery. | **LOW for discovery / MEDIUM for verification** |
| **WebFetch** | Reading scrape-friendly pages | **MEDIUM** — fails on bot-protected sites |
| **Instagram `/reel/<id>/` OG metadata** | Public Reel URL without login | **HIGH** — returns account, caption, post date, thumbnail |
| **Reddit browser** | Search dashcam subs | **BROKEN** — captcha even on .json API |
| **MSHP crash DB** | Missouri crash report lookup | **DEPENDS ON NETWORK** — may be geoblocked |
| **Reddit .json API** | Unauthenticated queries | **BROKEN** — returns challenge HTML now |

### Critical: WebSearch is split-purpose

**For discovery** (you have nothing but the viral URL): WebSearch is LOW
effectiveness. Generic queries return hallucinated Reddit threads and conflated
incidents. Spend your time on comment mining and reverse image search instead.

**For verification** (you have a candidate username/handle): WebSearch is MEDIUM
effectiveness. A quoted-username query + 2–3 scene-specific keywords reliably
returns the specific source URL almost always:
```
"diktator56" instagram bridge dashcam truck
→ returns the specific Reel URL via caption keyword match
```

Don't waste WebSearch before Phase 2. Use it after you have a candidate
(credit from Phase 0, commenter ID from Phase 2, etc.).

### Critical: WebFetch vs Playwright on Bot-Protected Sites

`WebFetch` fails on most sites with anti-bot defenses:
- `web.archive.org` — blocked
- News sites with Cloudflare/anti-scraping — 403
- Instagram without auth — 401
- Reddit — challenge
- Missouri state sites — sometimes timed out

`Playwright` browser works on virtually all of these because it mimics a real
browser session (handles JS challenges, cookies, redirects). **Default to
Playwright for any page that needs JS or has anti-bot.**

---

## Common Patterns

### Pattern 1: Single Repost Layer

```
Viral YT Shorts (only known version)
→ Comments contain attribution link to IG Reel
→ IG Reel is the earliest indexed source
```

### Pattern 2: News Coverage as Bridge

```
Viral YT Shorts
→ News article (e.g., GM Authority) embedded IG Reel in iframe
→ Extract IG Reel ID from iframe src
→ IG Reel is one layer earlier
```

### Pattern 3: Multiple Compilation Layers

```
Raw creator (private/deleted)
→ Compilation IG account (@the_worst_driver)
→ News article
→ Compilation YT/TikTok channel (18 Wheel Heroes)
→ Viral repost
```

This is the most common. The earliest indexed layer is usually a compilation IG
account, not the raw creator. The raw creator's post is typically in a private
FB group, IG close-friends list, or has been deleted.

### Pattern 4: Local Comment Leads

When location ID is wrong in handoff, comment threads often correct it:
```
Handoff says "Lee's Summit, MO"
Comment with 1 like says "hate the 291 and I-70 interchange"
→ Actual location: US-291 / I-70 in Independence, MO
→ The green "Lee's Summit" sign visible in clip was a directional exit sign,
  not an indicator of being in Lee's Summit
```

This pattern appears in nearly every investigation where a viral clip has local
recognizability. Always trust locals over handoff inferences.

### Pattern 5: Uploader Credit Named in Description

The viral upload's own description names the source:
```
YT Shorts description: "Video Courtesy of Diktator56 (on instagram)"
→ @diktator56 IG Reel → original dashcam footage
→ 1 phase instead of 6 — credit-by-name republishers (Momento dashcam brand)
  preserve credits, anonymous reposters (18 Wheel Heroes) strip them.
```

**Detection**: search the description for `(?i)(courtesy|credit|source|original|via|repost)\s*[:\-]?\s*\S+`. If found, load the credited account, find the matching Reel, verify post-date predates the viral upload. Done.

Investigation difficulty scaling:
- **1–3 / 10**: Branded republisher + named credit → 30 min, Phase 0 only
- **4–6 / 10**: Anonymous reposter + decent comment thread → full Phases 2–6
- **7–8 / 10**: Anonymous reposter + sparse comments + no news coverage → outreach only
- **9–10 / 10**: Deleted/private original + no comment leads + no news + uncropped identifiers with no search hits → likely unrecoverable

### Pattern 6: Hashtag Location Beats Geotags

The original creator's Reel often has no geotag, but their caption hashtags name
the location directly:
```
Caption hashtags: #bullsbridge #ct
→ Bulls Bridge, Kent, Connecticut (one-lane covered bridge over the Housatonic)
→ Confirmed visually by the distinctive wooden covered bridge in the dashcam footage
```

Treat author hashtags on the source post as equivalent to local commenter intel —
just check both. Often the hashtags come BEFORE the comments and require no
guesswork.

---

## Anti-Patterns (What Doesn't Work)

### ❌ Trusting WebSearch for niche-event DISCOVERY
WebSearch returns hallucinated results for obscure queries (fake Reddit threads,
conflated incidents across states). If a query returns short or irrelevant
results, the topic is too niche for Google's general index. **Don't use WebSearch
as your first tool when you have nothing but the viral URL** — spend that time
on comment mining and reverse image search instead.

WebSearch IS useful as a verification tool once you have a candidate username
(see the Tool Effectiveness Matrix above). Just don't use it for discovery.

### ❌ Trusting top WebSearch hits without verification
WebSearch can return topically-related but unrelated clips that LOOK like
matches:
```
Query: "bridge dashcam truck" → top hit was a Reel of a truck cab coming off
at a bridge — different incident, different Reel ID
```
Always verify the candidate Reel's actual content (thumbnail, caption scene
description, OG image) before trusting it as the source. A correct-looking
snippet can be a completely different incident.

### ❌ Trusting Google AI Overview
AI Overview hallucinates details routinely. Plate states ("Pennsylvania"),
incident locations, vehicle descriptions — all unverified. Only trust structured
visual matches.

### ❌ WebFetch on bot-protected sites
Returns 403/timeout/challenge for most news sites, archives, social platforms.
Default to Playwright.

### ❌ Reddit for clip sourcing
Browser captcha. .json API blocked. Old.reddit.com blocked. Even with proper
User-Agent. Don't waste cycles — go to news/IG/TikTok instead.

### ❌ Assuming the raw original is findable
It's almost never on the indexed public web. Compilation accounts are the
typical earliest layer. The raw creator's post is in a private group or deleted.

### ❌ Searching news before comments
Comments contain hyper-local knowledge (location, identification, source links)
that news never has. Always scan comments first.

---

## Investigation Output Template

Save findings to `<investigation_dir>/FINDINGS.md`:

```markdown
# Clip X Source Investigation

## Starting Point
- URL: <viral url>
- Platform: YT Shorts / TikTok / IG Reel
- Uploader: <channel>
- Upload date: <date>
- Duration: <s>
- Engagement: <views>, <likes>

## Chain of Custody
1. <raw creator> — <date> — <url or "unknown">
2. <earliest repost> — <date> — <url>
3. <news coverage> — <date> — <url>
4. <viral repost> — <date> — <url>

## Earliest Known Source
- URL: <url>
- Account: <name>
- Date: <date>
- Caption: "<text>"

## Key Comments
- @author (X likes): "<quote>"

## Visual Identifiers
- License plate: <text> (state unverified / Missouri / Pennsylvania)
- Other: <badge / sign / face>

## Dead Ends
- ❌ <lead that didn't pan out>
- ❌ <lead that didn't pan out>

## Remaining Avenues
- ⏳ <outreach to follow up on>
```

---

## Related Tools

- **yt-dlp** — extract video metadata, comments, frames
- **Playwright MCP browser** — interact with social/news sites
- **Perplexity/research skill** — broader web research when trace fails
- **gws_drive.py** — if source video needs to be retrieved from Drive

## When to Stop

Stop investigating when:
1. You've traced back to the earliest indexed source (usually a compilation account)
2. All remaining avenues are outreach-based (DM the compilation account, email
   the viral channel, contact a TV station tip line)
3. The raw original is provably not on the indexed web (negative results from
   plate search, reverse image, comment scan all consistent)

The raw creator's post is most likely unrecoverable without paid P.I. work or
direct outreach. Document the chain of custody as far as it goes, then move on.