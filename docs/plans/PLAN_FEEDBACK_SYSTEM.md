# Feedback System Improvement Plan

## Executive Summary

This plan outlines improvements to the content filtering and feedback loop system to:
1. Learn from employer/editorial rejections automatically
2. Build channel reputation scores from historical data
3. Create project-specific presets that evolve over time
4. Enable NLE-to-pipeline feedback via marker export

## Current System Analysis

### What We Have

| Component | Location | Description |
|-----------|----------|-------------|
| Title Blacklist | `config.download.title_blacklist` | Static list, checks title + channel |
| LLM Title Filter | `TitleFilter.filter_titles_with_llm()` | Batch relevance scoring via Gemini |
| Content Presets | `DEFAULT_PRESETS` in title_filter.py | raw, documentary, stock_footage |
| Sources Tracking | `sources.json` per project | Metadata for downloaded videos |

### Current Limitations

1. **No learning from rejections** - Manual blacklist updates required
2. **No channel-level scoring** - Each video evaluated independently
3. **Static presets** - Same rules for all projects
4. **No NLE feedback loop** - Employer notes via text, not structured data
5. **No cross-project learning** - Each project starts fresh

---

## Proposed Improvements

### 1. Rejection Learning System

**Goal**: Automatically learn from videos rejected in post-production.

#### 1.1 Rejection Database Schema

```python
# src/feedback/rejections.py
@dataclass
class RejectedVideo:
    video_id: str           # YouTube video ID
    channel_id: str         # Channel ID
    channel_name: str       # Channel name
    title: str              # Video title
    rejection_reason: str   # Why rejected (trainer, movie, low quality, etc.)
    rejection_source: str   # How identified (manual, marker, auto)
    project: str            # Which project
    timestamp: datetime     # When rejected

@dataclass
class RejectionDatabase:
    rejections: List[RejectedVideo]
    channel_scores: Dict[str, float]  # channel_id -> trust score (0-1)
```

#### 1.2 Rejection Sources

| Source | How It Works | Priority |
|--------|--------------|----------|
| DaVinci Markers | Parse exported CSV markers with "REJECT:" prefix | High |
| Manual Blacklist | User adds to `project_rejections.yaml` | High |
| LLM Detection | Auto-detect trainer/movie via description analysis | Medium |
| Cross-Project | Videos rejected in other projects | Low |

#### 1.3 Implementation

```python
# In title_filter.py
def _apply_rejection_database(self, videos: List[Dict]) -> List[Dict]:
    """Filter videos based on rejection history."""
    db = load_rejection_database()
    filtered = []

    for video in videos:
        # Check if video ID was previously rejected
        if video['id'] in db.rejected_video_ids:
            logger.debug(f"Skipping previously rejected: {video['title']}")
            continue

        # Check channel reputation score
        channel_score = db.channel_scores.get(video['channel_id'], 0.5)
        if channel_score < self.config.min_channel_score:
            logger.debug(f"Skipping low-reputation channel: {video['channel']}")
            continue

        filtered.append(video)

    return filtered
```

---

### 2. Channel Reputation Scoring

**Goal**: Score channels based on historical acceptance/rejection rates.

#### 2.1 Channel Score Calculation

```python
def calculate_channel_score(channel_id: str, db: RejectionDatabase) -> float:
    """
    Calculate channel trust score (0.0 - 1.0).

    Score = (accepted - rejected * 2) / total + base_score

    Factors:
    - acceptance_rate: Videos from this channel that were used
    - rejection_rate: Videos explicitly rejected (weighted 2x)
    - subscriber_count: YouTube API metric (credibility signal)
    - video_quality_avg: Average LLM relevance score
    """
    stats = db.get_channel_stats(channel_id)

    if stats.total_videos == 0:
        return 0.5  # Neutral for unknown channels

    acceptance_rate = stats.accepted / stats.total_videos
    rejection_penalty = (stats.rejected * 2) / stats.total_videos

    base_score = 0.5
    score = base_score + (acceptance_rate * 0.3) - (rejection_penalty * 0.4)

    # Boost for high-subscriber channels (proxy for quality)
    if stats.subscriber_count > 1_000_000:
        score += 0.1
    elif stats.subscriber_count > 100_000:
        score += 0.05

    return max(0.0, min(1.0, score))
```

#### 2.2 Channel Categories

```yaml
# config.yaml
channel_categories:
  trusted:  # Always accept
    - "BBC"
    - "National Geographic"
    - "PBS"
  blocked:  # Always reject
    - "Cesar Millan"
    - "Zak George"
    - "Movieclips"
  neutral:  # Use scoring
    - default
```

---

### 3. Project-Specific Presets

**Goal**: Each project can define and evolve its own content preferences.

#### 3.1 Preset Inheritance

```yaml
# project_config.yaml
content_filter:
  # Start from base preset
  base_preset: documentary

  # Project-specific additions
  additional_blacklist:
    - "specific trainer name"
    - "competitor brand"

  # Override acceptance criteria
  prefer:
    - "shelter footage"
    - "adoption events"

  # Learn from this project's rejections
  learn_rejections: true

  # Apply rejections from other projects by same client
  cross_project_learning: true
  client_id: "theresa"
```

#### 3.2 Preset Evolution

```python
def evolve_preset(project_dir: Path) -> Dict:
    """
    Generate an evolved preset based on project history.

    Analyzes:
    - Which videos were used vs skipped
    - Rejection patterns (channels, keywords, duration)
    - LLM relevance scores
    """
    sources = load_sources_json(project_dir)
    rejections = load_project_rejections(project_dir)

    # Find patterns in rejections
    rejected_channels = Counter(r.channel_name for r in rejections)
    rejected_keywords = extract_keywords_from_titles([r.title for r in rejections])

    # Find patterns in accepted videos
    accepted = [s for s in sources if s.was_used]
    preferred_duration = statistics.median(s.duration for s in accepted)
    preferred_channels = Counter(s.channel for s in accepted).most_common(10)

    return {
        "auto_blacklist": list(rejected_channels.keys())[:20],
        "auto_keywords_blacklist": rejected_keywords[:30],
        "preferred_duration_range": (preferred_duration * 0.5, preferred_duration * 2),
        "preferred_channels": [c for c, _ in preferred_channels],
    }
```

---

### 4. NLE Feedback Integration

**Goal**: Import feedback from DaVinci Resolve via markers/notes.

#### 4.1 Marker-Based Rejection

**Workflow**:
1. Editor reviews timeline in DaVinci
2. Adds markers to rejected clips: `REJECT: trainer content`
3. Exports markers as CSV
4. Pipeline parses CSV and updates rejection database

```python
# src/feedback/davinci_import.py
def import_davinci_markers(csv_path: Path) -> List[RejectedVideo]:
    """
    Parse DaVinci Resolve marker export CSV.

    Expected format:
    #,Name,Start TC,End TC,Duration,Notes,Color
    1,REJECT: trainer,01:00:00:00,01:00:05:00,00:00:05:00,Cesar Millan clip,Red
    """
    rejections = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Name'].startswith('REJECT:'):
                reason = row['Name'].replace('REJECT:', '').strip()
                timecode = row['Start TC']

                # Find which video is at this timecode
                video = find_video_at_timecode(timecode)
                if video:
                    rejections.append(RejectedVideo(
                        video_id=video.id,
                        channel_id=video.channel_id,
                        channel_name=video.channel,
                        title=video.title,
                        rejection_reason=reason,
                        rejection_source='davinci_marker',
                        timestamp=datetime.now()
                    ))

    return rejections
```

#### 4.2 Automated Marker Colors

| Color | Meaning | Action |
|-------|---------|--------|
| Red | Reject - don't use again | Add to rejection DB |
| Orange | Warning - review needed | Flag for review |
| Yellow | Replace - find alternative | Re-match segment |
| Green | Approved - use more like this | Boost channel score |
| Blue | Note - informational | Log only |

---

### 5. YouTube API Integration

**Goal**: Use YouTube Data API for richer metadata and channel scoring.

#### 5.1 Enhanced Metadata Fetching

```python
def fetch_channel_metadata(channel_ids: List[str]) -> Dict[str, ChannelInfo]:
    """
    Fetch channel metadata for reputation scoring.

    Returns:
    - subscriber_count
    - video_count
    - channel_age (upload date of first video)
    - category (e.g., "Education", "Entertainment")
    - country
    """
    youtube = build('youtube', 'v3', developerKey=api_key)

    response = youtube.channels().list(
        part='snippet,statistics,contentDetails',
        id=','.join(channel_ids)
    ).execute()

    return {
        item['id']: ChannelInfo(
            name=item['snippet']['title'],
            subscribers=int(item['statistics']['subscriberCount']),
            videos=int(item['statistics']['videoCount']),
            category=item['snippet'].get('category', 'Unknown'),
            country=item['snippet'].get('country', 'Unknown'),
        )
        for item in response['items']
    }
```

#### 5.2 Channel Quality Heuristics

```python
QUALITY_SIGNALS = {
    'high_quality': [
        lambda c: c.subscribers > 100_000,
        lambda c: c.videos > 50,
        lambda c: c.category in ['Education', 'Documentary', 'News'],
        lambda c: 'official' in c.name.lower(),
    ],
    'low_quality': [
        lambda c: c.subscribers < 1_000,
        lambda c: c.videos < 5,
        lambda c: c.category in ['Gaming', 'Music'],
        lambda c: any(x in c.name.lower() for x in ['reaction', 'vlog', 'daily']),
    ],
    'blocked_categories': [
        'trainer', 'training', 'tutorial', 'how to',
        'movie', 'trailer', 'clip', 'scene',
    ]
}
```

---

### 6. Cross-Project Learning

**Goal**: Share learnings across projects for the same client.

#### 6.1 Global Rejection Database

```
~/.matcher_rejections/
├── global_rejections.json     # All rejections across projects
├── channel_scores.json        # Computed channel reputation
├── client_theresa/            # Client-specific data
│   ├── rejections.json
│   ├── preferences.yaml
│   └── evolved_preset.yaml
└── client_stu/
    ├── rejections.json
    └── preferences.yaml
```

#### 6.2 Client Profiles

```yaml
# ~/.matcher_rejections/client_theresa/preferences.yaml
client_id: theresa
created: 2026-01-15

# Learned preferences
content_style: documentary
preferred_duration: 30-180  # seconds
avoid_trainers: true
avoid_movies: true

# Custom blacklist (learned + manual)
blacklist:
  channels:
    - "Cesar Millan"
    - "Will Atherton Canine Training"
    - "Movieclips"
  keywords:
    - "training tips"
    - "official trailer"

# Quality thresholds
min_channel_score: 0.4
min_llm_relevance: 0.6
```

---

## Implementation Phases

### Phase 1: Rejection Database (Week 1)
- [ ] Create `RejectionDatabase` class
- [ ] Add `project_rejections.yaml` support
- [ ] Integrate into `_apply_title_blacklist()`
- [ ] CLI command: `python main.py --import-rejections <csv>`

### Phase 2: Channel Scoring (Week 2)
- [ ] Fetch channel metadata via YouTube API
- [ ] Implement `calculate_channel_score()`
- [ ] Add `channel_categories` config
- [ ] Display channel scores in match report

### Phase 3: NLE Feedback (Week 3)
- [ ] DaVinci marker CSV parser
- [ ] Marker color conventions
- [ ] Auto-import on `--match-only`
- [ ] `/import-feedback` skill

### Phase 4: Cross-Project Learning (Week 4) ✅
- [x] Global rejection database
- [x] Client profiles (`src/feedback/client_profiles.py`)
- [x] Preset evolution algorithm (`evolve_preset_from_history()`)
- [x] `--client` flag for project grouping
- [x] `--evolve-preset` flag for generating evolved presets
- [x] `--list-clients` and `--client-stats` flags

---

## Config Changes

```yaml
# config.yaml additions
feedback:
  enabled: true

  # Rejection learning
  learn_from_rejections: true
  rejection_weight: 2.0  # How much to penalize rejected content

  # Channel scoring
  channel_scoring:
    enabled: true
    min_score: 0.3  # Skip channels below this
    boost_verified: 0.1  # Boost for verified channels

  # Cross-project learning
  cross_project:
    enabled: true
    client_id: ""  # Set per-project
    share_rejections: true

  # NLE integration
  davinci:
    import_markers: true
    marker_prefix: "REJECT:"
    auto_import_path: ""  # Path to marker CSV
```

---

## CLI Commands

```bash
# Import rejections from DaVinci markers
python main.py --import-rejections "path/to/markers.csv"

# View channel scores
python main.py --show-channel-scores

# Evolve preset from project history
python main.py --evolve-preset --project "E:/Edit Job/client/project"

# Apply learnings from another project
python main.py --apply-learnings "E:/Edit Job/client/other_project"

# Show rejection statistics
python main.py --rejection-stats
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Manual blacklist updates per project | 5-10 | 0-1 |
| Trainer/movie clips in output | 10-15% | <1% |
| Time to apply employer feedback | 30 min | 2 min |
| Cross-project consistency | N/A | 95% |

---

## References

- [Perplexity Research: Content Filtering Feedback Loops](research notes above)
- [YouTube Data API: Channels](https://developers.google.com/youtube/v3/docs/channels)
- [DaVinci Resolve Marker Export](https://www.blackmagicdesign.com/products/davinciresolve)
- [Active Learning for Content Moderation](https://arxiv.org/abs/2108.02697)
