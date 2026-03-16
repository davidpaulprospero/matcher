# Ralph's Choice - Design Document

> Let Ralph autonomously decide which focus area to work on based on project state, git activity, and strategic rotation.

## Overview

Ralph's Choice adds two new modes to the Ralph Loop launcher:
- **Ralph's Choice** - Ralph decides focus areas, user confirms each decision
- **Ralph's Choice Auto** - Ralph decides and continues autonomously

## Configuration

Add to `ralph-config.json`:

```json
"ralphsChoice": {
    "enabled": true,
    "activityWindowDays": 7,
    "maxSprintsBeforeRotate": 3,
    "weights": {
        "recentActivity": 0.5,
        "neglectedArea": 0.3,
        "categoryBalance": 0.2
    },
    "autoMode": {
        "continueDelaySeconds": 10,
        "logFile": "ralphs_choices.log",
        "maxConsecutiveSprints": 20
    }
}
```

| Field | Purpose | Default |
|-------|---------|---------|
| `enabled` | Master toggle for Ralph's Choice option | `true` |
| `activityWindowDays` | How far back to scan git history | `7` |
| `maxSprintsBeforeRotate` | Force rotation after N sprints in same area | `3` |
| `weights.recentActivity` | Weight for git commits touching area's files | `0.5` |
| `weights.neglectedArea` | Bonus for areas with 0 sprints | `0.3` |
| `weights.categoryBalance` | Spread work across categories | `0.2` |
| `autoMode.continueDelaySeconds` | Countdown before auto-continue | `10` |
| `autoMode.logFile` | Log file for autonomous decisions | `ralphs_choices.log` |
| `autoMode.maxConsecutiveSprints` | Safety limit for auto mode | `20` |

## Scoring Algorithm

When Ralph needs to pick a focus area, it scores all 16 areas:

```
For each focusArea:

    # 1. Recent Activity Score (0-1)
    commits = count git commits touching relevantFiles[area] in last N days
    activityScore = min(commits / 10, 1.0)  # caps at 10 commits

    # 2. Neglected Score (0-1)
    sprintsDone = focusAreaBreakdown[area].sprints or 0
    if sprintsDone == 0:
        neglectedScore = 1.0
    else:
        daysSince = days since last sprint in this area
        neglectedScore = min(daysSince / 14, 1.0)  # caps at 2 weeks

    # 3. Category Balance Score (0-1)
    categorySprintCount = sum of sprints in area's category
    totalSprints = total sprints across all categories
    categoryRatio = categorySprintCount / max(totalSprints, 1)
    balanceScore = 1.0 - categoryRatio  # under-represented = higher

    # Final weighted score
    score = (activityScore * 0.5) + (neglectedScore * 0.3) + (balanceScore * 0.2)
```

### Example Calculation

With current state (caption: 4 sprints, others: 0):

| Area | Activity | Neglected | Balance | Total |
|------|----------|-----------|---------|-------|
| caption | 0.8 | 0.0 | 0.0 | **0.40** |
| download | 0.6 | 1.0 | 1.0 | **0.70** |
| pipeline | 0.2 | 1.0 | 1.0 | **0.60** |

Ralph picks **download** (highest score).

## Stay vs Switch Logic

After each sprint completes, Ralph evaluates whether to stay or switch:

### Reasons to STAY
- Tests failing in current area
- Recent commits still flowing in (3+ commits in last day)
- Sprint generated unfinished stories

### Reasons to SWITCH
- Hit max sprints in area (default: 3)
- All tests passing, no recent activity
- Another area has significantly higher score (delta > 0.3)

### Decision Rule
```
if switchReasons.Count > 0 AND stayReasons.Count == 0:
    SWITCH to highest scoring area
else:
    STAY in current area
```

## UI Integration

### Launcher Menu

```
╔══════════════════════════════════════════════════╗
║           Ralph Loop Launcher                    ║
╠══════════════════════════════════════════════════╣
║  [S] Standard    - Work through focus areas      ║
║  [T] TrueAuto    - Continuous improvement        ║
║  [R] Resume      - Continue where you left off   ║
║  [Q] Smart Queue - Describe work, Ralph picks    ║
║                                                  ║
║  [A] Ralph's Choice      - Ralph decides (confirm each) ║
║  [Z] Ralph's Choice Auto - Ralph decides (fully auto)   ║
║                                                  ║
║  [X] Exit                                        ║
╚══════════════════════════════════════════════════╝
```

### Reasoning Display

```
┌─────────────────────────────────────────────────────┐
│  Ralph's Choice - Analyzing project state...        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Git Activity (last 7 days):                        │
│    * src/downloader/     8 commits                  │
│    * src/caption_fetcher 3 commits                  │
│    * src/stages/         2 commits                  │
│                                                     │
│  Sprint History:                                    │
│    * caption: 4 sprints (42 stories) - last: 2h ago │
│    * other areas: 0 sprints                         │
│                                                     │
│  Category Coverage:                                 │
│    * acquisition: 100% of work (over-represented)   │
│    * core, processing, output, intelligence, meta: 0│
│                                                     │
├─────────────────────────────────────────────────────┤
│  Scores:                                            │
│    download ........... 0.70                        │
│    pipeline ........... 0.60                        │
│    rate-limiting ...... 0.55                        │
│    quality ............ 0.50                        │
│                                                     │
│  > Ralph recommends: download                       │
│    Reason: High git activity (8 commits) +          │
│            never worked on + category needs balance │
│                                                     │
├─────────────────────────────────────────────────────┤
│  [Enter] Accept   [S] See all scores   [M] Manual   │
└─────────────────────────────────────────────────────┘
```

### Between-Sprint Display

```
┌─────────────────────────────────────────────────────┐
│  Sprint 6 complete! (download, 12 stories)          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Stay vs Switch Analysis:                           │
│                                                     │
│  Reasons to STAY in download:                       │
│    x Tests passing (0 failures)                     │
│    x No unfinished stories                          │
│                                                     │
│  Reasons to SWITCH:                                 │
│    + Area stable: tests passing, no recent commits  │
│    + pipeline needs attention (score: 0.65)         │
│                                                     │
│  > Ralph recommends: SWITCH to pipeline             │
│                                                     │
├─────────────────────────────────────────────────────┤
│  [Enter] Accept   [K] Keep download   [M] Manual    │
└─────────────────────────────────────────────────────┘
```

### Auto Mode Display

```
┌─────────────────────────────────────────────────────┐
│  Ralph's Choice Auto                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Sprint 6 complete! (download, 12 stories)          │
│                                                     │
│  Decision: SWITCH to pipeline                       │
│  Reason: Area stable, pipeline has high activity    │
│                                                     │
│  Continuing in 10s... [Press any key to pause]      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Mode Comparison

| Aspect | Ralph's Choice | Ralph's Choice Auto |
|--------|----------------|---------------------|
| Initial area selection | Show reasoning, wait for confirm | Show reasoning, auto-continue after 10s |
| Between sprints | Show stay/switch, wait for confirm | Log decision, auto-continue |
| User can interrupt | At each decision point | Press any key during countdown |
| Logging | Normal | Verbose (writes to `ralphs_choices.log`) |

## Implementation

### Files to Modify

| File | Changes |
|------|---------|
| `ralph-config.json` | Add `ralphsChoice` config block |
| `launcher.ps1` | Add `[A]` and `[Z]` modes, `Show-RalphsReasoning` function |
| `ralph.ps1` | Add `Start-RalphsChoiceLoop`, `Get-StayOrSwitchDecision`, scoring functions |
| `interview.ps1` | Fix `Ask-Area` empty response to trigger Ralph's Choice |

### New Functions

```powershell
# In ralph.ps1
Get-FocusAreaScore          # Calculate weighted score for one area
Get-AllFocusAreaScores      # Score all 16 areas
Get-GitActivityByArea       # Count commits per area (last N days)
Get-StayOrSwitchDecision    # After sprint: stay or rotate?
Start-RalphsChoiceLoop      # Main loop for Ralph's Choice mode
Start-RalphsChoiceAutoLoop  # Fully autonomous loop with countdown

# In launcher.ps1
Show-RalphsReasoning        # Display scoring breakdown + recommendation
Start-RalphsChoice          # Entry point from launcher menu
```

### Flow Diagram

```
Launcher: [A] Ralph's Choice / [Z] Ralph's Choice Auto
    |
    v
Show-RalphsReasoning()
    | <- User: [Enter] Accept (or auto-continue in Auto mode)
    v
Start-RalphsChoiceLoop() / Start-RalphsChoiceAutoLoop()
    |
    +---> Generate sprint PRD for chosen area
    |
    +---> Run sprint (existing Invoke-ClaudeForFocusArea)
    |
    +---> Sprint complete
    |         |
    |         v
    |    Get-StayOrSwitchDecision()
    |         |
    |         v
    |    Show-RalphsReasoning() <- with stay/switch context
    |         | <- User: [Enter] Accept (or auto-continue)
    |         |
    +---------+ (loop)
```

## Integration Points

1. **Launcher menu** (`launcher.ps1:34-64`) - Add new mode options
2. **Interview `Ask-Area`** (`interview.ps1:294-296`) - Fix empty response to use Ralph's Choice
3. **TrueAuto fallback** (`ralph.ps1:2404-2405`) - Can optionally use Ralph's Choice scoring instead of hardcoded defaults
