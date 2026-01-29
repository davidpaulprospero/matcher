# Ralph Loop

Autonomous development assistant based on the Ralph Loop technique.

> *"Sleep! That's where I'm a Viking!"* — Ralph Wiggum

## Quick Start

```powershell
# Interactive launcher (recommended)
.\scripts\ralph\sleep-thats-where-im-a-viking.bat

# Or start directly
.\scripts\ralph\ralph.ps1
```

## Entry Points

| File | Purpose |
|------|---------|
| [`sleep-thats-where-im-a-viking.bat`](sleep-thats-where-im-a-viking.bat) | **Main launcher** - Interactive mode selection |
| [`focus.bat`](focus.bat) | Quick focus on a single area (full auto) |
| [`watch.bat`](watch.bat) | Monitor progress dashboard |
| [`interview.ps1`](interview.ps1) | Guided interview mode |

### Launcher Shortcuts

```powershell
# Overnight mode (TrueAuto + testing)
.\scripts\ralph\sleep-thats-where-im-a-viking.bat overnight

# YOLO mode (TrueAuto, no specific area)
.\scripts\ralph\sleep-thats-where-im-a-viking.bat yolo

# With specific mode and focus
.\scripts\ralph\sleep-thats-where-im-a-viking.bat -Mode trueauto -FocusArea pipeline
```

## Modes

The launcher ([`launcher.ps1`](launcher.ps1)) provides these modes:

| Mode | Description |
|------|-------------|
| **Standard** | Work through stories, pause on sprint complete |
| **TrueAuto** | Continuous improvement, auto-generate new sprints |
| **Resume** | Continue where you left off |
| **Smart Queue** | Describe what you want, Ralph picks focus areas |
| **Ralph's Choice** | Ralph decides focus areas (confirm each) |
| **Ralph's Choice Auto** | Ralph decides (fully autonomous) |
| **Status** | Quick status check |
| **Morning** | Morning check-in with commits |
| **Logs** | View logs and reports |
| **Recovery** | Emergency stop and recovery |
| **Watch Only** | Just the watch dashboard |

## Interview Mode

Give Ralph specific direction before he starts working.

```powershell
# Start interview
.\scripts\ralph\interview.ps1

# Or via launcher, choose "Smart Queue"
.\scripts\ralph\sleep-thats-where-im-a-viking.bat
```

### Interview Flow

1. Ralph asks what kind of work (bug/feature/improvement/client)
2. You describe what you want
3. Ralph asks follow-up questions if needed (area, client, priority)
4. Ralph suggests 3-5 focus areas based on keywords
5. You approve/modify the list
6. Ralph spawns loop + watch windows and gets to work

### Focus Area Keywords

Interview mode detects keywords in your description and suggests relevant focus areas:

| Keywords | Focus Area |
|----------|------------|
| otio, timeline, edl, xml, davinci, resolve | otio |
| download, youtube, yt-dlp, 429, rate limit | rate-limiting |
| match, confidence, score, quality | quality |
| caption, subtitle, srt, transcript | caption |
| config, yaml, setting | config |
| test, coverage, pytest | testing |
| speed, slow, performance, cache | speed |
| client, preset, learn | client-learning |
| heal, recover, retry, error | agents |
| compile, compilation, topic, keyword | compilation |

### Modifying Suggestions

After Ralph suggests focus areas, you can:
- `[A]` - Approve all and start
- `[1-5]` - Remove specific area by number
- `[+area]` - Add an area (e.g., `+testing`)
- `[R]` - Restart interview

### After Completion

When all focus areas are done, Ralph asks:
- **[I]nterview** - Give more direction
- **[T]rueAuto** - Continue improving on his own
- **[S]top** - Check results later

### Crash Recovery

If interrupted, next interview start offers to resume:

```
Found interrupted session:
  Context: bug: OTIO crashes on unicode
  Progress: 2/4 focus areas complete

  Resume this session? [Y]es / [N]ew interview
```

## Focus Areas

Focus areas are organized into categories:

### Core Infrastructure [C]
- **pipeline** - Core pipeline stages and orchestration
- **config** - Configuration management and validation

### Content Acquisition [A]
- **rate-limiting** - YouTube download resilience (cookies, circuit breaker, impersonation)
- **caption** - Caption-first mode and transcript handling
- **download** - Core download logic, yt-dlp integration, browser impersonation

### Content Processing [P]
- **quality** - Match accuracy and confidence scores
- **speed** - Pipeline speed and caching optimization
- **compilation** - Keyword compilation and montage features

### Output Generation [O]
- **otio** - Timeline formats: OTIO, EDL, XML

### AI & Learning [I]
- **agents** - Self-healing agents and error recovery
- **client-learning** - Cross-project learning and client presets

### Development & Testing [M]
- **testing** - General test coverage
- **unit-tests** - Isolated function tests with mocking
- **integration-tests** - Component interaction tests
- **mutation-tests** - Verify test quality catches changes
- **documentation** - API docs, README, inline comments
- **ux** - CLI prompts, progress feedback, error messages

## Configuration

| File | Purpose |
|------|---------|
| [`ralph-config.json`](ralph-config.json) | Focus areas, autonomy settings, test patterns |
| [`clients.json`](clients.json) | Client profiles and preferences |
| [`prd.json`](prd.json) | Product requirements (user stories) |
| [`progress.txt`](progress.txt) | Iteration log |
| [`queue.json`](queue.json) | Interview queue state |
| [`metrics.csv`](metrics.csv) | Performance metrics and costs |
| [`sprint_history.json`](sprint_history.json) | Archive of completed sprints |

## ralph.ps1 Parameters

```powershell
.\scripts\ralph\ralph.ps1 [options]

Options:
  -Queue             Process focus areas from queue.json (interview mode)
  -SkipPlanApproval  Skip plan approval prompts
  -TrueAuto          Continuous improvement mode (no exit on sprint complete)
  -Resume            Resume previous sprint instead of starting new
  -FocusArea <area>  Override focus area for this session
  -RalphsChoice      Ralph decides focus areas, user confirms each
  -RalphsChoiceAuto  Ralph decides and continues autonomously
  -Task <task>       Specific task description
```

## Domain Modules

The main [`ralph.ps1`](ralph.ps1) loads these modules from [`lib/`](lib/):

| Module | Purpose |
|--------|---------|
| [`sprint.ps1`](lib/sprint.ps1) | PRD generation, sprint lifecycle, archive |
| [`scoring.ps1`](lib/scoring.ps1) | Ralph's Choice algorithm, focus area scoring |
| [`queue.ps1`](lib/queue.ps1) | Interview queue processing |
| [`metrics.ps1`](lib/metrics.ps1) | CSV metrics, iteration tracking |
| [`quality.ps1`](lib/quality.ps1) | Quality gates, test baselines |
| [`prompts.ps1`](lib/prompts.ps1) | Claude prompt building |
| [`healing.ps1`](lib/healing.ps1) | Self-healing and error recovery |
| [`heartbeat.ps1`](lib/heartbeat.ps1) | Health monitoring |
| [`claude.ps1`](lib/claude.ps1) | Claude API interaction |
| [`display.ps1`](lib/display.ps1) | Console output formatting |
| [`loops.ps1`](lib/loops.ps1) | Main execution loops |

## Logs

Session logs are stored in `scripts/ralph/logs/YYYY-MM-DD_HHMMSS/`:
- Claude conversations
- Iteration summaries
- Error traces
- Interview logs

## Architecture

```
scripts/ralph/
├── ralph.ps1              # Main orchestrator (~800 lines)
├── launcher.ps1           # Interactive mode launcher
├── interview.ps1          # Guided interview mode
├── watch.ps1 / watch.bat  # Progress dashboard
├── status.ps1             # Quick status checks
├── lib/                   # Domain modules
│   ├── sprint.ps1
│   ├── scoring.ps1
│   ├── queue.ps1
│   ├── metrics.ps1
│   ├── quality.ps1
│   ├── prompts.ps1
│   ├── healing.ps1
│   ├── heartbeat.ps1
│   ├── claude.ps1
│   ├── display.ps1
│   └── loops.ps1
├── *.json                 # Config and state files
├── *.csv                  # Metrics
└── logs/                  # Session logs
```

## Ralph's Choice Algorithm

When using Ralph's Choice modes, Ralph scores focus areas based on:
- **Git Activity** - Recent commits per area
- **Code Churn** - Files changed in recent work
- **Neglected Areas** - Time since last work on area
- **Category Balance** - Distribution across categories

Ralph presents the highest-scoring area for confirmation (or proceeds autonomously in Auto mode).
