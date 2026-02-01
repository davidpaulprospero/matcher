# Ralph Loop

Autonomous development assistant based on the Ralph Loop technique.

> *"Sleep! That's where I'm a Viking!"* — Ralph Wiggum

## Quick Start

```powershell
# Interactive interview (recommended)
.\scripts\ralph\interview.ps1

# Quick start with specific focus area
.\scripts\ralph\interview.ps1 -FocusArea pipeline -Mode trueauto

# Overnight mode with multiple areas
.\scripts\ralph\interview.ps1 -Mode overnight -FocusAreas download,quality

# Resume previous session
.\scripts\ralph\interview.ps1 -Resume

# Utility commands
.\scripts\ralph\interview.ps1 -Status    # Check status
.\scripts\ralph\interview.ps1 -Watch     # Open dashboard
.\scripts\ralph\interview.ps1 -Stop      # Graceful stop
.\scripts\ralph\interview.ps1 -Logs      # View logs
.\scripts\ralph\interview.ps1 -Recovery  # Emergency recovery
.\scripts\ralph\interview.ps1 -Morning   # Morning check-in
```

## Entry Point

**`interview.ps1`** is the **ONLY entry point** for Ralph. All functionality has been consolidated here.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `-Mode <mode>` | Execution mode (see below) |
| `-FocusArea <area>` | Work on specific focus area directly |
| `-FocusAreas <area1,area2>` | Multiple areas for overnight mode |
| `-Resume` | Continue previous session |
| `-NoLaunch` | Save queue but don't spawn windows |
| `-MaxHours <hours>` | Max hours for overnight (default: 12) |

### Utility Flags

| Flag | Action |
|------|--------|
| `-Status` | Show status and exit |
| `-Watch` | Open watch dashboard and exit |
| `-Stop` | Request graceful stop and exit |
| `-Logs` | View recent logs and exit |
| `-Recovery` | Emergency recovery menu |
| `-Morning` | Morning check-in with commits |
| `-Queue` | Skip work type, go to queue selection |

## Modes

| Mode | Description |
|------|-------------|
| **standard** | Work through stories, pause on sprint complete |
| **trueauto** | Continuous improvement, auto-generate new sprints |
| **resume** | Continue where you left off |
| **smartqueue** | Describe what you want, Ralph picks focus areas |
| **ralphschoice** | Ralph decides focus areas (confirm each) |
| **ralphschoiceauto** | Ralph decides (fully autonomous) |
| **overnight** | Multi-focus rotation (12+ hours) |

## Interactive Interview Flow

When running without parameters:

1. Ralph asks what kind of work (bug/feature/improvement/client/queue)
2. You describe what you want
3. Ralph asks follow-up questions if needed (area, client, priority)
4. Ralph suggests 3-5 focus areas based on keywords
5. You approve/modify the list
6. Choose execution mode
7. Ralph spawns loop + watch windows and gets to work

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
- `[A]` - Approve all and continue to mode selection
- `[1-5]` - Remove specific area by number
- `[+area]` - Add an area (e.g., `+testing`)
- `[R]` - Restart interview

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
| `config/ralph-config.json` | Focus areas, autonomy settings, test patterns |
| `config/clients.json` | Client profiles and preferences |
| `state/prd.json` | Product requirements (user stories) |
| `state/queue.json` | Interview queue state |
| `state/progress.txt` | Iteration log |
| `state/metrics.csv` | Performance metrics and costs |
| `state/sprint_history.json` | Archive of completed sprints |

## Domain Modules

The main `ralph.ps1` (internal engine) loads these modules from `lib/`:

| Module | Purpose |
|--------|---------|
| `sprint.ps1` | PRD generation, sprint lifecycle, archive |
| `scoring.ps1` | Ralph's Choice algorithm, focus area scoring |
| `queue.ps1` | Interview queue processing |
| `metrics.ps1` | CSV metrics, iteration tracking |
| `quality.ps1` | Quality gates, test baselines |
| `prompts.ps1` | Claude prompt building |
| `healing.ps1` | Self-healing and error recovery |
| `heartbeat.ps1` | Health monitoring |
| `claude.ps1` | Claude API interaction |
| `display.ps1` | Console output formatting |
| `loops.ps1` | Main execution loops |
| `interview.ps1` | Context improvement, LLM suggestions |
| `paths.ps1` | Centralized path definitions |

## Logs

Session logs are stored in `logs/YYYY-MM-DD_HHMMSS/`:
- Claude conversations
- Iteration summaries
- Error traces
- Interview logs

## Architecture

```
scripts/ralph/
├── interview.ps1          # UNIFIED ENTRY POINT (use this!)
├── ralph.ps1              # Internal execution engine (~900 lines)
├── watch.ps1              # Progress dashboard
├── status.ps1             # Quick status checks
├── graceful-stop.ps1      # Graceful stop management
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
│   ├── loops.ps1
│   ├── interview.ps1
│   ├── learning.ps1
│   ├── paths.ps1
│   └── reporting.ps1
├── config/                # Configuration files
├── state/                 # Runtime state files
├── logs/                  # Session logs
└── tests/                 # Pester tests
```

## Script Organization

**Convention**: Shared functions live in `lib/`, standalone scripts use them but don't define shared functions.

| Location | Purpose | Rule |
|----------|---------|------|
| `lib/*.ps1` | Shared functions (sourced by ralph.ps1) | Define functions called by multiple scripts |
| `*.ps1` (root) | Standalone entry points | Use lib functions, don't define shared ones |

Standalone scripts should have this header:
```powershell
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1
```

This convention is enforced by `tests/RalphStructure.Tests.ps1`.

## Ralph's Choice Algorithm

When using Ralph's Choice modes, Ralph scores focus areas based on:
- **Git Activity** - Recent commits per area
- **Code Churn** - Files changed in recent work
- **Neglected Areas** - Time since last work on area
- **Category Balance** - Distribution across categories

Ralph presents the highest-scoring area for confirmation (or proceeds autonomously in Auto mode).

## Testing

```powershell
# Run all Ralph tests
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```
