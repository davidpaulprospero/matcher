# Ralph Loop

Autonomous development assistant based on the Ralph Loop technique.

## Quick Start

```powershell
# Morning check-in (recommended)
.\scripts\ralph\1-im-learnding.bat

# Or start directly
.\scripts\ralph\ralph.ps1
```

## Batch Files

| File | Purpose |
|------|---------|
| 1-im-learnding.bat | Morning check-in: see progress, choose next action |
| 2-sleep-thats-where-im-a-viking.bat | Overnight mode |
| 3-me-fail-english.bat | Error review |
| 4-tastes-like-burning.bat | View detailed logs |
| 5-i-bent-my-wookie.bat | Debug mode |
| 6-the-leprechaun-tells-me-to-burn-things.bat | Watch progress |
| 7-hi-super-nintendo-chalmers.bat | Interview mode |

## Interview Mode

Give Ralph specific direction before he starts working.

```powershell
# Start interview
.\scripts\ralph\7-hi-super-nintendo-chalmers.bat

# Or from morning check-in
.\scripts\ralph\1-im-learnding.bat
# Choose option 3
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

## Configuration

| File | Purpose |
|------|---------|
| ralph-config.json | Focus areas, autonomy settings, test patterns |
| clients.json | Client profiles and preferences |
| prd.json | Product requirements (user stories) |
| progress.txt | Iteration log |
| queue.json | Interview queue state |

## Modes

| Mode | Description | Command |
|------|-------------|---------|
| Standard | Interactive sprint execution | `ralph.ps1` |
| TrueAuto | Continuous improvement | `ralph.ps1 -TrueAuto` |
| Queue | Process focus area queue | `ralph.ps1 -Queue` |
| Interview | Get user direction first | `7-hi-super-nintendo-chalmers.bat` |

## ralph.ps1 Parameters

```powershell
.\scripts\ralph\ralph.ps1 [options]

Options:
  -Queue             Process focus areas from queue.json (interview mode)
  -SkipPlanApproval  Skip plan approval prompts
  -TrueAuto          Continuous improvement mode (no exit on sprint complete)
  -Resume            Resume previous sprint instead of starting new
  -FocusArea <area>  Override focus area for this session
```

## Morning Check-in Options

When you run `1-im-learnding.bat`, you'll see:
1. Latest session info
2. Progress summary
3. Recent commits
4. Metrics summary
5. Any blockers

Then choose:
1. Continue Ralph (interactive)
2. Start TrueAuto mode
3. Interview mode (give specific direction)
4. View detailed logs
5. Just exit

## Logs

Session logs are stored in `scripts/ralph/logs/YYYY-MM-DD_HHMMSS/`:
- Claude conversations
- Iteration summaries
- Error traces
