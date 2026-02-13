# Ralph Deep Interview Module
# Provides dynamic follow-up questioning for thorough context gathering

# ============================================================================
# SEARCH BUDGET FUNCTIONS
# ============================================================================

function Get-SearchBudgetInfo {
    <#
    .SYNOPSIS
        Reads config.yaml to get video search budget settings
    .DESCRIPTION
        Extracts max_total_results and results_per_keyword from config.yaml
        to help users understand search budget allocation
    .PARAMETER ProjectRoot
        Root directory of the project (defaults to parent of RalphDir)
    .RETURNS
        Hashtable with: maxKeywords, currentBudget, resultsPerKeyword, configPath, found
    #>
    param(
        [string]$ProjectRoot = $null
    )

    # Determine project root
    if (-not $ProjectRoot) {
        $ProjectRoot = if ($script:ProjectRoot) {
            $script:ProjectRoot
        } else {
            Split-Path -Parent (Split-Path -Parent $script:RalphDir)
        }
    }

    $configPath = Join-Path $ProjectRoot "config.yaml"

    if (-not (Test-Path $configPath)) {
        Write-Verbose "  config.yaml not found at $configPath"
        return @{
            maxKeywords = 10
            currentBudget = 200
            resultsPerKeyword = 20
            configPath = $configPath
            found = $false
        }
    }

    try {
        # Read config.yaml and extract search budget values using regex
        # This avoids dependency on PowerShell-Yaml module
        $configContent = Get-Content $configPath -Raw -ErrorAction Stop

        # First try to extract search_budget section (primary)
        $searchBudgetMatch = $configContent -match '(?s)search_budget:\s*\n((?:\s{2,}.+\n?)+)'

        $maxTotal = 200
        $resultsPerKw = 20
        $source = "defaults"

        if ($searchBudgetMatch -and $matches) {
            # Found dedicated search_budget section
            $searchBudgetSection = $matches[1]

            if ($searchBudgetSection -match 'max_total_results:\s*(\d+)') {
                $maxTotal = [int]$matches[1]
            }

            if ($searchBudgetSection -match 'results_per_keyword:\s*(\d+)') {
                $resultsPerKw = [int]$matches[1]
            }
            $source = "search_budget"
        }
        else {
            # Fallback to video_search section
            $videoSearchMatch = $configContent -match '(?s)video_search:\s*\n((?:\s{2,}.+\n?)+)'

            if ($videoSearchMatch -and $matches) {
                $videoSearchSection = $matches[1]

                if ($videoSearchSection -match 'max_total_results:\s*(\d+)') {
                    $maxTotal = [int]$matches[1]
                }

                if ($videoSearchSection -match 'results_per_keyword:\s*(\d+)') {
                    $resultsPerKw = [int]$matches[1]
                }
                $source = "video_search"
            }
        }

        Write-Verbose "  Search budget loaded from: $source"

        return @{
            maxKeywords = [Math]::Floor($maxTotal / $resultsPerKw)
            currentBudget = $maxTotal
            resultsPerKeyword = $resultsPerKw
            maxTotalResults = $maxTotal
            configPath = $configPath
            source = $source
            found = $true
        }
    }
    catch {
        Write-Verbose "  Error reading config.yaml: $($_.Exception.Message)"
        return @{
            maxKeywords = 10
            currentBudget = 200
            resultsPerKeyword = 20
            maxTotalResults = 200
            configPath = $configPath
            source = "defaults"
            found = $false
        }
    }
}

function Get-DistributedKeywordBudget {
    <#
    .SYNOPSIS
        Calculates adjusted results_per_keyword based on keyword count
    .DESCRIPTION
        Always distributes budget evenly: floor(max_total_results / keyword_count)
        Ensures all keywords receive at least 1 search attempt
    .PARAMETER Keywords
        Array of keyword strings
    .PARAMETER MaxTotalResults
        Maximum total results from config (default: 200)
    .PARAMETER ResultsPerKeyword
        Desired results per keyword from config (default: 20)
    .RETURNS
        Hashtable with: adjustedResultsPerKeyword, totalKeywords, willReduce,
                       warningMessage, effectiveTotal
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$Keywords,
        [int]$MaxTotalResults = 200,
        [int]$ResultsPerKeyword = 20
    )

    # Handle edge cases
    if ($Keywords.Count -eq 0) {
        return @{
            adjusted_budget = 0
            keyword_count = 0
            willReduce = $false
            warningMessage = "No keywords provided"
            effectiveTotal = 0
        }
    }

    $keyword_count = $Keywords.Count

    # Always use the formula: floor(max_total_results / keyword_count)
    # This ensures: 1 keyword -> 200, 5 keywords -> 40, 10 keywords -> 20, etc.
    $adjusted_budget = [Math]::Floor($MaxTotalResults / $keyword_count)
    $effectiveTotal = $keyword_count * $adjusted_budget

    # Determine if we're reducing from the base results_per_keyword
    $willReduce = $adjusted_budget -lt $ResultsPerKeyword

    # Calculate budget threshold
    $budgetThreshold = [Math]::Floor($MaxTotalResults / $ResultsPerKeyword)

    $warningMessage = $null
    if ($willReduce) {
        $warningMsg = "WARNING: $keyword_count keywords exceeds budget threshold of $budgetThreshold. " +
                      "Results per keyword will be reduced from $ResultsPerKeyword to $adjusted_budget to stay within $MaxTotalResults limit."
        $warningMessage = $warningMsg
    }

    return @{
        adjusted_budget = $adjusted_budget
        keyword_count = $keyword_count
        willReduce = $willReduce
        warningMessage = $warningMessage
        effectiveTotal = $effectiveTotal
    }
}

function Get-ChapterDistributedKeywords {
    <#
    .SYNOPSIS
        Splits chapter patterns into individual search terms
    .DESCRIPTION
        Handles chapter-based keywords like "Chapter 1: Introduction",
        "Section A - Overview", numbered lists, comma/semicolon-separated
        topics like "Feature: Part 1, Part 2, Part 3", etc., and splits them
        into atomic search terms
    .PARAMETER Keywords
        Array of keyword strings that may contain chapter/section patterns
    .RETURNS
        Array of individual search terms
    #>
    param(
        [Parameter(Mandatory=$true)]
        [AllowEmptyCollection()]
        [string[]]$Keywords
    )

    if ($null -eq $Keywords -or $Keywords.Count -eq 0) {
        return @()
    }

    $individualTerms = @()

    foreach ($keyword in $Keywords) {
        if ([string]::IsNullOrWhiteSpace($keyword)) {
            continue
        }

        # First, check if this is a comma or semicolon separated list
        # Pattern: "Feature: Part 1, Part 2, Part 3" or "Auth, Rate Limit, Caching"
        $separatedParts = @()
        $prefix = ""
        if ($keyword -match ',' -or $keyword -match ';') {
            # Extract prefix if present (e.g., "Feature:" from "Feature: Part 1, Part 2")
            if ($keyword -match '^([^:]+:\s*)') {
                $prefix = $matches[1].Trim()
            }
            # Split on comma or semicolon
            $separatedParts = $keyword -split '[;,]' | ForEach-Object { $_.Trim() }
        } else {
            $separatedParts = @($keyword)
        }

        # Process each part (either from split or original keyword)
        $isFirst = $true
        foreach ($part in $separatedParts) {
            # Allow single-character numeric parts (like "1", "2", "3")
            $isNumeric = $part -match '^\d+$'
            if ([string]::IsNullOrWhiteSpace($part) -or ($part.Length -lt 2 -and -not $isNumeric)) {
                continue
            }

            # Apply prefix to subsequent parts if this is a comma-separated list
            $processedPart = $part
            if ($prefix -and -not $isFirst) {
                # Check if part starts with "Part X" or "Chapter X" or number - if so, add prefix
                if ($part -match '^(?i)(part|chapter|section|step)\s*\d' -or $part -match '^\d+[\.\)]') {
                    $processedPart = "$prefix $part"
                }
            }
            $isFirst = $false

            # Split on common chapter/section patterns
            # Pattern 1: "Chapter X: Title" or "Chapter X - Title"
            # Pattern 2: "Section X. Title" or "Section X - Title"
            # Pattern 3: "Part X: Title"
            # Pattern 4: Numbered lists: "1. Title", "2) Title"
            # Pattern 5: Hyphen-separated: "X - Title" or "X: Title"
            # Pattern 6: Standalone "Part X" or "Chapter X" without title

            $patterns = @(
                '(?i)^(chapter|section|part|step)\s*(\d+|[ivxlcdm]+)\s*[:\-]?\s*(.*)$',
                '^\s*(\d+[\.\)]\s*)(.+)',
                '^\s*([ivxlcdm]+[\.\)]\s*)(.+)',
                '^\s*([A-Z]\.\s*)(.+)'
            )

            $matched = $false

            foreach ($pattern in $patterns) {
                if ($processedPart -match $pattern) {
                    $matched = $true

                    # Add the full part
                    $individualTerms += $processedPart.Trim()

                    # Also add just the title part (after chapter/section)
                    if ($matches.Count -ge 4 -and $matches[3]) {
                        $titlePart = $matches[3].Trim()
                        if ($titlePart -and $titlePart.Length -gt 2) {
                            $individualTerms += $titlePart
                        }
                    }
                    # For patterns like "Part 1" without title, add just the chapter part
                    elseif ($matches.Count -ge 3 -and $matches[2] -and $matches[2].Trim()) {
                        $chapterPart = ($matches[1], $matches[2] -join ' ').Trim()
                        if ($chapterPart -and $chapterPart.Length -gt 2) {
                            $individualTerms += $chapterPart
                        }
                    }

                    break
                }
            }

            if (-not $matched) {
                # Check for "top N" patterns (e.g., "Top 5 tips", "top 10 features")
                # This pattern indicates the user wants N specific items
                # Pattern: "Top" or "Best" followed by number, then the topic
                if ($processedPart -match '(?i)^(top|best)\s+(\d+)\s+(.+)$') {
                    $individualTerms += $processedPart.Trim()

                    # Also add the core topic without the "top N" prefix
                    if ($matches.Count -ge 4 -and $matches[3]) {
                        $topicPart = $matches[3].Trim()
                        if ($topicPart -and $topicPart.Length -gt 2) {
                            $individualTerms += $topicPart
                        }
                    }
                }
                else {
                    # No chapter pattern found - add as-is
                    $individualTerms += $processedPart.Trim()
                }
            }
        }
    }

    # Remove duplicates while preserving order
    $seen = @{}
    $uniqueTerms = @()
    foreach ($term in $individualTerms) {
        $key = $term.ToLower()
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $uniqueTerms += $term
        }
    }

    return $uniqueTerms
}

function Show-SearchBudgetStatus {
    <#
    .SYNOPSIS
        Displays search budget status in the interview UI
    .DESCRIPTION
        Shows current budget allocation and warnings when keywords exceed budget.
        Also applies chapter/topic splitting to show effective keyword count.
    .PARAMETER Keywords
        Array of keywords to analyze
    .PARAMETER ShowWarnings
        Whether to show warning messages (default: true)
    .RETURNS
        The budget calculation result from Get-DistributedKeywordBudget
    #>
    param(
        [string[]]$Keywords = @(),
        [bool]$ShowWarnings = $true
    )

    # Get budget info from config
    $budgetInfo = Get-SearchBudgetInfo

    # Apply chapter/topic splitting to keywords
    $splitKeywords = @()
    if ($Keywords.Count -gt 0) {
        $splitKeywords = Get-ChapterDistributedKeywords -Keywords $Keywords
    }

    Write-Host ""
    Write-Host "  ===== Search Budget Status =====" -ForegroundColor Cyan

    if ($budgetInfo.found) {
        Write-Host "  Config: $($budgetInfo.configPath)" -ForegroundColor Gray
    }
    Write-Host "  Max Total Results: $($budgetInfo.maxTotalResults)" -ForegroundColor White
    Write-Host "  Results Per Keyword: $($budgetInfo.resultsPerKeyword)" -ForegroundColor White

    # Show keyword splitting info if applicable
    if ($splitKeywords.Count -gt 0 -and $splitKeywords.Count -ne $Keywords.Count) {
        Write-Host "  Keywords Provided: $($Keywords.Count) → $($splitKeywords.Count) after splitting" -ForegroundColor Cyan
    } else {
        Write-Host "  Keywords Provided: $($Keywords.Count)" -ForegroundColor White
    }

    # Calculate distributed budget using split keywords
    $distributed = Get-DistributedKeywordBudget `
        -Keywords $splitKeywords `
        -MaxTotalResults $budgetInfo.maxTotalResults `
        -ResultsPerKeyword $budgetInfo.resultsPerKeyword

    # Show effective keyword count from budget calculation
    if ($splitKeywords.Count -gt 0) {
        Write-Host "  Effective Keywords: $($distributed.totalKeywords)" -ForegroundColor White
    }

    # Display the exact budget message format required by acceptance criteria
    if ($splitKeywords.Count -gt 0) {
        $budgetMessage = "Budget: $($budgetInfo.maxTotalResults) max | $($distributed.totalKeywords) keywords = $($distributed.adjusted_budget) results each"
        Write-Host "  $budgetMessage" -ForegroundColor Cyan
    }

    if ($distributed.willReduce -and $ShowWarnings) {
        Write-Host ""
        Write-Host "  $($distributed.warningMessage)" -ForegroundColor Yellow
    }
    elseif ($distributed.totalKeywords -gt 0 -and -not $distributed.willReduce) {
        Write-Host "  Status: OK - Budget sufficient for all keywords" -ForegroundColor Green
    }
    elseif ($distributed.totalKeywords -eq 0) {
        Write-Host "  Status: Waiting for keywords..." -ForegroundColor Gray
    }

    Write-Host "  =================================" -ForegroundColor Cyan
    Write-Host ""

    return $distributed
}

# ============================================================================
# CONTEXT IMPROVEMENT - Convert vague input to actionable context
# ============================================================================

function Improve-InterviewContext {
    <#
    .SYNOPSIS
        Takes vague user input and improves it into specific, actionable context
    .DESCRIPTION
        Uses LLM-orchestrated scan decisions to analyze the codebase and generate
        structured context for sprint generation. Falls back to keyword matching
        if LLM is unavailable.
    .PARAMETER RawInput
        The user's vague context string (e.g., "ratelimiting on caption, refactoring god files")
    .PARAMETER FocusAreas
        Optional array of focus areas to constrain the analysis
    .PARAMETER NoLLM
        Skip LLM analysis and use keyword-based fallback only
    .RETURNS
        Hashtable with: improvedContext (string), detectedAreas (array), codebaseFindings (hashtable)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$RawInput,
        [array]$FocusAreas = @(),
        [switch]$NoLLM
    )

    Write-Host ""
    Write-Host "  Analyzing context and scanning codebase..." -ForegroundColor Cyan

    $findings = @{
        largeFiles = @()
        relevantCode = @()
        existingPatterns = @()
        configOptions = @()
    }
    $detectedAreas = @()

    # Determine project root
    $projectRoot = if ($script:ProjectRoot) { $script:ProjectRoot } else { Split-Path -Parent (Split-Path -Parent $script:RalphDir) }

    # Get scan decisions (LLM or keyword-based)
    if ($NoLLM) {
        Write-Host "    Using keyword-based scan decisions..." -ForegroundColor Gray
        $scanDecisions = Get-KeywordScanDecisions -UserInput $RawInput
    } else {
        Write-Host "    Asking LLM to analyze problem..." -ForegroundColor Gray
        $scanDecisions = Get-LLMScanDecisions -UserInput $RawInput
    }

    # Execute decided scans
    $scanResults = @{}
    foreach ($scan in $scanDecisions.scans) {
        Write-Host "    Running scan: $($scan.id) (priority $($scan.priority))..." -ForegroundColor Gray

        $result = switch ($scan.id) {
            "large_files" { Invoke-LargeFilesScan -ProjectRoot $projectRoot }
            "rate_limit_patterns" { Invoke-RateLimitPatternsScan -ProjectRoot $projectRoot }
            "caption_infrastructure" { Invoke-CaptionInfrastructureScan -ProjectRoot $projectRoot }
            "download_system" { Invoke-DownloadSystemScan -ProjectRoot $projectRoot }
            "test_coverage" { Invoke-TestCoverageScan -ProjectRoot $projectRoot }
            "config_structure" { Invoke-ConfigStructureScan -ProjectRoot $projectRoot }
            "pipeline_stages" { Invoke-PipelineStagesScan -ProjectRoot $projectRoot }
            "error_patterns" { Invoke-ErrorPatternsScan -ProjectRoot $projectRoot }
            "matching_infrastructure" { Invoke-MatchingInfrastructureScan -ProjectRoot $projectRoot }
            default { $null }
        }

        if ($result) {
            $scanResults[$scan.id] = $result

            # Show findings
            if ($result.found) {
                Write-Host "      Found relevant data" -ForegroundColor Green
            }

            # Map scan to focus areas
            $areas = Get-AreasForScan -ScanId $scan.id
            $detectedAreas += $areas
        }
    }

    # Populate legacy findings structure for backward compatibility
    if ($scanResults.ContainsKey("large_files") -and $scanResults["large_files"].found) {
        $findings.largeFiles = $scanResults["large_files"].data.largeFiles
    }
    if ($scanResults.ContainsKey("rate_limit_patterns") -and $scanResults["rate_limit_patterns"].found) {
        $findings.existingPatterns = $scanResults["rate_limit_patterns"].data.existingPatterns
    }
    if ($scanResults.ContainsKey("caption_infrastructure") -and $scanResults["caption_infrastructure"].found) {
        $findings.relevantCode += $scanResults["caption_infrastructure"].data.captionFiles
    }
    if ($scanResults.ContainsKey("download_system") -and $scanResults["download_system"].found) {
        $findings.relevantCode += $scanResults["download_system"].data.downloadFiles
    }
    if ($scanResults.ContainsKey("matching_infrastructure") -and $scanResults["matching_infrastructure"].found) {
        $findings.relevantCode += $scanResults["matching_infrastructure"].data.matchingFiles
    }
    if ($scanResults.ContainsKey("config_structure") -and $scanResults["config_structure"].found) {
        $findings.configOptions = $scanResults["config_structure"].data.configSections
    }

    # Generate improved context with LLM framing (or template fallback)
    if ($NoLLM) {
        $improvedSpecs = Format-ScanResultsTemplate -ScanResults $scanResults
    } else {
        Write-Host "    Generating LLM-framed specifications..." -ForegroundColor Gray
        $framing = if ($scanDecisions.framing) { $scanDecisions.framing } else { "General codebase improvement" }
        $improvedSpecs = Format-ScanResultsWithLLM `
            -UserInput $RawInput `
            -ScanResults $scanResults `
            -Framing $framing
    }

    # Build improved context
    $improvedContext = "Context: $RawInput`n`n$improvedSpecs"

    # Remove duplicates from detected areas
    $detectedAreas = @($detectedAreas | Select-Object -Unique)

    Write-Host ""
    Write-Host "  Context improved!" -ForegroundColor Green
    if ($detectedAreas.Count -gt 0) {
        Write-Host "  Detected areas: $($detectedAreas -join ', ')" -ForegroundColor Cyan
    }

    return @{
        improvedContext = $improvedContext
        detectedAreas = $detectedAreas
        codebaseFindings = $findings
        rawInput = $RawInput
    }
}

function Show-ImprovedContext {
    <#
    .SYNOPSIS
        Displays the improved context and asks for user confirmation
    .PARAMETER ImprovedResult
        The result from Improve-InterviewContext
    .RETURNS
        $true if user approves, $false to edit
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$ImprovedResult
    )

    Write-Host ""
    Write-Host "  ===== Improved Context =====" -ForegroundColor Cyan
    Write-Host ""
    Write-Host $ImprovedResult.improvedContext -ForegroundColor White
    Write-Host ""
    Write-Host "  =============================" -ForegroundColor Cyan
    Write-Host ""

    if ($ImprovedResult.codebaseFindings.largeFiles.Count -gt 0) {
        Write-Host "  Large files found:" -ForegroundColor Yellow
        foreach ($file in $ImprovedResult.codebaseFindings.largeFiles | Select-Object -First 5) {
            Write-Host "    - $($file.path): $($file.lines) lines" -ForegroundColor Gray
        }
        Write-Host ""
    }

    Write-Host "  Use this improved context? [Y]es / [E]dit / [O]riginal" -ForegroundColor Yellow
    $response = Read-Host "  "

    switch -Regex ($response) {
        "^[Yy]" { return @{ approved = $true; context = $ImprovedResult.improvedContext } }
        "^[Oo]" { return @{ approved = $true; context = $ImprovedResult.rawInput } }
        "^[Ee]" {
            Write-Host ""
            Write-Host "  Enter your edited context (or paste improved version):" -ForegroundColor Cyan
            $edited = Read-Host "  "
            return @{ approved = $true; context = $edited }
        }
        default { return @{ approved = $true; context = $ImprovedResult.improvedContext } }
    }
}

# ============================================================================
# LLM-BASED AREA SUGGESTION
# ============================================================================

function Get-LLMSuggestedAreas {
    <#
    .SYNOPSIS
        Uses Claude to analyze problem and suggest focus areas
    .PARAMETER Problem
        The problem description from the user
    .RETURNS
        Hashtable with: areas (array), reasoning (string), success (bool)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Problem
    )

    # Known areas in this codebase (for context)
    $knownAreas = @(
        "download", "caption", "transcription", "matching", "otio", "output",
        "config", "testing", "agents", "healing", "speed", "rate-limiting",
        "architecture", "refactoring", "quality", "pipeline", "compilation"
    )

    $prompt = @"
Analyze this problem description and determine which areas of the codebase need work.

Problem: $Problem

Known focus areas in this project:
$($knownAreas -join ", ")

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "areas": ["area1", "area2", "area3"],
  "reasoning": "brief explanation",
  "files_to_check": ["optional list of likely files"]
}

Rules:
- Return 2-5 areas from the known list above
- Order by relevance (most relevant first)
- If the problem mentions "god file" or "large file", include "refactoring" and "architecture"
- If the problem mentions rate limiting, throttling, or 429 errors, include "rate-limiting"
"@

    try {
        Write-Host "  Asking Claude to analyze problem..." -ForegroundColor Gray

        # Use claude CLI with --print for simple one-shot response
        $claudePath = Get-Command "claude" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
        if (-not $claudePath) {
            $claudePath = "claude"
        }

        # Run claude with timeout
        $result = $null
        $job = Start-Job -ScriptBlock {
            param($prompt, $claudePath)
            $output = echo $prompt | & $claudePath --print 2>&1
            return $output
        } -ArgumentList $prompt, $claudePath

        $completed = Wait-Job $job -Timeout 30
        if ($completed) {
            $result = Receive-Job $job
        }
        Remove-Job $job -Force -ErrorAction SilentlyContinue

        if (-not $result) {
            Write-Host "  LLM timeout, falling back to keywords..." -ForegroundColor Yellow
            return @{ success = $false; areas = @(); reasoning = "timeout" }
        }

        # Parse JSON from response (handle markdown code blocks)
        $jsonText = $result -join "`n"
        if ($jsonText -match '```json\s*([\s\S]*?)\s*```') {
            $jsonText = $matches[1]
        } elseif ($jsonText -match '```\s*([\s\S]*?)\s*```') {
            $jsonText = $matches[1]
        }

        # Find JSON object in response
        if ($jsonText -match '\{[\s\S]*\}') {
            $jsonText = $matches[0]
        }

        $parsed = $jsonText | ConvertFrom-Json -ErrorAction Stop

        if ($parsed.areas -and $parsed.areas.Count -gt 0) {
            Write-Host "  LLM suggested: $($parsed.areas -join ', ')" -ForegroundColor Green
            if ($parsed.reasoning) {
                Write-Host "  Reasoning: $($parsed.reasoning)" -ForegroundColor DarkGray
            }
            return @{
                success = $true
                areas = @($parsed.areas)
                reasoning = $parsed.reasoning
                filesToCheck = $parsed.files_to_check
            }
        }
    }
    catch {
        Write-Host "  LLM parse error: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    return @{ success = $false; areas = @(); reasoning = "parse_error" }
}

function Get-KeywordSuggestedAreas {
    <#
    .SYNOPSIS
        Fallback: suggests focus areas based on keyword matching
    .PARAMETER Problem
        The problem description from the user
    .RETURNS
        Array of suggested focus area IDs (max 5)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Problem
    )

    $problemLower = $Problem.ToLower()
    $suggestions = @()

    # Keyword-to-area mapping (patterns on left, areas on right)
    $keywordMap = @{
        "slow|timeout|hang|performance|speed" = @("speed", "rate-limiting", "download")
        "download|youtube|yt-dlp|429|rate.limit" = @("download", "rate-limiting")
        "cookie|403|forbidden|block|bot" = @("download", "rate-limiting")
        "caption|subtitle|srt|transcript|whisper" = @("caption", "transcription")
        "match|confidence|score|quality|poor" = @("quality", "matching")
        "otio|timeline|edl|xml|davinci|resolve" = @("otio", "output")
        "config|yaml|setting" = @("config")
        "test|coverage|pytest|fail" = @("testing", "quality")
        "client|feedback|theresa|stu" = @("client-learning")
        "heal|recover|retry|error|fail" = @("agents", "healing")
        "embed|vector|semantic" = @("matching", "transcription")
        "memory|leak|resource" = @("speed", "agents")
        "api|anthropic|claude|gemini|llm" = @("agents", "config")
        "compile|compilation|topic|keyword" = @("compilation", "keyword")
        "refactor|god.?file|split|extract|large.?file|modular" = @("refactoring", "architecture")
        "rate.?limit|throttle|quota|too.?many|429" = @("rate-limiting", "caption")
        "mullvad|vpn|ip.?rotat|geograph|server.?switch" = @("mullvad-vpn", "rate-limiting")
        "fetch|cache|async|concurrent" = @("speed", "architecture")
        "chapter|listicle|group|coherence|section|numbered" = @("listicle-matching", "quality")
        "context|title|description|metadata|enrich" = @("context-matching", "caption")
    }

    foreach ($pattern in $keywordMap.Keys) {
        if ($problemLower -match $pattern) {
            foreach ($area in $keywordMap[$pattern]) {
                if ($suggestions -notcontains $area) {
                    $suggestions += $area
                }
            }
        }
    }

    # Cap at 5 suggestions
    if ($suggestions.Count -gt 5) {
        $suggestions = $suggestions[0..4]
    }

    # If no matches, return default areas
    if ($suggestions.Count -eq 0) {
        $suggestions = @("pipeline", "quality", "testing")
    }

    return $suggestions
}

function Get-SuggestedAreas {
    <#
    .SYNOPSIS
        Suggests focus areas - tries LLM first, falls back to keywords
    .PARAMETER Problem
        The problem description from the user
    .PARAMETER UseLLM
        Whether to try LLM first (default: true)
    .RETURNS
        Array of suggested focus area IDs (max 5)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Problem,
        [switch]$NoLLM
    )

    # Try LLM first unless disabled
    if (-not $NoLLM) {
        $llmResult = Get-LLMSuggestedAreas -Problem $Problem
        if ($llmResult.success -and $llmResult.areas.Count -gt 0) {
            return $llmResult.areas
        }
        Write-Host "  Falling back to keyword matching..." -ForegroundColor Yellow
    }

    # Fallback to keyword-based
    return Get-KeywordSuggestedAreas -Problem $Problem
}

# ============================================================================
# DEEP INTERVIEW FLOW
# ============================================================================

function Invoke-DeepInterview {
    <#
    .SYNOPSIS
        Conducts a deep interview with dynamic follow-up questions
    .DESCRIPTION
        Asks follow-up questions based on keyword detection in responses.
        Continues until enough context is gathered or user says 'done'.
    .PARAMETER AutoAreas
        If true, automatically uses suggested areas without prompting
    .PARAMETER AutoPriority
        If set, skips priority question and uses this value (critical/high/medium/low)
    .RETURNS
        Hashtable with: problem, symptoms, areas, priority, constraints, timeline
        or $null if user cancels
    #>
    param(
        [switch]$AutoAreas,
        [string]$AutoPriority = ""
    )

    Write-Host ""
    Write-Host "=== Ralph Deep Interview ===" -ForegroundColor Cyan
    Write-Host "Let's understand what you need so I can generate the best stories." -ForegroundColor Gray
    Write-Host "(Type 'done' when ready, 'skip' to use my choice)" -ForegroundColor DarkGray
    Write-Host ""

    $context = @{
        problem = ""
        symptoms = [System.Collections.ArrayList]::new()
        areas = @()
        priority = "medium"
        constraints = [System.Collections.ArrayList]::new()
        timeline = ""
    }

    # ---- Question 1: Open-ended problem ----
    $context.problem = Read-Host "What problem are you trying to solve?"

    if ($context.problem -eq "skip") {
        Write-Host "  Using Ralph's Choice for focus areas..." -ForegroundColor Magenta
        return $null
    }

    if ($context.problem -eq "done" -or [string]::IsNullOrWhiteSpace($context.problem)) {
        Write-Host "  No problem specified. Using Ralph's Choice..." -ForegroundColor Magenta
        return $null
    }

    # ---- Dynamic follow-ups based on keywords ----

    # Performance/timeout follow-ups
    if ($context.problem -match "slow|timeout|hang|performance") {
        Write-Host ""
        Write-Host "  Sounds like a performance issue. Let me understand better..." -ForegroundColor Yellow

        $symptom = Read-Host "  When does it slow down? (specific operation/time)"
        if (-not [string]::IsNullOrWhiteSpace($symptom)) {
            [void]$context.symptoms.Add("When: $symptom")
        }

        $freq = Read-Host "  How often? ([A]lways, [S]ometimes, [R]arely)"
        $freqMap = @{ "A" = "Always"; "S" = "Sometimes"; "R" = "Rarely" }
        $freqText = if ($freqMap[$freq.ToUpper()]) { $freqMap[$freq.ToUpper()] } else { $freq }
        [void]$context.symptoms.Add("Frequency: $freqText")
    }

    # Error/crash follow-ups
    if ($context.problem -match "error|crash|fail|exception|broken") {
        Write-Host ""
        Write-Host "  Let me get some error details..." -ForegroundColor Yellow

        $errorMsg = Read-Host "  What's the error message? (paste or summarize)"
        if (-not [string]::IsNullOrWhiteSpace($errorMsg)) {
            [void]$context.symptoms.Add("Error: $errorMsg")
        }

        $repro = Read-Host "  Can you reproduce it consistently? [Y/N/Sometimes]"
        [void]$context.symptoms.Add("Reproducible: $repro")
    }

    # Download/rate-limit follow-ups
    if ($context.problem -match "download|429|rate.limit|cookie|403|forbidden") {
        Write-Host ""
        Write-Host "  Download issues often have specific patterns..." -ForegroundColor Yellow

        $when = Read-Host "  When does it fail? (time of day, after N videos, random)"
        if (-not [string]::IsNullOrWhiteSpace($when)) {
            [void]$context.symptoms.Add("Timing: $when")
        }

        $channels = Read-Host "  Specific channels/videos or all? (specific/all)"
        [void]$context.symptoms.Add("Scope: $channels")
    }

    # Quality/matching follow-ups
    if ($context.problem -match "match|quality|wrong|poor|bad") {
        Write-Host ""
        Write-Host "  Let me understand the quality issue..." -ForegroundColor Yellow

        $example = Read-Host "  Can you give an example? (what was expected vs what happened)"
        if (-not [string]::IsNullOrWhiteSpace($example)) {
            [void]$context.symptoms.Add("Example: $example")
        }
    }

    # ---- Question 2: Scope/Areas ----
    $suggested = Get-SuggestedAreas -Problem $context.problem

    if ($AutoAreas) {
        # Auto-mode: use suggested areas without prompting
        Write-Host ""
        Write-Host "  Auto-detected areas from problem:" -ForegroundColor Green
        foreach ($area in $suggested) {
            Write-Host "    - $area" -ForegroundColor White
        }
        $context.areas = $suggested
    } else {
        Write-Host ""
        $scope = Read-Host "Which areas might be involved? (comma-separated, or 'not sure')"

        if ($scope -eq "not sure" -or [string]::IsNullOrWhiteSpace($scope)) {
            # Ralph suggests based on problem
            Write-Host ""
            Write-Host "  Based on your description, these areas might be relevant:" -ForegroundColor Green
            foreach ($area in $suggested) {
                Write-Host "    - $area" -ForegroundColor White
            }
            Write-Host ""
            $useSuggested = Read-Host "  Use these? [Y]es / [N]o / [M]odify"

            switch -Regex ($useSuggested) {
                "^[Yy]" {
                    $context.areas = $suggested
                }
                "^[Mm]" {
                    $modified = Read-Host "  Enter areas (comma-separated)"
                    $context.areas = $modified -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
                }
                default {
                    $newAreas = Read-Host "  Enter areas (comma-separated)"
                    $context.areas = $newAreas -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
                }
            }
        } else {
            $context.areas = $scope -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        }
    }

    # ---- Question 3: Constraints ----
    if (-not $AutoAreas) {
        Write-Host ""
        $hasConstraints = Read-Host "Any constraints? (time, backward compatibility, specific tools) [Y/N]"

        if ($hasConstraints -match "^[Yy]") {
            do {
                $constraint = Read-Host "  Enter constraint (or 'done')"
                if ($constraint -ne "done" -and -not [string]::IsNullOrWhiteSpace($constraint)) {
                    [void]$context.constraints.Add($constraint)
                }
            } while ($constraint -ne "done")
        }
    }

    # ---- Question 4: Priority & Timeline ----
    $priorityMap = @{
        "C" = "critical"
        "H" = "high"
        "M" = "medium"
        "L" = "low"
    }

    if ($AutoPriority) {
        # Auto-mode: use provided priority
        $context.priority = if ($priorityMap[$AutoPriority.ToUpper()]) {
            $priorityMap[$AutoPriority.ToUpper()]
        } elseif ($AutoPriority -in @("critical", "high", "medium", "low")) {
            $AutoPriority
        } else {
            "medium"
        }
        $context.timeline = "This week"
        Write-Host "  Auto-set priority: $($context.priority)" -ForegroundColor Gray
    } else {
        Write-Host ""
        $priorityInput = Read-Host "Priority? [C]ritical (blocking), [H]igh, [M]edium, [L]ow"
        $context.priority = if ($priorityMap[$priorityInput.ToUpper()]) {
            $priorityMap[$priorityInput.ToUpper()]
        } else {
            "medium"
        }

        $timeline = Read-Host "When do you need this? [A]SAP, [T]his week, [N]ext week, [M]onth"
        $timelineMap = @{
            "A" = "ASAP"
            "T" = "This week"
            "N" = "Next week"
            "M" = "This month"
        }
        $context.timeline = if ($timelineMap[$timeline.ToUpper()]) {
            $timelineMap[$timeline.ToUpper()]
        } else {
            $timeline
        }
    }

    # ---- Summary & Confirmation ----
    Write-Host ""
    Write-Host "=== Interview Summary ===" -ForegroundColor Cyan
    Write-Host "Problem: $($context.problem)" -ForegroundColor White

    if ($context.symptoms.Count -gt 0) {
        Write-Host "Symptoms:" -ForegroundColor White
        foreach ($symptom in $context.symptoms) {
            Write-Host "  - $symptom" -ForegroundColor Gray
        }
    }

    Write-Host "Areas: $($context.areas -join ', ')" -ForegroundColor White
    Write-Host "Priority: $($context.priority)" -ForegroundColor White

    if ($context.constraints.Count -gt 0) {
        Write-Host "Constraints:" -ForegroundColor White
        foreach ($constraint in $context.constraints) {
            Write-Host "  - $constraint" -ForegroundColor Gray
        }
    }

    Write-Host "Timeline: $($context.timeline)" -ForegroundColor White
    Write-Host ""

    if ($AutoAreas) {
        # Auto-mode: skip confirmation
        Write-Host "  Auto-proceeding with PRD generation..." -ForegroundColor Green
        $context.symptoms = @($context.symptoms)
        $context.constraints = @($context.constraints)
        return $context
    }

    $confirm = Read-Host "Generate PRD based on this? [Y]es / [N]o / [R]edo"

    switch -Regex ($confirm) {
        "^[Rr]" {
            Write-Host "  Restarting interview..." -ForegroundColor Yellow
            return Invoke-DeepInterview -AutoAreas:$AutoAreas -AutoPriority $AutoPriority
        }
        "^[Nn]" {
            Write-Host "  Interview cancelled." -ForegroundColor Red
            return $null
        }
        default {
            # Convert ArrayLists to arrays for cleaner return
            $context.symptoms = @($context.symptoms)
            $context.constraints = @($context.constraints)
            return $context
        }
    }
}

# ============================================================================
# INTERVIEW CONTEXT TO QUEUE CONVERSION
# ============================================================================

function ConvertTo-QueueContext {
    <#
    .SYNOPSIS
        Converts deep interview context to queue-compatible format
    .PARAMETER DeepContext
        The context hashtable from Invoke-DeepInterview
    .RETURNS
        Hashtable compatible with Save-InterviewQueue
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$DeepContext
    )

    # Build detailed description from problem and symptoms
    $description = $DeepContext.problem
    if ($DeepContext.symptoms -and $DeepContext.symptoms.Count -gt 0) {
        $description += " | " + ($DeepContext.symptoms -join "; ")
    }
    if ($DeepContext.constraints -and $DeepContext.constraints.Count -gt 0) {
        $description += " | Constraints: " + ($DeepContext.constraints -join ", ")
    }

    # Map priority
    $priorityMap = @{
        "critical" = "high"
        "high" = "high"
        "medium" = "normal"
        "low" = "low"
    }

    return @{
        workType = if ($DeepContext.problem -match "bug|error|crash|fail") { "bug" } else { "improvement" }
        details = $description
        area = if ($DeepContext.areas.Count -gt 0) { $DeepContext.areas[0] } else { "" }
        client = ""
        priority = $priorityMap[$DeepContext.priority]
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        deepContext = $DeepContext  # Preserve full context for PRD generation
    }
}

# ============================================================================
# LOGGING FUNCTION
# ============================================================================

function Write-InterviewLog {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"

    # Create logs directory if needed
    $logsDir = Join-Path $script:RalphDir "logs"
    if (-not (Test-Path $logsDir)) {
        New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    }

    # Log to interview-specific log file
    $logFile = Join-Path $logsDir "interview.log"
    Add-Content -Path $logFile -Value $logLine

    # Also write to verbose stream for debugging
    Write-Verbose $logLine
}

# ============================================================================
# QUEUE SAVE FUNCTION
# ============================================================================

function Save-InterviewQueue {
    <#
    .SYNOPSIS
        Saves the interview context and focus areas to queue.json
    .PARAMETER Context
        The interview context hashtable from Start-Interview
    .PARAMETER FocusAreas
        Array or ArrayList of approved focus area IDs
    .RETURNS
        The queue hashtable that was saved
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Context,
        [Parameter(Mandatory=$true)]
        $FocusAreas
    )

    # Build focus area objects with tracking fields
    $focusAreaObjects = @()
    foreach ($area in $FocusAreas) {
        $focusAreaObjects += @{
            id = $area
            completed = $false
            startedAt = $null
            completedAt = $null
        }
    }

    # Create the queue structure
    $queue = @{
        interviewContext = "$($Context.workType): $($Context.details)"
        interviewDetails = $Context
        focusAreas = $focusAreaObjects
        createdAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        sessionId = [guid]::NewGuid().ToString().Substring(0, 8)
    }

    # Save to queue.json (atomic write)
    Save-StateFile -Path $script:QueueFile -Data $queue

    Write-Host "  Saved queue with $($FocusAreas.Count) focus areas" -ForegroundColor Green
    Write-InterviewLog "Queue saved - Session: $($queue.sessionId), Areas: $($FocusAreas.Count)"

    return $queue
}

# ============================================================================
# LLM-ORCHESTRATED SCAN SYSTEM
# ============================================================================

function Get-ScanRegistry {
    <#
    .SYNOPSIS
        Load the scan registry configuration
    .RETURNS
        Hashtable with scan definitions
    #>
    $registryPath = Join-Path $script:RalphDir "config\scan-registry.json"
    if (-not (Test-Path $registryPath)) {
        Write-Host "  Warning: scan-registry.json not found" -ForegroundColor Yellow
        return $null
    }

    try {
        $registry = Get-Content $registryPath -Raw | ConvertFrom-Json
        return $registry
    }
    catch {
        Write-Host "  Warning: Failed to parse scan-registry.json: $_" -ForegroundColor Yellow
        return $null
    }
}

# ============================================================================
# MODULAR SCAN EXECUTORS
# ============================================================================

function Invoke-LargeFilesScan {
    <#
    .SYNOPSIS
        Find Python files over 400 lines that may need refactoring
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $srcPath = Join-Path $ProjectRoot "src"
    $largeFiles = @()

    if (Test-Path $srcPath) {
        $largeFiles = Get-ChildItem -Path $srcPath -Recurse -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
            $lineCount = (Get-Content $_.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
            if ($lineCount -gt 400) {
                @{
                    path = $_.FullName.Replace($ProjectRoot, "").TrimStart("\", "/")
                    lines = $lineCount
                }
            }
        } | Where-Object { $_ } | Sort-Object -Property lines -Descending | Select-Object -First 10
    }

    return @{
        scanId = "large_files"
        found = ($largeFiles.Count -gt 0)
        data = @{ largeFiles = @($largeFiles) }
    }
}

function Invoke-RateLimitPatternsScan {
    <#
    .SYNOPSIS
        Check existing rate limiting, backoff, and throttling patterns
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $patternFiles = @(
        "src/downloader/rate_limit_budget.py",
        "src/downloader/cookie_rotator.py",
        "src/caption_fetcher.py",
        "src/caption_timeout_manager.py",
        "src/rate_limit/"
    )

    $existing = @()
    $missing = @()

    foreach ($file in $patternFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $existing += $file
        } else {
            $missing += $file
        }
    }

    return @{
        scanId = "rate_limit_patterns"
        found = ($existing.Count -gt 0)
        data = @{
            existingPatterns = $existing
            missingPatterns = $missing
        }
    }
}

function Invoke-CaptionInfrastructureScan {
    <#
    .SYNOPSIS
        Analyze caption fetching, caching, and error handling
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $captionFiles = @(
        "src/caption_fetcher.py",
        "src/caption_fetcher_cache.py",
        "src/caption_timeout_manager.py",
        "src/stages/caption_stage.py",
        "src/caption/"
    )

    $foundFiles = @()
    foreach ($file in $captionFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $foundFiles += $file
        }
    }

    return @{
        scanId = "caption_infrastructure"
        found = ($foundFiles.Count -gt 0)
        data = @{ captionFiles = $foundFiles }
    }
}

function Invoke-DownloadSystemScan {
    <#
    .SYNOPSIS
        Analyze yt-dlp integration, bypass mechanisms, cookie handling
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $downloadFiles = @(
        "src/downloader/core.py",
        "src/downloader/rate_limit_budget.py",
        "src/downloader/impersonation.py",
        "src/downloader/escalation.py",
        "src/downloader/escalation_manager.py",
        "src/downloader/circuit_breaker.py"
    )

    $foundFiles = @()
    foreach ($file in $downloadFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $foundFiles += $file
        }
    }

    return @{
        scanId = "download_system"
        found = ($foundFiles.Count -gt 0)
        data = @{ downloadFiles = $foundFiles }
    }
}

function Invoke-TestCoverageScan {
    <#
    .SYNOPSIS
        Count tests, identify untested modules, find coverage gaps
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $testDir = Join-Path $ProjectRoot "tests"
    $srcDir = Join-Path $ProjectRoot "src"

    $testCount = 0
    $testFiles = @()
    $srcModules = @()

    if (Test-Path $testDir) {
        $testFileObjs = Get-ChildItem -Path $testDir -Recurse -Filter "test_*.py" -ErrorAction SilentlyContinue
        $testCount = ($testFileObjs | Measure-Object).Count
        $testFiles = $testFileObjs | Select-Object -ExpandProperty Name -First 20
    }

    if (Test-Path $srcDir) {
        $srcModules = Get-ChildItem -Path $srcDir -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
    }

    # Find modules without corresponding tests
    $untestedModules = @()
    foreach ($module in $srcModules) {
        $hasTest = $testFiles | Where-Object { $_ -match "test_$module" -or $_ -match "${module}_test" }
        if (-not $hasTest) {
            $untestedModules += $module
        }
    }

    return @{
        scanId = "test_coverage"
        found = $true
        data = @{
            testCount = $testCount
            srcModules = $srcModules
            untestedModules = $untestedModules
            testFiles = $testFiles
        }
    }
}

function Invoke-ConfigStructureScan {
    <#
    .SYNOPSIS
        List config sections and their current state
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $configSectionsPath = Join-Path $ProjectRoot "src/config/sections"
    $configYamlPath = Join-Path $ProjectRoot "config.yaml"

    $configSections = @()
    $yamlExists = Test-Path $configYamlPath

    if (Test-Path $configSectionsPath) {
        $configSections = Get-ChildItem -Path $configSectionsPath -Filter "*.py" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "__init__.py" } |
            Select-Object -ExpandProperty Name |
            ForEach-Object { $_ -replace "\.py$", "" }
    }

    return @{
        scanId = "config_structure"
        found = ($configSections.Count -gt 0 -or $yamlExists)
        data = @{
            configSections = $configSections
            yamlExists = $yamlExists
        }
    }
}

function Invoke-PipelineStagesScan {
    <#
    .SYNOPSIS
        Analyze pipeline stages, execution order, dependencies
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $stagesPath = Join-Path $ProjectRoot "src/stages"
    $stages = @()

    if (Test-Path $stagesPath) {
        $stages = Get-ChildItem -Path $stagesPath -Filter "*_stage.py" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name |
            ForEach-Object { $_ -replace "_stage\.py$", "" }
    }

    return @{
        scanId = "pipeline_stages"
        found = ($stages.Count -gt 0)
        data = @{ stages = $stages }
    }
}

function Invoke-ErrorPatternsScan {
    <#
    .SYNOPSIS
        Scan logs and code for common error patterns
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $errorPatterns = @()

    # Check for healing-related files
    $healingFiles = @(
        "src/agents/resilient_runner.py",
        "src/agents/healing.py"
    )

    $foundHealingFiles = @()
    foreach ($file in $healingFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $foundHealingFiles += $file
        }
    }

    # Check recent log files for error patterns
    $logDir = Join-Path $ProjectRoot "logs"
    $recentErrors = @()
    if (Test-Path $logDir) {
        $recentLogs = Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 3

        foreach ($log in $recentLogs) {
            $content = Get-Content $log.FullName -Tail 100 -ErrorAction SilentlyContinue
            $errors = $content | Select-String -Pattern "ERROR|Exception|Traceback" -SimpleMatch | Select-Object -First 5
            if ($errors) {
                $recentErrors += @{
                    file = $log.Name
                    errors = @($errors | ForEach-Object { $_.Line.Substring(0, [Math]::Min(100, $_.Line.Length)) })
                }
            }
        }
    }

    return @{
        scanId = "error_patterns"
        found = ($foundHealingFiles.Count -gt 0 -or $recentErrors.Count -gt 0)
        data = @{
            healingFiles = $foundHealingFiles
            recentErrors = $recentErrors
        }
    }
}

function Invoke-MatchingInfrastructureScan {
    <#
    .SYNOPSIS
        Analyze matching pipeline, confidence scoring, chapter detection, and context usage
    .PARAMETER ProjectRoot
        Root directory of the project
    .RETURNS
        Hashtable with scanId, found (bool), and data
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectRoot
    )

    $matchingFiles = @(
        "src/matching/scoring.py",
        "src/matching/llm_reranker.py",
        "src/matching/tiered_matcher.py",
        "src/matching/embedding_search.py",
        "src/stages/match.py",
        "src/stages/iterative_match.py",
        "src/caption/models.py",
        "src/caption_fetcher.py",
        "src/state.py",
        "src/config/sections/matching.py"
    )

    $chapterFiles = @(
        "src/chapter_detection/",
        "src/chapter_detection/detector.py",
        "src/chapter_detection/listicle_detector.py"
    )

    $foundMatchingFiles = @()
    foreach ($file in $matchingFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $foundMatchingFiles += $file
        }
    }

    $foundChapterFiles = @()
    foreach ($file in $chapterFiles) {
        $fullPath = Join-Path $ProjectRoot $file
        if (Test-Path $fullPath) {
            $foundChapterFiles += $file
        }
    }

    # Check what metadata is currently extracted from yt-dlp
    $contextUsage = @()
    $captionFetcherPath = Join-Path $ProjectRoot "src/caption_fetcher.py"
    if (Test-Path $captionFetcherPath) {
        $content = Get-Content $captionFetcherPath -Raw -ErrorAction SilentlyContinue
        if ($content -match "description") { $contextUsage += "description (referenced)" }
        if ($content -match "chapters") { $contextUsage += "chapters (referenced)" }
        if ($content -match "tags") { $contextUsage += "tags (referenced)" }
        if ($content -match "title") { $contextUsage += "title (referenced)" }
    }

    return @{
        scanId = "matching_infrastructure"
        found = ($foundMatchingFiles.Count -gt 0)
        data = @{
            matchingFiles = $foundMatchingFiles
            chapterFiles = $foundChapterFiles
            contextUsage = $contextUsage
            scoringComponents = @($foundMatchingFiles | Where-Object { $_ -match "scoring|reranker|matcher" })
        }
    }
}

# ============================================================================
# LLM SCAN ORCHESTRATION
# ============================================================================

function Invoke-LLMWithTimeout {
    <#
    .SYNOPSIS
        Invoke Claude CLI with a timeout
    .PARAMETER Prompt
        The prompt to send
    .PARAMETER TimeoutSeconds
        Timeout in seconds (default: 30)
    .RETURNS
        Hashtable with Success (bool) and Data (parsed JSON or raw output)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,
        [int]$TimeoutSeconds = 30
    )

    try {
        $claudePath = Get-Command "claude" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
        if (-not $claudePath) {
            $claudePath = "claude"
        }

        # Run claude with timeout using job
        $job = Start-Job -ScriptBlock {
            param($prompt, $claudePath)
            $output = $prompt | & $claudePath --print 2>&1
            return $output
        } -ArgumentList $Prompt, $claudePath

        $completed = Wait-Job $job -Timeout $TimeoutSeconds
        if ($completed) {
            $result = Receive-Job $job
            Remove-Job $job -Force -ErrorAction SilentlyContinue

            if ($result) {
                # Join array output if necessary
                $jsonText = if ($result -is [array]) { $result -join "`n" } else { $result }

                # Extract JSON from markdown code blocks if present
                if ($jsonText -match '```json\s*([\s\S]*?)\s*```') {
                    $jsonText = $matches[1]
                } elseif ($jsonText -match '```\s*([\s\S]*?)\s*```') {
                    $jsonText = $matches[1]
                }

                # Find JSON object in response
                if ($jsonText -match '\{[\s\S]*\}') {
                    $jsonText = $matches[0]
                }

                try {
                    $parsed = $jsonText | ConvertFrom-Json -ErrorAction Stop
                    return @{ Success = $true; Data = $parsed }
                }
                catch {
                    # Return raw text if not valid JSON
                    return @{ Success = $true; Data = $jsonText }
                }
            }
        }
        else {
            Remove-Job $job -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Host "  LLM call error: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    return @{ Success = $false; Data = $null }
}

function Get-LLMScanDecisions {
    <#
    .SYNOPSIS
        Uses Claude to analyze problem and determine which scans to run
    .PARAMETER UserInput
        The user's problem description
    .PARAMETER ShowDetail
        Show detailed output
    .RETURNS
        Hashtable with scans (array of {id, priority, reason}) and framing (string)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$UserInput,
        [switch]$ShowDetail
    )

    # Load scan registry
    $registry = Get-ScanRegistry
    if (-not $registry) {
        return Get-KeywordScanDecisions -UserInput $UserInput
    }

    # Build scan descriptions for prompt
    $scanDescriptions = @()
    foreach ($scanProp in $registry.scans.PSObject.Properties) {
        $scan = $scanProp.Value
        $relevantFor = if ($scan.relevantFor -is [array]) { $scan.relevantFor -join ", " } else { $scan.relevantFor }
        $scanDescriptions += "- $($scan.id): $($scan.description) (relevant for: $relevantFor)"
    }

    $prompt = @"
Analyze this user's problem description and determine which codebase scans would be helpful.

USER'S PROBLEM:
$UserInput

AVAILABLE SCANS:
$($scanDescriptions -join "`n")

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "scans": [
    {
      "id": "scan_id",
      "priority": 1,
      "reason": "brief reason why this scan is relevant"
    }
  ],
  "framing": "how to frame findings for this specific problem (1 sentence)"
}

RULES:
- Select 1-5 scans, ordered by relevance (priority 1 = most relevant)
- Only include scans that directly help with the stated problem
- If the problem is vague, include exploratory scans (large_files, test_coverage)
- The "framing" field should guide how to present findings to the user
"@

    $result = Invoke-LLMWithTimeout -Prompt $prompt -TimeoutSeconds 30

    if ($result.Success -and $result.Data -and $result.Data.scans) {
        if ($ShowDetail) {
            Write-Host "  LLM decided on $($result.Data.scans.Count) scans" -ForegroundColor Green
        }
        return $result.Data
    }

    # Fallback to keyword-based decisions
    if ($ShowDetail) {
        Write-Host "  LLM analysis failed, using keyword fallback..." -ForegroundColor Yellow
    }
    return Get-KeywordScanDecisions -UserInput $UserInput
}

function Get-KeywordScanDecisions {
    <#
    .SYNOPSIS
        Fallback: determine scans based on keyword matching
    .PARAMETER UserInput
        The user's problem description
    .RETURNS
        Hashtable with scans (array) and framing (string)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$UserInput
    )

    $input = $UserInput.ToLower()
    $scans = @()
    $priority = 1

    # Keyword-to-scan mapping (preserves current behavior as fallback)
    $keywordMap = @{
        "god.?file|refactor|large.?file|split|extract|modular|unwieldy" = "large_files"
        "rate.?limit|throttl|429|quota|backoff|too.?many" = "rate_limit_patterns"
        "caption|subtitle|transcript|srt|whisper" = "caption_infrastructure"
        "download|yt-?dlp|youtube|403|forbidden|cookie|fail" = "download_system"
        "test|coverage|pytest|unit|integration" = "test_coverage"
        "config|setting|yaml|option" = "config_structure"
        "pipeline|stage|flow|order|bottleneck" = "pipeline_stages"
        "error|crash|exception|bug|broken|debug" = "error_patterns"
        "match|scor|confiden|chapter|listicle|context|title|description" = "matching_infrastructure"
    }

    foreach ($pattern in $keywordMap.Keys) {
        if ($input -match $pattern) {
            $scans += @{
                id = $keywordMap[$pattern]
                priority = $priority++
                reason = "Keyword match: $pattern"
            }
        }
    }

    # Default scans if nothing matched
    if ($scans.Count -eq 0) {
        $scans = @(
            @{ id = "large_files"; priority = 1; reason = "Default exploratory scan" }
            @{ id = "test_coverage"; priority = 2; reason = "Default quality scan" }
        )
    }

    return @{
        scans = $scans
        framing = "General codebase improvement"
    }
}

function Get-AreasForScan {
    <#
    .SYNOPSIS
        Map a scan ID to focus areas
    .PARAMETER ScanId
        The scan ID to look up
    .RETURNS
        Array of focus area IDs
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$ScanId
    )

    $registry = Get-ScanRegistry
    if ($registry -and $registry.scans.$ScanId) {
        $focusAreas = $registry.scans.$ScanId.focusAreas
        if ($focusAreas) {
            return @($focusAreas)
        }
    }

    # Fallback mapping
    $fallbackMap = @{
        "large_files" = @("refactoring", "architecture")
        "rate_limit_patterns" = @("rate-limiting", "download")
        "caption_infrastructure" = @("caption", "transcription")
        "download_system" = @("download", "rate-limiting")
        "test_coverage" = @("testing", "quality")
        "config_structure" = @("config")
        "pipeline_stages" = @("pipeline", "architecture")
        "error_patterns" = @("healing", "quality", "agents")
        "matching_infrastructure" = @("context-matching", "listicle-matching", "quality")
    }

    if ($fallbackMap.ContainsKey($ScanId)) {
        return $fallbackMap[$ScanId]
    }

    return @()
}

# ============================================================================
# LLM-FRAMED OUTPUT GENERATION
# ============================================================================

function Format-ScanResultsWithLLM {
    <#
    .SYNOPSIS
        Use LLM to frame scan findings for user's specific problem
    .PARAMETER UserInput
        The user's original problem description
    .PARAMETER ScanResults
        Hashtable of scan results keyed by scan ID
    .PARAMETER Framing
        Framing guidance from scan decision
    .RETURNS
        Formatted string with improved specifications
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$UserInput,
        [Parameter(Mandatory=$true)]
        [hashtable]$ScanResults,
        [Parameter(Mandatory=$true)]
        [string]$Framing
    )

    # Build structured findings
    $findings = @()
    foreach ($scanId in $ScanResults.Keys) {
        $scan = $ScanResults[$scanId]
        if ($scan.found) {
            $dataJson = $scan.data | ConvertTo-Json -Compress -Depth 3
            $findings += "[$scanId] $dataJson"
        }
    }

    if ($findings.Count -eq 0) {
        return "Improved Specifications:`n- No specific findings from codebase analysis`n- Proceed with general approach"
    }

    $prompt = @"
Generate an "Improved Specifications" section for a sprint based on these findings.

USER'S PROBLEM:
$UserInput

FRAMING GUIDANCE:
$Framing

SCAN FINDINGS:
$($findings -join "`n")

Generate a concise, actionable specification section. Format:

Improved Specifications:
[Section Name]:
- Bullet point 1
- Bullet point 2
- Target files: [if applicable]
- Existing patterns to integrate: [if applicable]

[Another Section if needed]:
...

Codebase Analysis:
- Summary of findings

RULES:
- Only include sections relevant to the user's problem
- Be specific with file paths and line counts
- Frame findings in terms of the user's stated problem
- Keep it under 500 words
"@

    $result = Invoke-LLMWithTimeout -Prompt $prompt -TimeoutSeconds 45

    if ($result.Success -and $result.Data) {
        # Return raw text if it's a string, otherwise format
        $output = if ($result.Data -is [string]) {
            $result.Data
        } else {
            $result.Data | Out-String
        }
        return $output.Trim()
    }

    # Fallback to template-based formatting
    return Format-ScanResultsTemplate -ScanResults $ScanResults
}

function Format-ScanResultsTemplate {
    <#
    .SYNOPSIS
        Fallback: format scan results using templates
    .PARAMETER ScanResults
        Hashtable of scan results keyed by scan ID
    .RETURNS
        Formatted string with improved specifications
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$ScanResults
    )

    $sections = @()

    # Large Files Section
    if ($ScanResults.ContainsKey("large_files") -and $ScanResults["large_files"].found) {
        $files = $ScanResults["large_files"].data.largeFiles
        if ($files -and $files.Count -gt 0) {
            $fileList = ($files | ForEach-Object { "  * $($_.path) ($($_.lines) lines)" }) -join "`n"
            $sections += @"
God File Refactoring:
- Split large files in src/ that exceed 400 lines
- Target files:
$fileList
- Extract shared utilities into focused modules
- Apply Single Responsibility Principle
"@
        }
    }

    # Rate Limiting Section
    if ($ScanResults.ContainsKey("rate_limit_patterns") -and $ScanResults["rate_limit_patterns"].found) {
        $patterns = $ScanResults["rate_limit_patterns"].data.existingPatterns
        $patternList = if ($patterns) { $patterns -join ", " } else { "none found" }
        $sections += @"
Rate Limiting Infrastructure:
- Implement bulletproof rate limiting for all yt-dlp/YouTube operations
- Add exponential backoff with jitter for 429/quota errors
- Existing patterns to integrate: $patternList
- Do NOT reduce download quantities - implement smarter throttling instead
"@
    }

    # Caption Section
    if ($ScanResults.ContainsKey("caption_infrastructure") -and $ScanResults["caption_infrastructure"].found) {
        $files = $ScanResults["caption_infrastructure"].data.captionFiles
        $fileList = if ($files) { $files -join ", " } else { "none found" }
        $sections += @"
Caption System:
- Relevant files: $fileList
- Review error handling and retry logic
- Ensure proper caching for performance
"@
    }

    # Download Section
    if ($ScanResults.ContainsKey("download_system") -and $ScanResults["download_system"].found) {
        $files = $ScanResults["download_system"].data.downloadFiles
        $fileList = if ($files) { $files -join ", " } else { "none found" }
        $sections += @"
Download Infrastructure:
- Review/improve download reliability in src/downloader/
- Handle 403/forbidden errors with escalation strategies
- Relevant files: $fileList
- Integrate with existing bypass/escalation tier system
"@
    }

    # Test Coverage Section
    if ($ScanResults.ContainsKey("test_coverage") -and $ScanResults["test_coverage"].found) {
        $data = $ScanResults["test_coverage"].data
        $untested = if ($data.untestedModules) { $data.untestedModules -join ", " } else { "none identified" }
        $sections += @"
Testing Requirements:
- Current test count: ~$($data.testCount) test files
- Potentially untested modules: $untested
- Focus on edge cases, error handling, and integration tests
- Use pytest with markers (fast, integration, requires_api)
"@
    }

    # Config Section
    if ($ScanResults.ContainsKey("config_structure") -and $ScanResults["config_structure"].found) {
        $data = $ScanResults["config_structure"].data
        $sectionList = if ($data.configSections) { $data.configSections -join ", " } else { "none found" }
        $sections += @"
Configuration System:
- Existing config sections: $sectionList
- Ensure new features have corresponding config options
- Follow pattern in src/config/sections/
"@
    }

    # Pipeline Section
    if ($ScanResults.ContainsKey("pipeline_stages") -and $ScanResults["pipeline_stages"].found) {
        $stages = $ScanResults["pipeline_stages"].data.stages
        $stageList = if ($stages) { $stages -join ", " } else { "none found" }
        $sections += @"
Pipeline Stages:
- Existing stages: $stageList
- Review stage dependencies and execution order
- Identify bottlenecks and optimization opportunities
"@
    }

    # Error Patterns Section
    if ($ScanResults.ContainsKey("error_patterns") -and $ScanResults["error_patterns"].found) {
        $data = $ScanResults["error_patterns"].data
        $healingFiles = if ($data.healingFiles) { $data.healingFiles -join ", " } else { "none found" }
        $sections += @"
Error Handling:
- Healing infrastructure files: $healingFiles
- Review recent error patterns in logs
- Improve resilience and recovery mechanisms
"@
    }

    # Matching Infrastructure Section
    if ($ScanResults.ContainsKey("matching_infrastructure") -and $ScanResults["matching_infrastructure"].found) {
        $data = $ScanResults["matching_infrastructure"].data
        $matchFiles = if ($data.matchingFiles) { $data.matchingFiles -join ", " } else { "none found" }
        $chapterFiles = if ($data.chapterFiles -and $data.chapterFiles.Count -gt 0) { $data.chapterFiles -join ", " } else { "none yet" }
        $contextUsage = if ($data.contextUsage -and $data.contextUsage.Count -gt 0) { $data.contextUsage -join ", " } else { "none detected" }
        $sections += @"
Matching & Scoring Infrastructure:
- Matching files: $matchFiles
- Chapter detection modules: $chapterFiles
- Current context usage: $contextUsage
- Review scoring adjustments and confidence breakdown
- Identify integration points for video metadata context
"@
    }

    # Build final output
    if ($sections.Count -eq 0) {
        return "Improved Specifications:`n- No specific improvements identified from scans"
    }

    $analysis = "Codebase Analysis:`n"
    $analysis += "- Scans completed: $($ScanResults.Count)`n"
    foreach ($scanId in $ScanResults.Keys) {
        if ($ScanResults[$scanId].found) {
            $analysis += "- $scanId : findings available`n"
        }
    }

    return "Improved Specifications:`n$($sections -join "`n`n")`n`n$analysis"
}
