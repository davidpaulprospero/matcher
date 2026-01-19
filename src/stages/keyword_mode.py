"""
Keyword Mode Stages - Pipeline without voiceover input.

Provides stages for running the pipeline with just keywords:
- KeywordInputStage: Sets up keywords and validates config
- MontageSegmentStage: Creates equal-duration segments (montage mode)
- ScriptSynthesisStage: LLM generates narration script (script mode)
- CollectionStage: Organizes downloads by keyword (collection mode)

Created: 2026-01-19
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


# =============================================================================
# KEYWORD INPUT STAGE
# =============================================================================

@register_stage
class KeywordInputStage(Stage):
    """
    Initializes keyword mode by setting up keywords from CLI or config.

    This stage runs first in keyword mode and:
    - Validates that keyword mode is enabled
    - Gets keywords from CLI args or config
    - Validates we have at least one keyword
    - Sets state.keywords for downstream stages

    Inputs:
        - config.keyword_mode.keywords (or CLI --keyword-list)

    Outputs:
        - state.keywords: List of input keywords
        - state.metadata['keyword_mode']: Mode name
    """

    name = "KEYWORD_INPUT"
    description = "Set up keywords for keyword mode pipeline"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Set up keywords from config or CLI."""
        try:
            print(f"\n  --- Stage: KEYWORD INPUT ---")

            kw_config = config.keyword_mode

            # Get keywords (CLI args take precedence, stored in state by main.py)
            keywords = state.keywords or kw_config.keywords

            if not keywords:
                return StageResult.fail(
                    "No keywords provided. Use --keyword-list or set keyword_mode.keywords in config."
                )

            # Clean and validate keywords
            keywords = [k.strip() for k in keywords if k.strip()]

            if not keywords:
                return StageResult.fail("All keywords were empty after cleaning")

            state.keywords = keywords
            state.metadata['keyword_mode'] = kw_config.mode
            state.metadata['keyword_source'] = 'keyword_mode'

            print(f"  [OK] {len(keywords)} keywords ready: {', '.join(keywords[:5])}")
            if len(keywords) > 5:
                print(f"    ... and {len(keywords) - 5} more")
            print(f"  Mode: {kw_config.mode}")

            checkpoint_data = {
                'keywords': keywords,
                'mode': kw_config.mode,
            }

            return StageResult.ok(checkpoint_data)

        except Exception as e:
            logger.exception(f"Keyword input stage failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if we can skip this stage."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore from checkpoint."""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            state.keywords = data.get('keywords', [])
            state.metadata['keyword_mode'] = data.get('mode', 'script')
            state.metadata['keyword_source'] = 'keyword_mode'

            return True
        except Exception as e:
            logger.error(f"Failed to restore keyword input: {e}")
            return False


# =============================================================================
# MONTAGE SEGMENT STAGE
# =============================================================================

@register_stage
class MontageSegmentStage(Stage):
    """
    Creates equal-duration segments from keywords (montage mode).

    Each keyword becomes a segment with configurable duration.
    The segments are converted to VoiceoverSegments for pipeline compatibility.

    Config options:
        - montage.segment_duration: Duration per keyword (default 10s)
        - montage.total_duration: If set, divides evenly among keywords
        - montage.transition_gap: Gap between segments (default 0s)

    Inputs:
        - state.keywords: List of keywords

    Outputs:
        - state.voiceover_segments: List of VoiceoverSegment (one per keyword)
        - state.metadata['segment_source']: 'montage'
    """

    name = "MONTAGE_SEGMENT"
    description = "Create equal-duration segments from keywords"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Create segments from keywords."""
        try:
            print(f"\n  --- Stage: MONTAGE SEGMENTS ---")

            from ..state import VoiceoverSegment, KeywordSegment

            keywords = state.keywords
            if not keywords:
                return StageResult.fail("No keywords available")

            # Handle dict/object config (Rule 6)
            montage_config = config.keyword_mode.montage
            if isinstance(montage_config, dict):
                segment_duration = montage_config.get('segment_duration', 10.0)
                total_duration = montage_config.get('total_duration')
                transition_gap = montage_config.get('transition_gap', 0.0)
            else:
                segment_duration = getattr(montage_config, 'segment_duration', 10.0)
                total_duration = getattr(montage_config, 'total_duration', None)
                transition_gap = getattr(montage_config, 'transition_gap', 0.0)

            # Calculate duration per segment
            if total_duration:
                # Divide total duration among keywords
                num_keywords = len(keywords)
                total_gap_time = transition_gap * (num_keywords - 1) if num_keywords > 1 else 0
                available_time = total_duration - total_gap_time
                segment_duration = max(1.0, available_time / num_keywords)
                print(f"  Total duration: {total_duration}s / {num_keywords} keywords = {segment_duration:.1f}s each")
            else:
                print(f"  Segment duration: {segment_duration}s per keyword")

            # Create segments
            keyword_segments: List[KeywordSegment] = []
            voiceover_segments: List[VoiceoverSegment] = []
            current_time = 0.0

            for i, keyword in enumerate(keywords):
                # Create KeywordSegment
                kw_seg = KeywordSegment(
                    index=i,
                    keyword=keyword,
                    start_time=current_time,
                    end_time=current_time + segment_duration,
                    description=keyword,  # Use keyword as description
                    source="keyword_montage",
                )
                keyword_segments.append(kw_seg)

                # Convert to VoiceoverSegment
                vo_seg = kw_seg.to_voiceover_segment()
                voiceover_segments.append(vo_seg)

                # Advance time
                current_time += segment_duration + transition_gap

            state.voiceover_segments = voiceover_segments
            state.metadata['segment_source'] = 'montage'
            state.metadata['keyword_segments'] = [
                {
                    'index': ks.index,
                    'keyword': ks.keyword,
                    'start': ks.start_time,
                    'end': ks.end_time,
                }
                for ks in keyword_segments
            ]

            total_time = voiceover_segments[-1].end if voiceover_segments else 0
            print(f"  [OK] Created {len(voiceover_segments)} segments")
            print(f"  Total timeline: {total_time:.1f}s")

            checkpoint_data = {
                'segments': [
                    {
                        'index': s.index,
                        'start': s.start,
                        'end': s.end,
                        'text': s.text,
                        'duration': s.duration,
                    }
                    for s in voiceover_segments
                ],
                'keyword_segments': state.metadata['keyword_segments'],
                'total_duration': total_time,
            }

            return StageResult.ok(checkpoint_data)

        except Exception as e:
            logger.exception(f"Montage segment stage failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if we can skip this stage."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore segments from checkpoint."""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            from ..state import VoiceoverSegment

            segments = []
            for seg_dict in data.get('segments', []):
                segments.append(VoiceoverSegment(
                    index=seg_dict.get('index', 0),
                    start=seg_dict.get('start', 0),
                    end=seg_dict.get('end', 0),
                    text=seg_dict.get('text', ''),
                    duration=seg_dict.get('duration', 0),
                ))

            state.voiceover_segments = segments
            state.metadata['segment_source'] = 'montage'
            state.metadata['keyword_segments'] = data.get('keyword_segments', [])

            return True
        except Exception as e:
            logger.error(f"Failed to restore montage segments: {e}")
            return False


# =============================================================================
# SCRIPT SYNTHESIS STAGE
# =============================================================================

# Script style prompts
STYLE_PROMPTS = {
    'documentary': """Write authoritative, informative narration. Use measured pacing with
facts and context. Build understanding progressively.""",

    'promotional': """Write punchy, benefit-focused copy. Highlight value propositions.
Include a clear call-to-action. Keep energy high.""",

    'narrative': """Write with a story arc - setup, development, resolution. Use emotional
beats and personal connection. Create journey.""",

    'listicle': """Write with clear numbered structure. Each point gets one paragraph.
Be concise and scannable. Front-load key info.""",

    'poetic': """Write evocative, metaphorical prose. Use sensory language and imagery.
Let meaning emerge through atmosphere.""",

    'minimal': """Write very short phrases. One thought per line. Let visuals carry meaning.
Spare and impactful.""",
}

SCRIPT_GENERATION_PROMPT = """You are a documentary scriptwriter. Write narration for a {style} video.

Topic keywords: {keywords}
Target duration: {duration} seconds (approximately {word_count} words at {wpm} WPM)
Tone: {tone}

Requirements:
1. Write in a natural speaking voice suitable for narration
2. Use paragraph breaks to indicate natural pauses/scene changes
3. Each paragraph should be 1-3 sentences (one visual idea)
4. Start with a hook, end with impact
5. Weave ALL keywords naturally into the script
6. Don't use headers, bullets, or formatting - just flowing paragraphs

Style notes for {style}:
{style_notes}

Write the script only, no commentary.
"""


@register_stage
class ScriptSynthesisStage(Stage):
    """
    Generates narration script from keywords using LLM.

    Pipeline: Keywords -> LLM Script -> Timed SRT -> Normal matching
    The SRT is the artifact - user can optionally do TTS later.

    Config options:
        - script.style: documentary | promotional | narrative | listicle | poetic | minimal
        - script.tone: inspiring | serious | playful | urgent | contemplative
        - script.target_duration: Target video duration in seconds
        - script.words_per_minute: Speaking rate for timing
        - script.llm: LLM provider settings (Ollama by default)

    Inputs:
        - state.keywords: List of keywords

    Outputs:
        - state.voiceover_segments: Timed segments from generated SRT
        - state.voiceover_path: Path to generated SRT file
        - state.metadata['script_source']: 'synthetic'
        - Saves: voiceover/synthetic.srt, voiceover/synthetic_script.txt
    """

    name = "SCRIPT_SYNTHESIS"
    description = "Generate narration script from keywords using LLM"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Generate script and convert to SRT."""
        warnings = []

        try:
            print(f"\n  --- Stage: SCRIPT SYNTHESIS ---")

            keywords = state.keywords
            if not keywords:
                return StageResult.fail("No keywords available for script generation")

            script_config = config.keyword_mode.script

            # Calculate target word count
            target_words = int(script_config.target_duration * script_config.words_per_minute / 60)
            print(f"  Target: {script_config.target_duration}s @ {script_config.words_per_minute} WPM = ~{target_words} words")
            print(f"  Style: {script_config.style}, Tone: {script_config.tone}")
            print(f"  Keywords: {', '.join(keywords[:5])}")
            if len(keywords) > 5:
                print(f"    ... and {len(keywords) - 5} more")

            # Generate script with LLM
            print(f"\n  Generating script with {script_config.llm.provider}...")
            script = self._generate_script(keywords, script_config)

            if not script:
                return StageResult.fail("LLM failed to generate script")

            actual_words = len(script.split())
            print(f"  [OK] Generated {actual_words} words")

            # Validate keywords are present
            if script_config.require_all_keywords:
                missing = self._check_missing_keywords(script, keywords)
                if missing:
                    warnings.append(f"Keywords not in script: {', '.join(missing)}")
                    print(f"  [!] Missing keywords: {', '.join(missing)}")

            # Convert to timed SRT
            print(f"\n  Converting to timed SRT...")
            srt_content = self._script_to_srt(
                script,
                script_config.target_duration,
                script_config.words_per_minute,
                script_config.min_segment_duration,
                script_config.segment_gap,
                script_config.split_mode,
            )

            # Save outputs
            project_dir = Path(state.project_dir)
            voiceover_dir = project_dir / "voiceover"
            voiceover_dir.mkdir(parents=True, exist_ok=True)

            srt_path = voiceover_dir / "synthetic.srt"
            script_path = voiceover_dir / "synthetic_script.txt"

            if script_config.save_srt:
                srt_path.write_text(srt_content, encoding='utf-8')
                print(f"  [OK] Saved: {srt_path}")

            if script_config.save_script_txt:
                script_path.write_text(script, encoding='utf-8')
                print(f"  [OK] Saved: {script_path}")

            # Parse SRT into segments
            segments = self._parse_srt(srt_content)
            state.voiceover_segments = segments
            state.voiceover_path = str(srt_path)
            state.metadata['script_source'] = 'synthetic'
            state.metadata['script_style'] = script_config.style
            state.metadata['script_tone'] = script_config.tone

            print(f"  [OK] Created {len(segments)} segments from SRT")

            checkpoint_data = {
                'script': script,
                'srt_content': srt_content,
                'srt_path': str(srt_path),
                'segments': [
                    {
                        'index': s.index,
                        'start': s.start,
                        'end': s.end,
                        'text': s.text,
                        'duration': s.duration,
                    }
                    for s in segments
                ],
                'word_count': actual_words,
                'style': script_config.style,
                'tone': script_config.tone,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Script synthesis stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _generate_script(self, keywords: List[str], script_config) -> str:
        """Generate script using LLM."""
        llm_config = script_config.llm

        # Build prompt
        style_notes = STYLE_PROMPTS.get(script_config.style, STYLE_PROMPTS['documentary'])
        target_words = int(script_config.target_duration * script_config.words_per_minute / 60)

        prompt = SCRIPT_GENERATION_PROMPT.format(
            style=script_config.style,
            keywords=", ".join(keywords),
            duration=script_config.target_duration,
            word_count=target_words,
            wpm=script_config.words_per_minute,
            tone=script_config.tone,
            style_notes=style_notes,
        )

        # Try primary model first, then fallbacks
        models_to_try = [llm_config.model] + llm_config.fallback_models

        for model in models_to_try:
            try:
                if llm_config.provider == "ollama":
                    script = self._call_ollama(prompt, model, llm_config)
                elif llm_config.provider == "gemini":
                    script = self._call_gemini(prompt, model, llm_config)
                elif llm_config.provider == "anthropic":
                    script = self._call_anthropic(prompt, model, llm_config)
                else:
                    logger.warning(f"Unknown LLM provider: {llm_config.provider}")
                    continue

                if script and len(script.strip()) > 50:
                    return script.strip()

            except Exception as e:
                logger.warning(f"LLM call failed with {model}: {e}")
                continue

        return ""

    def _call_ollama(self, prompt: str, model: str, llm_config) -> str:
        """Call Ollama API for script generation."""
        import requests

        url = f"{llm_config.ollama_host}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": llm_config.temperature,
                "num_predict": llm_config.max_tokens,
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=llm_config.timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {llm_config.ollama_host}")
            logger.error("Run: ollama serve (or: OLLAMA_MODELS=D:\\ollama\\models ollama serve)")
            raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def _call_gemini(self, prompt: str, model: str, llm_config) -> str:
        """Call Gemini API for script generation."""
        import os
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        # Use the LLM client abstraction
        from ..llm_client import create_client, LLMRequest

        client = create_client("gemini", api_key=api_key, model=model)
        response = client.generate(LLMRequest(
            prompt=prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        ))
        return response.text

    def _call_anthropic(self, prompt: str, model: str, llm_config) -> str:
        """Call Anthropic API for script generation."""
        import os
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        from ..llm_client import create_client, LLMRequest

        client = create_client("anthropic", api_key=api_key, model=model)
        response = client.generate(LLMRequest(
            prompt=prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        ))
        return response.text

    def _check_missing_keywords(self, script: str, keywords: List[str]) -> List[str]:
        """Check which keywords are missing from the script."""
        script_lower = script.lower()
        missing = []
        for kw in keywords:
            # Check for whole word match (not just substring)
            if kw.lower() not in script_lower:
                missing.append(kw)
        return missing

    def _script_to_srt(
        self,
        script: str,
        target_duration: float,
        wpm: int,
        min_segment_duration: float,
        segment_gap: float,
        split_mode: str,
    ) -> str:
        """Convert script text to timed SRT format."""
        # Split into segments based on mode
        if split_mode == "sentence":
            segments = self._split_by_sentences(script)
        elif split_mode == "hybrid":
            segments = self._split_hybrid(script)
        else:  # paragraph
            segments = self._split_by_paragraphs(script)

        if not segments:
            return ""

        # Calculate timing
        total_words = sum(len(s.split()) for s in segments)
        words_per_second = wpm / 60

        natural_duration = total_words / words_per_second if words_per_second > 0 else target_duration
        time_scale = target_duration / natural_duration if natural_duration > 0 else 1.0

        srt_lines = []
        current_time = 0.0

        for i, text in enumerate(segments):
            word_count = len(text.split())
            segment_duration = max(
                (word_count / words_per_second) * time_scale,
                min_segment_duration
            )

            if i > 0:
                current_time += segment_gap

            start_time = current_time
            end_time = current_time + segment_duration

            # Format SRT entry
            srt_lines.append(str(i + 1))
            srt_lines.append(f"{self._format_srt_time(start_time)} --> {self._format_srt_time(end_time)}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between entries

            current_time = end_time

        return "\n".join(srt_lines)

    def _split_by_paragraphs(self, script: str) -> List[str]:
        """Split script by paragraph breaks."""
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        return paragraphs

    def _split_by_sentences(self, script: str) -> List[str]:
        """Split script by sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', script)
        return [s.strip() for s in sentences if s.strip()]

    def _split_hybrid(self, script: str) -> List[str]:
        """Split by paragraph, then further split long paragraphs by sentence."""
        paragraphs = self._split_by_paragraphs(script)
        segments = []

        for para in paragraphs:
            word_count = len(para.split())
            if word_count > 40:  # Long paragraph
                sentences = self._split_by_sentences(para)
                segments.extend(sentences)
            else:
                segments.append(para)

        return segments

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _parse_srt(self, srt_content: str) -> List:
        """Parse SRT content into VoiceoverSegment list."""
        from ..state import VoiceoverSegment

        segments = []
        entries = srt_content.strip().split('\n\n')

        for entry in entries:
            lines = entry.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0])

                # Parse timing
                timing_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1]
                )
                if not timing_match:
                    continue

                start = (
                    int(timing_match.group(1)) * 3600 +
                    int(timing_match.group(2)) * 60 +
                    int(timing_match.group(3)) +
                    int(timing_match.group(4)) / 1000
                )
                end = (
                    int(timing_match.group(5)) * 3600 +
                    int(timing_match.group(6)) * 60 +
                    int(timing_match.group(7)) +
                    int(timing_match.group(8)) / 1000
                )

                text = ' '.join(lines[2:])

                segments.append(VoiceoverSegment(
                    index=index,
                    start=start,
                    end=end,
                    text=text,
                ))

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse SRT entry: {e}")
                continue

        return segments

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if we can skip this stage."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore from checkpoint."""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            from ..state import VoiceoverSegment

            segments = []
            for seg_dict in data.get('segments', []):
                segments.append(VoiceoverSegment(
                    index=seg_dict.get('index', 0),
                    start=seg_dict.get('start', 0),
                    end=seg_dict.get('end', 0),
                    text=seg_dict.get('text', ''),
                    duration=seg_dict.get('duration', 0),
                ))

            state.voiceover_segments = segments
            state.voiceover_path = data.get('srt_path', '')
            state.metadata['script_source'] = 'synthetic'
            state.metadata['script_style'] = data.get('style', 'documentary')
            state.metadata['script_tone'] = data.get('tone', 'inspiring')

            return True
        except Exception as e:
            logger.error(f"Failed to restore script synthesis: {e}")
            return False


# =============================================================================
# OLLAMA UTILITIES
# =============================================================================

PREFERRED_MODELS = [
    "mistral:7b",
    "mistral:latest",
    "qwen2.5:7b",
    "qwen2.5:latest",
    "llama3.2:3b",
    "llama3.2:latest",
    "phi4:14b",
]


def list_ollama_models(host: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    import requests

    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        logger.warning(f"Failed to list Ollama models: {e}")
        return []


def get_best_ollama_model(host: str = "http://localhost:11434") -> Optional[str]:
    """Find the best available Ollama model."""
    available = list_ollama_models(host)

    if not available:
        return None

    for model in PREFERRED_MODELS:
        if model in available:
            return model

    # Fallback to first available
    return available[0] if available else None
