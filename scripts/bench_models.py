#!/usr/bin/env python3
"""Benchmark multiple Ollama models on the entity extraction task.

For each model:
- Build the same prompt (uses the current ENTITY_EXTRACT_SYSTEM_PROMPT in lower_thirds.py)
- Capture the raw response
- Score: hallucination count, type accuracy, role-as-type confusion, JSON validity
- Print a comparison table

Run:
    python scripts/bench_models.py [SRT_PATH]

If SRT_PATH is omitted, uses the welsh-farmer-birds project.
"""
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.llm_client.providers.ollama import OllamaClient, check_ollama_model_available
from src.llm_client.base import LLMRequest, ResponseFormat
from src.utils import parse_srt_file, SRTSegment
from lower_thirds import (
    ENTITY_EXTRACT_SYSTEM_PROMPT,
    ENTITY_EXTRACT_USER_TEMPLATE,
    _normalize_entity_type,
    EntityType,
    build_ollama_prompt,
)

# Models to test — local ones from `ollama list`; cloud ones skipped (need network creds)
CANDIDATE_MODELS = [
    "llama3.2",            # baseline (current)
    "gemma3:4b",
    "mistral:7b",
    "granite3.2:8b",
    "dolphin3:8b",
    "functiongemma",
    "qwen3-coder:30b",
    "llama2",
]

DEFAULT_SRT = r"E:\Edit Job\Samples Projects\welsh-farmer-birds__2026-08-01\voiceover_trimmed.srt"


def build_prompt(segments: list[SRTSegment]) -> str:
    """Use the EXACT prompt the production lower_thirds.py uses."""
    sys_prompt, user_prompt = build_ollama_prompt(segments)
    return sys_prompt + "\n\n" + user_prompt


def extract_json_array(raw: str) -> list[dict] | None:
    """Best-effort JSON array extraction (no scoring)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Use brace/bracket counting to find the array (avoids greedy regex matching non-JSON)
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return None
    return None


def score_response(
    raw: str,
    segments: list[SRTSegment],
) -> dict:
    """Score one raw model response."""
    if not raw or not raw.strip():
        return {"raw_chars": 0, "valid_json": False, "entities": 0}

    entities = extract_json_array(raw) or []
    valid_json = entities is not None and len(entities) > 0
    n_total = len(entities) if entities else 0

    # Classify each entity
    type_correct = 0
    type_in_role = 0
    type_unknown = 0
    hallucinated = 0
    in_srt = 0
    srt_idx_provided = 0
    srt_idx_correct = 0

    srt_texts = [s.text.lower() for s in segments]

    for item in entities:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue

        # Where did the model put the type?
        raw_type = item.get("entity_type", item.get("type"))
        norm = _normalize_entity_type(raw_type)
        role = str(item.get("role", "")).strip()
        role_norm = _normalize_entity_type(role)

        if norm is not None:
            type_correct += 1
        elif raw_type is not None and norm is None:
            type_unknown += 1
        elif role_norm is not None:
            # Safety net applied
            type_in_role += 1
        # else: missing type (defaults to PERSON in our parser)

        # Hallucination check: name must appear in SRT (word boundary, case-insensitive)
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        hit = any(re.search(pattern, t) for t in srt_texts)
        if hit:
            in_srt += 1
        else:
            hallucinated += 1

        # Did the model provide srt_indices? Were they correct?
        idx = item.get("srt_indices")
        if idx:
            srt_idx_provided += 1
            if isinstance(idx, list):
                # Check first idx points to a segment containing the name
                first = idx[0] if idx else None
                if first is not None:
                    seg = next((s for s in segments if s.index == first), None)
                    if seg and re.search(pattern, seg.text.lower()):
                        srt_idx_correct += 1

    return {
        "raw_chars": len(raw),
        "valid_json": valid_json,
        "entities": n_total,
        "type_correct": type_correct,
        "type_in_role": type_in_role,
        "type_unknown": type_unknown,
        "in_srt": in_srt,
        "hallucinated": hallucinated,
        "srt_idx_provided": srt_idx_provided,
        "srt_idx_correct": srt_idx_correct,
    }


def run_model(
    model: str,
    prompt: str,
    segments: list[SRTSegment],
    host: str = "http://localhost:11434",
) -> dict:
    """Run one model and score it. Returns dict with score + latency + raw text."""
    available, err = check_ollama_model_available(model, host)
    if not available:
        return {"model": model, "error": f"unavailable: {err}"}

    client = OllamaClient(model=model, host=host)
    req = LLMRequest(
        prompt=prompt,
        max_tokens=2048,
        temperature=0.1,
        response_format=ResponseFormat.TEXT,
        timeout=300,
        cache_key_prefix=f"bench_{model.replace(':', '_').replace('.', '_')}",
    )
    t0 = time.time()
    try:
        resp = client.generate(req)
        latency = time.time() - t0
    except Exception as e:
        return {"model": model, "error": f"call failed: {e}"}
    score = score_response(resp.text, segments)
    return {
        "model": model,
        "latency_s": round(latency, 1),
        **score,
    }


def main():
    srt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_SRT)
    if not srt_path.exists():
        print(f"SRT not found: {srt_path}")
        return 1
    segments = parse_srt_file(str(srt_path))
    print(f"SRT: {srt_path} ({len(segments)} segments, {segments[-1].end_time:.1f}s)")
    prompt = build_prompt(segments)
    print(f"Prompt length: {len(prompt)} chars\n")

    print(f"Testing {len(CANDIDATE_MODELS)} models...\n")
    results = []
    for m in CANDIDATE_MODELS:
        print(f"  > {m} ...", end="", flush=True)
        r = run_model(m, prompt, segments)
        if "error" in r:
            print(f" SKIP ({r['error']})")
        else:
            print(f" {r['latency_s']}s, {r['entities']} entities, "
                  f"{r['type_correct']} typed, {r['in_srt']} in-SRT, "
                  f"{r['hallucinated']} hallucinated")
        results.append(r)

    # Sort: highest type_correct first, then most in-SRT
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: (-r["type_correct"], -r["in_srt"], r["hallucinated"]))

    print("\n" + "=" * 100)
    print(f"{'Model':<22} {'Lat':>5} {'#':>4} {'Typed':>6} {'InRole':>7} {'Unknown':>8} {'InSRT':>6} {'Hallu':>5} {'JSON':>5}  SRT-idx")
    print("=" * 100)
    print("  # = entities emitted | Typed = correct entity_type | InRole = type was in role field")
    print("  Unknown = unparseable type | InSRT = name found in SRT | Hallu = likely hallucination")
    print("  SRT-idx = (correct srt_indices)/(provided srt_indices) when the model emits any")
    print("=" * 100)
    for r in valid:
        print(f"{r['model']:<22} {r['latency_s']:>4}s {r['entities']:>4} {r['type_correct']:>6} "
              f"{r['type_in_role']:>7} {r['type_unknown']:>8} {r['in_srt']:>6} {r['hallucinated']:>5} "
              f"{'Y' if r['valid_json'] else 'N':>5}  {r['srt_idx_correct']}/{r['srt_idx_provided']}")
    for r in results:
        if "error" in r:
            print(f"{r['model']:<22} -- {r['error']}")


if __name__ == "__main__":
    main()
