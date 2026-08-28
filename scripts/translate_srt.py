#!/usr/bin/env python3
"""Translate a voiceover SRT to English.

Preserves segment indices and timestamps; only the text is replaced.
Intended for non-English voiceovers (Spanish, etc.) where stock-footage search
needs English text to find good matches.

Methods:
    opus-mt   Helsinki-NLP/opus-mt-es-en (MarianMT) — local, free, GPU-accelerated.
              ~300MB download first run, then cached.
    nllb-200  Meta NLLB-200 distilled (spa_Latn -> eng_Latn). Local, free.
              --model-size 600M (2.4GB), 1.3B (5GB, default), 3.3B (13GB).
    llm       Configured LLM provider (Gemini/Anthropic/Ollama). Requires API key.
              Slower per batch but best at idioms and named entities.

Usage:
    python scripts/translate_srt.py input.srt output.en.srt
    python scripts/translate_srt.py input.srt output.en.srt --method llm
    python scripts/translate_srt.py input.srt output.en.srt --method opus-mt
    python scripts/translate_srt.py input.srt output.en.srt --method nllb-200 --model-size 1.3B
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")


def parse_srt(content: str):
    """Parse SRT into list of (index, start_str, end_str, text)."""
    entries = []
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        idx = int(lines[0].strip())
        times = lines[1].split(" --> ")
        start = times[0].strip()
        end = times[1].strip()
        text = "\n".join(lines[2:]).strip()
        entries.append((idx, start, end, text))
    return entries


def write_srt(path: Path, entries):
    """Write SRT file. Each entry: (idx, start_str, end_str, text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, start, end, text in entries:
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def translate_with_opus_mt(entries, batch_size: int = 32):
    """Translate using Helsinki-NLP/opus-mt-es-en (local MarianMT)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_name = "Helsinki-NLP/opus-mt-es-en"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    translated = []
    batches = list(chunked(entries, batch_size))
    total = len(batches)
    for i, batch in enumerate(batches, 1):
        texts = [text for _, _, _, text in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for (idx, start, end, _orig), new_text in zip(batch, translated_texts):
            translated.append((idx, start, end, new_text.strip()))

        idxs = [e[0] for e in batch]
        print(f"Translated batch {i}/{total} (segments {idxs[0]}..{idxs[-1]})")

    return translated


NLLB_SOURCE_LANGUAGES = {
    "es": "spa_Latn",
    "hr": "hrv_Latn",
}


def translate_with_nllb(entries, batch_size: int = 32, model_size: str = "1.3B", source_language: str = "es"):
    """Translate to English using Meta NLLB-200 distilled."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_name = f"facebook/nllb-200-distilled-{model_size}"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    src_lang = NLLB_SOURCE_LANGUAGES[source_language]
    tgt_lang = "eng_Latn"
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translated = []
    batches = list(chunked(entries, batch_size))
    total = len(batches)
    for i, batch in enumerate(batches, 1):
        texts = [text for _, _, _, text in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=1,
                do_sample=False,
            )

        translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for (idx, start, end, _orig), new_text in zip(batch, translated_texts):
            translated.append((idx, start, end, new_text.strip()))

        idxs = [e[0] for e in batch]
        print(f"Translated batch {i}/{total} (segments {idxs[0]}..{idxs[-1]})")

    return translated


SYSTEM_PROMPT = (
    "You are translating documentary voiceover to English for stock-footage search. "
    "Rules:\n"
    "- Capture the visual meaning of each segment (what would be on screen).\n"
    "- Use natural, idiomatic English.\n"
    "- Keep length close to the source.\n"
    "- Do NOT add commentary, explanations, or extra context.\n"
    "- Preserve speaker tense, voice, and named entities (people, places).\n"
    "- Output one numbered translation per input line, in the same order, "
    "with no blank lines and no other text."
)


def build_user_prompt(entries):
    lines = [f"{idx}. {text}" for idx, _s, _e, text in entries]
    return f"Translate these {len(entries)} voiceover segments to English:\n\n" + "\n".join(lines)


def parse_numbered_response(text: str, expected_count: int):
    """Parse LLM response into expected_count English segments.

    Tolerates: preambles ("Here are the translations:"), numbering restarts
    (1..N instead of the original SRT indices), and stray prose lines.
    Strategy: find the first numbered line, take the next `expected_count`
    numbered lines in order, and map them positionally to the batch.
    Raises only if the count itself doesn't match — so a missing/extra cue
    is still detected.
    """
    numbered = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)[\.\)]\s+(.+?)\s*$", line)
        if m:
            numbered.append(m.group(2))

    if len(numbered) < expected_count:
        raise ValueError(
            f"Translation count mismatch: expected {expected_count}, got {len(numbered)}.\n"
            f"Response was:\n{text}"
        )
    return numbered[:expected_count]


def translate_with_llm(entries, batch_size: int, config_path: Path):
    """Translate using the configured LLM provider (Gemini/Anthropic/Ollama)."""
    from src.config import load_config
    from src.llm_client.factory import create_client_from_config
    from src.llm_client.base import LLMRequest

    config = load_config(str(config_path), skip_final_validation=True)
    client = create_client_from_config(config)
    print(f"Using LLM provider: {client.provider_name} model: {client.model}")

    translated = []
    batches = list(chunked(entries, batch_size))
    total = len(batches)
    for i, batch in enumerate(batches, 1):
        idxs = [e[0] for e in batch]
        print(f"Translating batch {i}/{total} (segments {idxs[0]}..{idxs[-1]})...")
        prompt = build_user_prompt(batch)
        last_err = None
        for attempt in range(3):
            resp = client.generate(LLMRequest(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=min(8000, sum(len(t) for _, _, _, t in batch) * 4 + 200),
                temperature=0.2,
            ))
            try:
                texts = parse_numbered_response(resp.text, len(batch))
                break
            except ValueError as e:
                last_err = e
                print(f"  Retry {attempt + 1}/2 (parse error)")
        else:
            raise last_err
        for (idx, start, end, _orig), new_text in zip(batch, texts):
            translated.append((idx, start, end, new_text))
    return translated


def main():
    ap = argparse.ArgumentParser(description="Translate a voiceover SRT to English.")
    ap.add_argument("input", type=Path, help="Source SRT (e.g., Spanish voiceover)")
    ap.add_argument("output", type=Path, help="Destination English SRT")
    ap.add_argument("--method", choices=["opus-mt", "nllb-200", "llm"], default="opus-mt",
                    help="Translation method (default: opus-mt)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Segments per batch (default: 32)")
    ap.add_argument("--model-size", default="600M",
                    choices=["600M", "1.3B", "3.3B"],
                    help="NLLB-200 model size (default: 600M)")
    ap.add_argument("--source-language", choices=["es", "hr"], default="es",
                    help="Source language for NLLB translation (default: es)")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml",
                    help="Path to config.yaml (only used with --method llm)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    print(f"Reading SRT: {args.input}")
    entries = parse_srt(args.input.read_text(encoding="utf-8"))
    print(f"Found {len(entries)} segments")

    if not entries:
        sys.exit("No segments parsed from input SRT.")

    print(f"Method: {args.method}")
    if args.method == "opus-mt":
        translated = translate_with_opus_mt(entries, args.batch_size)
    elif args.method == "nllb-200":
        translated = translate_with_nllb(entries, args.batch_size, args.model_size, args.source_language)
    else:
        translated = translate_with_llm(entries, args.batch_size, args.config)

    write_srt(args.output, translated)
    print(f"Wrote {len(translated)} segments to {args.output}")


if __name__ == "__main__":
    main()