"""
Step 4 — LLM Scoring Module
Sends degraded texts to LLMs for quality rating (0-10 scale).

Standard API for both Google and OpenAI, with:
  - Exponential backoff on rate-limit / transient errors
  - JSONL checkpoint every 50 samples — resume from where you left off
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = "Rate the quality of the following text from 0 to 10. Respond with ONLY the number."

CHECKPOINT_EVERY = 50
MAX_RETRIES = 5


# ── Response Parsing ─────────────────────────────────────────────

def parse_score(raw: str) -> int | None:
    """Extract score 0-10 from raw LLM response."""
    try:
        match = re.search(r'\b(10|\d)\b', raw.strip())
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score
    except (ValueError, AttributeError):
        pass
    return None


# ── Checkpoint Helpers ───────────────────────────────────────────

def _checkpoint_path(output_dir: Path, model_name: str) -> Path:
    safe = model_name.replace(" ", "_").replace("/", "_")
    return output_dir / f"checkpoint_{safe}.jsonl"


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set]:
    results = []
    done_ids = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    results.append(entry)
                    done_ids.add(entry["sample_id"])
    return results, done_ids


def _flush_checkpoint(ckpt_path: Path, buffer: list[dict]):
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for entry in buffer:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── API Call with Retry ──────────────────────────────────────────

def _call_with_retry(provider: str, model_id: str, user_prompt: str,
                     api_key: str | None = None) -> str:
    """Call LLM with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "google":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                    ),
                )
                return response.text
            elif provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=1024,
                )
                return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            transient = any(k in err for k in
                            ["429", "rate", "limit", "quota", "503",
                             "overloaded", "timeout", "connection", "unavailable"])
            if transient and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"\n  [Retry {attempt+1}/{MAX_RETRIES}] "
                      f"{str(e)[:80]}... waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("All retries failed")


# ── Score All Samples ────────────────────────────────────────────

def _score_model(samples: list[dict], model_cfg: dict,
                 output_dir: Path, api_key: str | None = None) -> list[dict]:
    """Score all samples for one model. Checkpoints every 50."""
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]
    model_name = model_cfg["name"]

    ckpt = _checkpoint_path(output_dir, model_name)
    results, done_ids = _load_checkpoint(ckpt)

    if done_ids:
        print(f"  [Resume] {len(done_ids)} already scored, continuing...")

    remaining = [s for s in samples if s["id"] not in done_ids]
    buffer = []

    for sample in tqdm(remaining, desc=f"Scoring [{model_name}]",
                       initial=len(done_ids), total=len(samples)):
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )

        try:
            raw = _call_with_retry(provider, model_id, user_prompt,
                                   api_key=api_key)
            score = parse_score(raw)
            entry = {
                "sample_id": sample["id"],
                "model": model_name,
                "condition": "isolated",
                "repetition": 0,
                "score": score,
                "raw_response": raw,
                **({"parse_error": True} if score is None else {}),
            }
        except Exception as e:
            entry = {
                "sample_id": sample["id"],
                "model": model_name,
                "condition": "isolated",
                "repetition": 0,
                "score": None,
                "raw_response": str(e),
                "parse_error": True,
            }

        results.append(entry)
        buffer.append(entry)

        if len(buffer) >= CHECKPOINT_EVERY:
            _flush_checkpoint(ckpt, buffer)
            buffer.clear()

        time.sleep(0.3)

    if buffer:
        _flush_checkpoint(ckpt, buffer)
        buffer.clear()

    return results


# ── Main Runner ─────────────────────────────────────────────────

def run(config: dict, samples: list[dict]) -> list[dict]:
    """Run LLM scoring across all models."""
    llm_cfg = config["llm_scoring"]
    output_dir = Path(llm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "llm_scores.json"
    if output_file.exists():
        print("[LLM Scoring] Loading existing LLM scores...")
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    all_results = []

    for model_cfg in llm_cfg["models"]:
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]

        print(f"[LLM Scoring] {model_name} ({provider})")
        results = _score_model(samples, model_cfg, output_dir)
        all_results.extend(results)

    # Save final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[LLM Scoring] Saved {len(all_results)} ratings to {output_file}")

    # Clean up checkpoint files
    for model_cfg in llm_cfg["models"]:
        ckpt = _checkpoint_path(output_dir, model_cfg["name"])
        if ckpt.exists():
            ckpt.unlink()
            print(f"  [Cleanup] Removed {ckpt.name}")

    return all_results
