"""
Step 4 — LLM Scoring Module
Sends degraded texts to LLMs for quality rating (0-10 scale).

Standard API for both Google and OpenAI, with:
  - Exponential backoff on rate-limit / transient errors
  - JSONL checkpoint every 50 samples — resume from where you left off
"""

import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = "Rate the quality of the following text from 0 to 10. Respond with ONLY the number."

_SCORE_TOKENS = frozenset({"0","1","2","3","4","5","6","7","8","9","10"})
_SINGLE_DIGIT_TOKENS = frozenset({"0","1","2","3","4","5","6","7","8","9"})


def _extract_score_probs(lp) -> dict | None:
    """
    Build a normalised {0..10: float} probability dict from Ollama logprobs.

    Handles two tokenisation styles transparently:
      - Model has a single "10" token  → read it directly from top_logprobs.
      - Model splits "10" into "1"+"0" → detect via the generated sequence and
        compute P("10") = P("1") * P("0"|"1") exactly.

    For the ambiguous "1" token in top_logprobs:
      - If the vocabulary contains a dedicated "10" token (seen anywhere in
        top_logprobs), then "1" unambiguously means score 1.
      - If there is no "10" token in the vocabulary, "1" is the start of the
        two-token sequence for score 10; P(score=1) cannot be recovered from
        the first position alone, so we assign all "1" mass to score 10.
    """
    if not (lp and lp.content):
        return None

    raw_probs: dict[int, float] = {}

    first_pos = lp.content[0]
    first_tok = first_pos.token.strip()
    first_prob = math.exp(first_pos.logprob)

    # Scan top_logprobs at position 1 to learn about the vocabulary
    top_tokens = {alt.token.strip(): math.exp(alt.logprob)
                  for alt in first_pos.top_logprobs}
    has_dedicated_ten = "10" in top_tokens

    # ── Handle the generated (chosen) token(s) ───────────────────────────────
    if first_tok == "10":
        # Model has a single "10" token and chose it
        raw_probs[10] = first_prob

    elif first_tok == "1" and len(lp.content) > 1 and lp.content[1].token.strip() == "0":
        # Two-token "10": P("10") = P("1") * P("0" | "1") — exact joint prob
        raw_probs[10] = math.exp(first_pos.logprob + lp.content[1].logprob)

    elif first_tok in _SINGLE_DIGIT_TOKENS:
        raw_probs[int(first_tok)] = first_prob

    # ── Handle alternatives from top_logprobs ────────────────────────────────
    for tok, prob in top_tokens.items():
        if tok == "10":
            # Dedicated token — exact
            raw_probs[10] = raw_probs.get(10, 0.0) + prob

        elif tok in {"0","2","3","4","5","6","7","8","9"}:
            score = int(tok)
            raw_probs[score] = raw_probs.get(score, 0.0) + prob

        elif tok == "1":
            if has_dedicated_ten:
                # "1" is unambiguously score 1
                raw_probs[1] = raw_probs.get(1, 0.0) + prob
            else:
                # No dedicated "10" token: "1" is the first half of score 10
                raw_probs[10] = raw_probs.get(10, 0.0) + prob

    total = sum(raw_probs.values())
    if total <= 0:
        return None
    return {i: round(raw_probs.get(i, 0.0) / total, 6) for i in range(11)}

CHECKPOINT_EVERY = 5
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
                     api_key: str | None = None,
                     base_url: str = "http://localhost:11434",
                     temperature: float = 0.0) -> tuple[str, dict | None]:
    """Call LLM with exponential backoff. Returns (raw_text, score_probs).

    score_probs is a {0..10 -> float} normalised probability dict for local
    models (via Ollama logprobs), None for API providers.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "local":
                from openai import OpenAI as _OpenAI
                client = _OpenAI(
                    base_url=f"{base_url}/v1",
                    api_key="ollama",
                )
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=10,
                    logprobs=True,
                    top_logprobs=20,
                )
                raw_text = response.choices[0].message.content or ""
                score_probs = _extract_score_probs(response.choices[0].logprobs)
                return raw_text, score_probs
            elif provider == "google":
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
                return response.text, None
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
                return response.choices[0].message.content, None
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


# ── Async API Call with Retry ────────────────────────────────────

async def _call_with_retry_async(provider: str, model_id: str, user_prompt: str,
                                   api_key: str | None = None,
                                   base_url: str = "http://localhost:11434",
                                   temperature: float = 0.0) -> tuple[str, dict | None]:
    """Async version of _call_with_retry for parallel execution.
    
    Returns (raw_text, score_probs).
    score_probs is a {0..10 -> float} normalised probability dict for local
    models (via Ollama logprobs), None for API providers.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "local":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url=f"{base_url}/v1",
                    api_key="ollama",
                )
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=10,
                    logprobs=True,
                    top_logprobs=20,
                )
                raw_text = response.choices[0].message.content or ""
                score_probs = _extract_score_probs(response.choices[0].logprobs)
                return raw_text, score_probs
            elif provider == "google":
                # Google genai doesn't have async support, run in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: _call_with_retry(provider, model_id, user_prompt,
                                            api_key, base_url, temperature)
                )
            elif provider == "openai":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=1024,
                )
                return response.choices[0].message.content, None
        except Exception as e:
            err = str(e).lower()
            transient = any(k in err for k in
                            ["429", "rate", "limit", "quota", "503",
                             "overloaded", "timeout", "connection", "unavailable"])
            if transient and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"\n  [Retry {attempt+1}/{MAX_RETRIES}] "
                      f"{str(e)[:80]}... waiting {wait}s")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("All retries failed")


# ── Score All Samples (with async parallelism) ──────────────────

async def _score_sample_async(sample: dict, model_cfg: dict, api_key: str | None,
                               semaphore: asyncio.Semaphore) -> dict:
    """Score a single sample asynchronously."""
    async with semaphore:  # Limit to 4 concurrent requests
        provider = model_cfg["provider"]
        model_id = model_cfg["model_id"]
        model_name = model_cfg["name"]
        
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )

        try:
            raw, score_probs = await _call_with_retry_async(
                provider, model_id, user_prompt,
                api_key=api_key,
                base_url=model_cfg.get("base_url", "http://localhost:11434"),
                temperature=model_cfg.get("temperature", 0.0),
            )
            score = parse_score(raw)
            entry = {
                "sample_id": sample["id"],
                "model": model_name,
                "condition": "isolated",
                "repetition": 0,
                "score": score,
                "raw_response": raw,
                **({
                    "score_probs": score_probs
                } if score_probs is not None else {}),
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

        return entry


def _score_model(samples: list[dict], model_cfg: dict,
                 output_dir: Path, api_key: str | None = None) -> list[dict]:
    """Score all samples for one model with 4 concurrent workers. Checkpoints every 5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_cfg["name"]

    ckpt = _checkpoint_path(output_dir, model_name)
    results, done_ids = _load_checkpoint(ckpt)

    if done_ids:
        print(f"  [Resume] {len(done_ids)} already scored, continuing...")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        return results

    # Run async scoring
    async def _run_async():
        semaphore = asyncio.Semaphore(4)  # Max 4 concurrent requests
        buffer = []
        
        # Process in batches of 4 for better checkpoint alignment
        with tqdm(total=len(samples), initial=len(done_ids),
                  desc=f"Scoring [{model_name}]") as pbar:
            
            for i in range(0, len(remaining), 4):
                batch = remaining[i:i+4]
                
                # Launch 4 concurrent tasks
                tasks = [
                    _score_sample_async(sample, model_cfg, api_key, semaphore)
                    for sample in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                
                # Add to results and buffer
                results.extend(batch_results)
                buffer.extend(batch_results)
                pbar.update(len(batch_results))
                
                # Checkpoint every 5 samples (or at batch completion)
                if len(buffer) >= CHECKPOINT_EVERY:
                    _flush_checkpoint(ckpt, buffer)
                    buffer.clear()
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Flush any remaining buffered entries
            if buffer:
                _flush_checkpoint(ckpt, buffer)
        
        return results

    # Run the async function
    return asyncio.run(_run_async())


# ── Main Runner ─────────────────────────────────────────────────

def _model_output_path(output_dir: Path, model_name: str) -> Path:
    safe = model_name.replace(" ", "_").replace("/", "_").replace(":", "_")
    return output_dir / f"{safe}_scores.json"


def run(config: dict, samples: list[dict]) -> list[dict]:
    """Run LLM scoring across all models, saving per-model score files."""
    llm_cfg = config["llm_scoring"]
    output_dir = Path(llm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_cfg in llm_cfg["models"]:
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]
        model_file = _model_output_path(output_dir, model_name)

        if model_file.exists():
            print(f"[LLM Scoring] {model_name}: already done → {model_file.name}")
            with open(model_file, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            continue

        print(f"[LLM Scoring] {model_name} ({provider})")
        results = _score_model(samples, model_cfg, output_dir)
        all_results.extend(results)

        # Save per-model file
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  → Saved {len(results)} ratings to {model_file.name}")

        # Clean up checkpoint
        ckpt = _checkpoint_path(output_dir, model_name)
        if ckpt.exists():
            ckpt.unlink()
            print(f"  [Cleanup] Removed {ckpt.name}")

    return all_results