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
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_TOP_LOGPROBS = 5

_SCORE_TOKENS = frozenset({"0","1","2","3","4","5","6","7","8","9","10"})
_SINGLE_DIGIT_TOKENS = frozenset({"0","1","2","3","4","5","6","7","8","9"})


def _extract_terminal_score_text(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None

    if re.fullmatch(r"(?:10|\d)", stripped):
        return stripped

    lines = [line.strip().strip("`*\"'") for line in stripped.splitlines() if line.strip()]
    if lines and re.fullmatch(r"(?:10|\d)", lines[-1]):
        return lines[-1]

    return None


def _normalise_score_probs(raw_probs: dict[int, float]) -> dict | None:
    total = sum(raw_probs.values())
    if total <= 0:
        return None
    return {i: round(raw_probs.get(i, 0.0) / total, 6) for i in range(11)}


def _is_gpt_oss_model(model_id: str) -> bool:
    return "gpt-oss" in model_id.lower()


def _is_reasoning_model(model_id: str) -> bool:
    """Models that emit hidden reasoning tokens requiring larger max_tokens budget."""
    mid = model_id.lower()
    return "gpt-oss" in mid or "minimax" in mid or "deepseek" in mid


def _concurrency_for_model(model_cfg: dict) -> int:
    provider = model_cfg.get("provider")
    if provider == "local":
        return 1

    model_id = str(model_cfg.get("model_id", "")).lower()
    if provider == "fireworks" and "deepseek" in model_id:
        return 4

    if provider == "fireworks" and "minimax" in model_id:
        return 8

    return 32


def _extract_score_probs_from_top_tokens(first_tok: str,
                                         first_prob: float | None,
                                         second_tok: str | None,
                                         second_logprob: float | None,
                                         top_tokens: dict[str, float]) -> dict | None:
    """Build normalised score probabilities from a first-token logprob view."""
    raw_probs: dict[int, float] = {}
    has_dedicated_ten = "10" in top_tokens

    if first_prob is not None:
        if first_tok == "10":
            raw_probs[10] = first_prob
        elif first_tok == "1" and second_tok == "0" and second_logprob is not None:
            raw_probs[10] = math.exp(math.log(first_prob) + second_logprob)
        elif first_tok in _SINGLE_DIGIT_TOKENS:
            raw_probs[int(first_tok)] = first_prob

    for tok, prob in top_tokens.items():
        if tok == "10":
            raw_probs[10] = raw_probs.get(10, 0.0) + prob
        elif tok in {"0", "2", "3", "4", "5", "6", "7", "8", "9"}:
            score = int(tok)
            raw_probs[score] = raw_probs.get(score, 0.0) + prob
        elif tok == "1":
            if has_dedicated_ten:
                raw_probs[1] = raw_probs.get(1, 0.0) + prob
            else:
                raw_probs[10] = raw_probs.get(10, 0.0) + prob

    return _normalise_score_probs(raw_probs)


def _extract_score_probs_from_legacy_position(tokens, token_logprobs, top_logprobs,
                                              index: int) -> dict | None:
    first_tok = str(tokens[index]).strip()
    first_logprob = token_logprobs[index]
    first_prob = math.exp(first_logprob) if first_logprob is not None else None
    raw_top = top_logprobs[index] or {}
    top_tokens = {
        str(tok).strip(): math.exp(logprob)
        for tok, logprob in dict(raw_top).items()
        if logprob is not None
    }
    second_tok = str(tokens[index + 1]).strip() if index + 1 < len(tokens) else None
    second_logprob = token_logprobs[index + 1] if index + 1 < len(token_logprobs) else None
    return _extract_score_probs_from_top_tokens(
        first_tok,
        first_prob,
        second_tok,
        second_logprob,
        top_tokens,
    )


def _extract_score_probs_from_content_position(content, index: int) -> dict | None:
    first_pos = content[index]
    first_tok = first_pos.token.strip()
    first_prob = math.exp(first_pos.logprob)
    top_tokens = {
        alt.token.strip(): math.exp(alt.logprob)
        for alt in (first_pos.top_logprobs or [])
    }
    second_tok = content[index + 1].token.strip() if index + 1 < len(content) else None
    second_logprob = content[index + 1].logprob if index + 1 < len(content) else None
    return _extract_score_probs_from_top_tokens(
        first_tok,
        first_prob,
        second_tok,
        second_logprob,
        top_tokens,
    )


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
    if not lp:
        return None

    content = getattr(lp, "content", None)
    if content:
        for index in range(len(content) - 1, -1, -1):
            stripped = content[index].token.strip()
            if stripped in _SCORE_TOKENS or stripped in _SINGLE_DIGIT_TOKENS:
                probs = _extract_score_probs_from_content_position(content, index)
                if probs is not None:
                    return probs

        return _extract_score_probs_from_content_position(content, 0)

    tokens = getattr(lp, "tokens", None)
    token_logprobs = getattr(lp, "token_logprobs", None)
    top_logprobs = getattr(lp, "top_logprobs", None)
    if tokens and token_logprobs and top_logprobs:
        for index in range(len(tokens) - 1, -1, -1):
            stripped = str(tokens[index]).strip()
            if stripped in _SCORE_TOKENS or stripped in _SINGLE_DIGIT_TOKENS:
                probs = _extract_score_probs_from_legacy_position(
                    tokens,
                    token_logprobs,
                    top_logprobs,
                    index,
                )
                if probs is not None:
                    return probs

        return _extract_score_probs_from_legacy_position(
            tokens,
            token_logprobs,
            top_logprobs,
            0,
        )

    return None


def _completion_kwargs(provider: str, model_id: str, temperature: float) -> dict:
    kwargs = {
        "temperature": temperature,
        "max_tokens": 10,
    }

    if provider == "fireworks":
        kwargs.update({
            "logprobs": True,
            "top_logprobs": FIREWORKS_TOP_LOGPROBS,
        })

    if provider == "fireworks" and _is_reasoning_model(model_id):
        kwargs["max_tokens"] = 512

    if provider == "fireworks" and _is_gpt_oss_model(model_id):
        kwargs["reasoning_effort"] = "low"

    if provider == "fireworks" and "deepseek" in model_id.lower():
        kwargs["reasoning_effort"] = "low"

    return kwargs

CHECKPOINT_EVERY = 5
MAX_RETRIES = 5


def _resolve_base_url(provider: str, base_url: str | None = None) -> str | None:
    """Resolve the correct OpenAI-compatible base URL for a provider."""
    if provider == "local":
        root = (base_url or "http://localhost:11434").rstrip("/")
        return root if root.endswith("/v1") else f"{root}/v1"
    if provider == "fireworks":
        return (base_url or FIREWORKS_BASE_URL).rstrip("/")
    if provider == "openai":
        return (base_url or os.environ.get("OPENAI_API_BASE") or "").rstrip("/") or None
    return base_url


def _resolve_api_key(provider: str, api_key: str | None = None,
                     api_key_env: str | None = None) -> str | None:
    """Resolve API key from explicit value, custom env var, or provider default."""
    if api_key:
        return api_key
    if api_key_env:
        value = os.environ.get(api_key_env)
        if value:
            return value
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "fireworks":
        return os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return None


def _resolve_model_id(model_cfg: dict) -> str:
    """Resolve model id, optionally targeting a Fireworks deployment."""
    model_id = model_cfg["model_id"]
    deployment_name = model_cfg.get("deployment_name")
    deployment_name_env = model_cfg.get("deployment_name_env")

    if not deployment_name and deployment_name_env:
        deployment_name = os.environ.get(deployment_name_env)

    if deployment_name:
        return f"{model_id}#{deployment_name}"
    return model_id


# ── Response Parsing ─────────────────────────────────────────────

def parse_score(raw: str) -> int | None:
    """Extract score 0-10 from raw LLM response."""
    try:
        score_text = _extract_terminal_score_text(raw)
        if score_text is not None:
            score = int(score_text)
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
                    # Only skip samples that were successfully scored;
                    # null-score entries (parse failures) will be retried.
                    if entry.get("score") is not None:
                        done_ids.add(entry["sample_id"])
    return results, done_ids


def _flush_checkpoint(ckpt_path: Path, buffer: list[dict]):
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for entry in buffer:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── API Call with Retry ──────────────────────────────────────────

def _call_with_retry(provider: str, model_id: str, user_prompt: str,
                     api_key: str | None = None,
                     api_key_env: str | None = None,
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
                    base_url=_resolve_base_url(provider, base_url),
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
            elif provider in {"openai", "fireworks"}:
                from openai import OpenAI
                client = OpenAI(
                    base_url=_resolve_base_url(provider, base_url),
                    api_key=_resolve_api_key(provider, api_key, api_key_env),
                )
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    **_completion_kwargs(provider, model_id, temperature),
                )
                raw_text = response.choices[0].message.content or ""
                if provider == "fireworks" and "minimax" in model_id.lower() and len(raw_text) > 50:
                    matches = re.findall(r'\b(10|\d)\b', raw_text)
                    if matches:
                        raw_text = matches[-1]
                score_probs = _extract_score_probs(response.choices[0].logprobs)
                return raw_text, score_probs
        except Exception as e:
            err = str(e).lower()
            transient = any(k in err for k in
                            ["429", "500", "502", "503", "504", "rate", "limit", "quota",
                             "overloaded", "timeout", "connection", "unavailable"])
            if transient and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"\n  [Retry {attempt+1}/{MAX_RETRIES}] "
                      f"{str(e)[:80]}... waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("All retries failed")


# ── Client Factory (create once, reuse) ───────────────────────────

def _make_async_client(provider: str, base_url: str = "http://localhost:11434",
                       api_key: str | None = None,
                       api_key_env: str | None = None):
    """Create a reusable async client for the given provider."""
    from openai import AsyncOpenAI
    if provider == "local":
        return AsyncOpenAI(base_url=_resolve_base_url(provider, base_url), api_key="ollama")
    elif provider in {"openai", "fireworks"}:
        return AsyncOpenAI(
            base_url=_resolve_base_url(provider, base_url),
            api_key=_resolve_api_key(provider, api_key, api_key_env),
        )
    return None


# ── Async API Call with Retry ────────────────────────────────────

async def _call_with_retry_async(provider: str, model_id: str, user_prompt: str,
                                   client=None,
                                   api_key: str | None = None,
                                   api_key_env: str | None = None,
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
                                            api_key, api_key_env, base_url, temperature)
                )
            elif provider in {"openai", "fireworks"}:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    **_completion_kwargs(provider, model_id, temperature),
                )
                raw_text = response.choices[0].message.content or ""
                if provider == "fireworks" and "minimax" in model_id.lower() and len(raw_text) > 50:
                    matches = re.findall(r'\b(10|\d)\b', raw_text)
                    if matches:
                        raw_text = matches[-1]
                score_probs = _extract_score_probs(response.choices[0].logprobs)
                return raw_text, score_probs
        except Exception as e:
            err = str(e).lower()
            transient = any(k in err for k in
                            ["429", "500", "502", "503", "504", "rate", "limit", "quota",
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
                               semaphore: asyncio.Semaphore,
                               client=None) -> dict:
    """Score a single sample asynchronously."""
    async with semaphore:
        provider = model_cfg["provider"]
        model_id = _resolve_model_id(model_cfg)
        model_name = model_cfg["name"]
        
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )

        try:
            raw, score_probs = await _call_with_retry_async(
                provider, model_id, user_prompt,
                client=client,
                api_key=api_key,
                api_key_env=model_cfg.get("api_key_env"),
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

    concurrency = _concurrency_for_model(model_cfg)

    # Create a single reusable client for this model
    provider = model_cfg["provider"]
    client = _make_async_client(
        provider,
        base_url=model_cfg.get("base_url", "http://localhost:11434"),
        api_key=api_key,
        api_key_env=model_cfg.get("api_key_env"),
    )

    # Run async scoring
    async def _run_async():
        semaphore = asyncio.Semaphore(concurrency)
        buffer = []
        
        with tqdm(total=len(samples), initial=len(done_ids),
                  desc=f"Scoring [{model_name}]") as pbar:
            
            for i in range(0, len(remaining), concurrency):
                batch = remaining[i:i+concurrency]
                
                tasks = [
                    _score_sample_async(sample, model_cfg, api_key, semaphore,
                                        client=client)
                    for sample in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                
                results.extend(batch_results)
                buffer.extend(batch_results)
                pbar.update(len(batch_results))
                
                if len(buffer) >= CHECKPOINT_EVERY:
                    _flush_checkpoint(ckpt, buffer)
                    buffer.clear()
            
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

    max_samples = llm_cfg.get("max_samples")
    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"[LLM Scoring] Limiting run to first {len(samples)} samples")

    all_results = []

    model_cfgs = [m for m in llm_cfg["models"] if m.get("enabled", True)]
    for model_cfg in model_cfgs:
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]
        model_file = _model_output_path(output_dir, model_name)

        if model_file.exists():
            with open(model_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            null_count = sum(1 for r in existing if r.get("score") is None)
            if null_count == 0:
                print(f"[LLM Scoring] {model_name}: already done → {model_file.name}")
                all_results.extend(existing)
                continue
            # Null scores present — rebuild checkpoint from valid entries and retry
            print(f"[LLM Scoring] {model_name}: {null_count} null scores found, retrying failed samples...")
            ckpt = _checkpoint_path(output_dir, model_name)
            valid_entries = [r for r in existing if r.get("score") is not None]
            with open(ckpt, "w", encoding="utf-8") as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            model_file.unlink()

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