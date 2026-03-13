"""
Step 4 — LLM Scoring Module
Sends degraded texts to LLMs for quality rating (0-10 scale).

Each text is scored independently (isolated mode).
Supports dual API key rotation for Google free-tier throughput doubling.

Robustness:
  - Retries with exponential backoff on rate-limit / transient errors
  - Checkpoint file saved every 50 samples — resume from where you left off
"""

import itertools
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

MAX_RETRIES = 5
CHECKPOINT_EVERY = 50


class KeyRotator:
    """Smart key rotator: prioritizes free keys, hard-caps paid key usage.

    Free keys are rotated round-robin. The paid key is only used as
    fallback when free keys hit rate limits, up to max_requests.
    """

    def __init__(self, key_entries: list[dict]):
        self._free_keys = []
        self._paid_key = None
        self._paid_max = 0
        self._paid_used = 0

        for entry in key_entries:
            env_var = entry["env"]
            key = os.environ.get(env_var)
            if not key:
                continue
            if entry.get("paid", False):
                self._paid_key = key
                self._paid_max = entry.get("max_requests", 100)
            else:
                self._free_keys.append(key)

        if not self._free_keys and not self._paid_key:
            raise RuntimeError("No API keys found")

        self._free_cycle = itertools.cycle(self._free_keys) if self._free_keys else None

    def next_free(self) -> str | None:
        """Get next free key (round-robin). Returns None if no free keys."""
        if self._free_cycle:
            return next(self._free_cycle)
        return None

    def next_paid(self) -> str | None:
        """Get paid key if under budget. Returns None if exhausted."""
        if self._paid_key and self._paid_used < self._paid_max:
            self._paid_used += 1
            return self._paid_key
        return None

    @property
    def paid_remaining(self) -> int:
        return self._paid_max - self._paid_used

    @property
    def has_free_keys(self) -> bool:
        return len(self._free_keys) > 0

    def summary(self) -> str:
        parts = [f"{len(self._free_keys)} free key(s)"]
        if self._paid_key:
            parts.append(f"1 paid key (cap: {self._paid_max})")
        return ", ".join(parts)

# ── Prompt Templates ─────────────────────────────────────────────

ISOLATED_SYSTEM_PROMPT = """Rate the quality of the following text from 0 to 10. Respond with ONLY the number."""


# ── API Clients ──────────────────────────────────────────────────

def _call_openai(model_id: str, system_prompt: str, user_prompt: str,
                 api_key: str | None = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    return response.choices[0].message.content


def _call_google(model_id: str, system_prompt: str, user_prompt: str,
                 api_key: str | None = None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY_1")
                    or os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        model_id,
        system_instruction=system_prompt,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )
    return response.text


PROVIDER_DISPATCH = {
    "openai": _call_openai,
    "google": _call_google,
}


def _call_with_retry(provider: str, model_id: str,
                     system_prompt: str, user_prompt: str,
                     key_rotator: KeyRotator | None = None) -> str:
    """Call LLM with smart key rotation and exponential backoff.

    Strategy:
    1. Try a free key first.
    2. If free key hits rate limit, back off and retry with next free key.
    3. If all free retries exhausted, fall back to paid key (if under budget).
    4. If paid key also fails or budget exhausted, raise.
    """
    fn = PROVIDER_DISPATCH[provider]
    last_error = None

    # Phase 1: try free keys with backoff
    if key_rotator and key_rotator.has_free_keys:
        for attempt in range(MAX_RETRIES):
            api_key = key_rotator.next_free()
            try:
                return fn(model_id, system_prompt, user_prompt, api_key=api_key)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in [
                    "429", "rate", "limit", "quota", "503", "overloaded",
                    "timeout", "timed out", "connection", "unavailable",
                ])
                if is_transient and attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                    print(f"\n  [Free key retry {attempt+1}/{MAX_RETRIES}] "
                          f"{str(e)[:80]}... waiting {wait}s")
                    time.sleep(wait)
                elif not is_transient:
                    raise  # auth error or something fatal

    # Phase 2: fall back to paid key
    if key_rotator:
        paid_key = key_rotator.next_paid()
        if paid_key:
            try:
                return fn(model_id, system_prompt, user_prompt, api_key=paid_key)
            except Exception as e:
                last_error = e
                print(f"\n  [Paid key failed] {str(e)[:100]}")

    # Phase 3: no key, try with env default (for OpenAI)
    if not key_rotator:
        for attempt in range(MAX_RETRIES):
            try:
                return fn(model_id, system_prompt, user_prompt, api_key=None)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in [
                    "429", "rate", "limit", "quota", "503", "overloaded",
                    "timeout", "timed out", "connection", "unavailable",
                ])
                if is_transient and attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt * 5
                    print(f"\n  [Retry {attempt+1}/{MAX_RETRIES}] "
                          f"{str(e)[:80]}... waiting {wait}s")
                    time.sleep(wait)
                elif not is_transient:
                    raise

    raise last_error or RuntimeError("All keys exhausted")


# ── Response Parsing ─────────────────────────────────────────────

def parse_isolated_response(raw: str) -> dict | None:
    """Extract score from isolated response (just a number 0-10)."""
    try:
        match = re.search(r'\b(10|\d)\b', raw.strip())
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return {"score": score, "raw_response": raw}
    except (ValueError, AttributeError):
        pass
    return {"score": None, "raw_response": raw, "parse_error": True}


# ── Checkpoint Helpers ───────────────────────────────────────────

def _checkpoint_path(output_dir: Path, model_name: str) -> Path:
    """Path for the incremental checkpoint file."""
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    return output_dir / f"checkpoint_{safe_name}.jsonl"


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set]:
    """Load existing checkpoint. Returns (results_list, set_of_done_sample_ids)."""
    results = []
    done_ids = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    results.append(entry)
                    done_ids.add(entry["sample_id"])
    return results, done_ids


def _append_checkpoint(ckpt_path: Path, entries: list[dict]):
    """Append entries to the checkpoint file (JSONL format)."""
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Isolated Scoring ─────────────────────────────────────────────

def score_isolated(samples: list[dict], model_cfg: dict,
                   output_dir: Path,
                   repetitions: int = 1,
                   key_rotator: KeyRotator | None = None) -> list[dict]:
    """Rate each sample independently. Saves checkpoint every N samples."""
    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]
    model_name = model_cfg["name"]

    # Load checkpoint to resume
    ckpt_path = _checkpoint_path(output_dir, model_name)
    results, done_ids = _load_checkpoint(ckpt_path)

    if done_ids:
        print(f"  [Resume] Found {len(done_ids)} already scored, "
              f"skipping to remaining")

    # Filter to only unscored samples
    remaining = [s for s in samples if s["id"] not in done_ids]
    pending_buffer = []

    for sample in tqdm(remaining, desc=f"Scoring [{model_name}]",
                       initial=len(done_ids), total=len(samples)):
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )

        for rep in range(repetitions):
            try:
                raw = _call_with_retry(provider, model_id,
                                       ISOLATED_SYSTEM_PROMPT, user_prompt,
                                       key_rotator=key_rotator)
                parsed = parse_isolated_response(raw)
            except Exception as e:
                parsed = {"score": None, "raw_response": str(e),
                          "parse_error": True}

            entry = {
                "sample_id": sample["id"],
                "model": model_name,
                "condition": "isolated",
                "repetition": rep,
                **parsed,
            }
            results.append(entry)
            pending_buffer.append(entry)

            time.sleep(0.5)

        # Flush checkpoint every N samples
        if len(pending_buffer) >= CHECKPOINT_EVERY:
            _append_checkpoint(ckpt_path, pending_buffer)
            pending_buffer.clear()

    # Flush remaining
    if pending_buffer:
        _append_checkpoint(ckpt_path, pending_buffer)
        pending_buffer.clear()

    return results


# ── Main Runner ─────────────────────────────────────────────────

def run(config: dict, samples: list[dict]) -> list[dict]:
    """Run LLM scoring across all models. Checkpoints after every 50 samples."""
    llm_cfg = config["llm_scoring"]
    output_dir = Path(llm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "llm_scores.json"
    if output_file.exists():
        print("[LLM Scoring] Loading existing LLM scores...")
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    all_results = []
    repetitions = llm_cfg.get("repetitions", 1)

    # Build key rotators from config
    rotation_cfg = llm_cfg.get("api_key_rotation", {})
    rotators: dict[str, KeyRotator | None] = {}
    for provider_name, key_entries in rotation_cfg.items():
        # Filter to entries where the env var is actually set
        available = [e for e in key_entries if os.environ.get(e["env"])]
        if available:
            rotators[provider_name] = KeyRotator(available)
            print(f"  [Keys] {provider_name}: "
                  f"{rotators[provider_name].summary()}")

    for model_cfg in llm_cfg["models"]:
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]

        key_rotator = rotators.get(provider)
        if not key_rotator:
            legacy_keys = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
            env_key = legacy_keys.get(provider, "")
            if not os.environ.get(env_key):
                print(f"  [SKIP] {model_name}: no API keys set")
                continue

        print(f"[LLM Scoring] {model_name} — isolated")
        results = score_isolated(
            samples, model_cfg, output_dir, repetitions,
            key_rotator=key_rotator,
        )
        all_results.extend(results)

    # Save final consolidated file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[LLM Scoring] Saved {len(all_results)} ratings to {output_file}")

    # Report paid key usage
    for provider_name, rotator in rotators.items():
        if rotator._paid_key:
            print(f"  [Paid key] {provider_name}: "
                  f"used {rotator._paid_used}/{rotator._paid_max} requests")

    # Clean up checkpoint files
    for model_cfg in llm_cfg["models"]:
        ckpt = _checkpoint_path(output_dir, model_cfg["name"])
        if ckpt.exists():
            ckpt.unlink()
            print(f"  [Cleanup] Removed checkpoint {ckpt.name}")

    return all_results
