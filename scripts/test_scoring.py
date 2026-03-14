"""
Dry-run test: send 3 samples to each model to verify the scoring pipeline works.

Google:  Batch API (small inline batch)
OpenAI:  Standard API (3 sequential calls)
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from src.llm_scoring import parse_score, SYSTEM_PROMPT


def load_test_samples(n=3):
    """Load n degraded samples — pick varied degradation levels."""
    deg_path = Path("data/degraded/degraded_samples.json")
    with open(deg_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Pick one low, one mid, one high degradation
    by_level = {}
    for s in samples:
        lvl = s.get("level", 0)
        by_level.setdefault(lvl, []).append(s)

    levels = sorted(by_level.keys())
    picks = []
    for lvl in [levels[0], levels[len(levels) // 2], levels[-1]]:
        picks.append(by_level[lvl][0])
        if len(picks) >= n:
            break

    return picks


def test_google_batch(samples):
    """Test Google Batch API with inline requests (no file upload needed)."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[SKIP] GOOGLE_API_KEY not set")
        return

    client = genai.Client(api_key=api_key)
    model_id = "gemini-3-flash-preview"

    # Build inline requests
    inline_requests = []
    for sample in samples:
        user_text = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text'][:500]}\n---"
        )
        inline_requests.append({
            "contents": [{"parts": [{"text": user_text}], "role": "user"}],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generation_config": {"temperature": 0.0},
        })

    print(f"[Google] Submitting {len(inline_requests)} inline requests...")
    batch_job = client.batches.create(
        model=model_id,
        src=inline_requests,
        config={"display_name": "test-scoring"},
    )
    print(f"[Google] Job created: {batch_job.name}")

    # Poll
    completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                 "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        job = client.batches.get(name=batch_job.name)
        state = job.state.name
        if state in completed:
            break
        print(f"  [{state}] waiting 10s...")
        time.sleep(10)

    print(f"[Google] Final state: {state}")

    if state == "JOB_STATE_SUCCEEDED":
        if job.dest and job.dest.inlined_responses:
            for i, resp in enumerate(job.dest.inlined_responses):
                if resp.response:
                    raw = resp.response.text
                    score = parse_score(raw)
                    lvl = samples[i].get("level", "?")
                    axis = samples[i].get("axis", "?")
                    print(f"  Sample {i+1} (axis={axis}, level={lvl}): "
                          f"score={score}, raw='{raw.strip()}'")
                elif resp.error:
                    print(f"  Sample {i+1}: ERROR {resp.error}")
        else:
            print("  No inline responses found")
    else:
        print(f"  Error: {getattr(job, 'error', 'unknown')}")


def test_openai(samples):
    """Test OpenAI standard API with a few calls."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[SKIP] OPENAI_API_KEY not set")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model_id = "gpt-4.1-nano"

    print(f"[OpenAI] Sending {len(samples)} requests...")
    for i, sample in enumerate(samples):
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text'][:500]}\n---"
        )
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            raw = resp.choices[0].message.content
            score = parse_score(raw)
            lvl = sample.get("level", "?")
            axis = sample.get("axis", "?")
            print(f"  Sample {i+1} (axis={axis}, level={lvl}): "
                  f"score={score}, raw='{raw.strip()}'")
        except Exception as e:
            print(f"  Sample {i+1}: ERROR {e}")


if __name__ == "__main__":
    samples = load_test_samples(3)
    print(f"Loaded {len(samples)} test samples\n")

    for s in samples:
        print(f"  id={s['id']}, axis={s.get('axis','?')}, "
              f"level={s.get('level','?')}, "
              f"text_len={len(s['degraded_text'])}")
    print()

    test_google_batch(samples)
    print("\nDone.")
