"""
Step 4 — LLM Scoring Module
Sends degraded texts to LLMs for quality rating (0-10 scale).

Google:  Batch API (50% cheaper, handles rate limits automatically)
OpenAI:  Standard API with retry
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "Rate the quality of the following text from 0 to 10. Respond with ONLY the number."

POLL_INTERVAL = 30  # seconds between batch status checks


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


# ── Google Batch API ─────────────────────────────────────────────

def _build_batch_requests(samples: list[dict]) -> list[dict]:
    """Build list of GenerateContentRequest dicts for the Batch API."""
    requests = []
    for sample in samples:
        user_text = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )
        requests.append({
            "key": sample["id"],
            "request": {
                "contents": [{"parts": [{"text": user_text}], "role": "user"}],
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generation_config": {"temperature": 0.0},
            },
        })
    return requests


def _score_google_batch(samples: list[dict], model_cfg: dict,
                        output_dir: Path) -> list[dict]:
    """Score all samples via Google Batch API (JSONL file method)."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    model_id = model_cfg["model_id"]
    model_name = model_cfg["name"]

    # Check for existing batch job (resume support)
    job_file = output_dir / f"batch_job_{model_name}.json"
    if job_file.exists():
        with open(job_file, "r") as f:
            job_info = json.load(f)
        job_name = job_info["name"]
        print(f"  [Resume] Found existing batch job: {job_name}")
    else:
        # Build JSONL file
        batch_requests = _build_batch_requests(samples)
        jsonl_path = output_dir / f"batch_input_{model_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        print(f"  [Upload] {len(batch_requests)} requests → {jsonl_path.name}")

        # Upload file
        uploaded = client.files.upload(
            file=str(jsonl_path),
            config=types.UploadFileConfig(
                display_name=f"batch-{model_name}",
                mime_type="jsonl",
            ),
        )
        print(f"  [Upload] File uploaded: {uploaded.name}")

        # Create batch job
        batch_job = client.batches.create(
            model=model_id,
            src=uploaded.name,
            config={"display_name": f"scoring-{model_name}"},
        )
        job_name = batch_job.name
        print(f"  [Batch] Created job: {job_name}")

        # Save job name for resume
        with open(job_file, "w") as f:
            json.dump({"name": job_name, "model": model_name,
                       "num_requests": len(batch_requests)}, f)

    # Poll until done
    completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                 "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        if state in completed:
            break
        print(f"  [Poll] {state} — waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    print(f"  [Batch] Finished: {state}")

    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {state}: {getattr(job, 'error', 'unknown')}")

    # Download results
    results = []
    if job.dest and job.dest.file_name:
        raw_bytes = client.files.download(file=job.dest.file_name)
        raw_text = raw_bytes.decode("utf-8")
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            sample_id = entry.get("key", "")
            if "response" in entry and entry["response"]:
                resp = entry["response"]
                try:
                    raw_response = resp["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    raw_response = str(resp)
                score = parse_score(raw_response)
                results.append({
                    "sample_id": sample_id,
                    "model": model_name,
                    "condition": "isolated",
                    "repetition": 0,
                    "score": score,
                    "raw_response": raw_response,
                    **({"parse_error": True} if score is None else {}),
                })
            else:
                error_msg = str(entry.get("error", "unknown error"))
                results.append({
                    "sample_id": sample_id,
                    "model": model_name,
                    "condition": "isolated",
                    "repetition": 0,
                    "score": None,
                    "raw_response": error_msg,
                    "parse_error": True,
                })
    elif job.dest and job.dest.inlined_responses:
        for resp in job.dest.inlined_responses:
            if resp.response:
                raw_response = resp.response.text
                score = parse_score(raw_response)
                results.append({
                    "sample_id": "",
                    "model": model_name,
                    "condition": "isolated",
                    "repetition": 0,
                    "score": score,
                    "raw_response": raw_response,
                    **({"parse_error": True} if score is None else {}),
                })

    # Clean up job file and input JSONL
    job_file.unlink(missing_ok=True)
    (output_dir / f"batch_input_{model_name}.jsonl").unlink(missing_ok=True)

    print(f"  [Done] {len(results)} results, "
          f"{sum(1 for r in results if r['score'] is not None)} parsed OK")
    return results


# ── OpenAI Standard API ──────────────────────────────────────────

def _score_openai(samples: list[dict], model_cfg: dict,
                  output_dir: Path) -> list[dict]:
    """Score all samples via OpenAI standard API with retry."""
    from openai import OpenAI
    from tqdm import tqdm

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_id = model_cfg["model_id"]
    model_name = model_cfg["name"]

    # Checkpoint for resume
    ckpt_path = output_dir / f"checkpoint_{model_name}.jsonl"
    results = []
    done_ids = set()
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    results.append(entry)
                    done_ids.add(entry["sample_id"])
        print(f"  [Resume] {len(done_ids)} already scored")

    remaining = [s for s in samples if s["id"] not in done_ids]
    buffer = []

    for sample in tqdm(remaining, desc=f"Scoring [{model_name}]",
                       initial=len(done_ids), total=len(samples)):
        user_prompt = (
            f"Please rate the quality of the following text:\n\n"
            f"---\n{sample['degraded_text']}\n---"
        )

        raw_response = None
        for attempt in range(5):
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
                raw_response = resp.choices[0].message.content
                break
            except Exception as e:
                err = str(e).lower()
                transient = any(k in err for k in
                                ["429", "rate", "limit", "timeout", "503"])
                if transient and attempt < 4:
                    wait = 2 ** attempt * 5
                    print(f"\n  [Retry {attempt+1}/5] waiting {wait}s...")
                    time.sleep(wait)
                elif not transient:
                    raw_response = str(e)
                    break

        score = parse_score(raw_response) if raw_response else None
        entry = {
            "sample_id": sample["id"],
            "model": model_name,
            "condition": "isolated",
            "repetition": 0,
            "score": score,
            "raw_response": raw_response or "all retries failed",
            **({"parse_error": True} if score is None else {}),
        }
        results.append(entry)
        buffer.append(entry)

        if len(buffer) >= 50:
            with open(ckpt_path, "a", encoding="utf-8") as f:
                for e in buffer:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            buffer.clear()

        time.sleep(0.3)

    # Flush
    if buffer:
        with open(ckpt_path, "a", encoding="utf-8") as f:
            for e in buffer:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    ckpt_path.unlink(missing_ok=True)

    print(f"  [Done] {len(results)} results, "
          f"{sum(1 for r in results if r['score'] is not None)} parsed OK")
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

        if provider == "google":
            results = _score_google_batch(samples, model_cfg, output_dir)
        elif provider == "openai":
            results = _score_openai(samples, model_cfg, output_dir)
        else:
            print(f"  [SKIP] Unknown provider: {provider}")
            continue

        all_results.extend(results)

    # Save final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[LLM Scoring] Saved {len(all_results)} ratings to {output_file}")

    return all_results
