"""
Condemned to B — Main Pipeline Orchestrator

Runs the full research pipeline:
  1. Fetch corpus (Wikipedia articles)
  2. Degrade texts along 4 axes at controlled levels
  3. Compute objective quality scores (Q)
  4. Send to LLMs for rating (GPT-4, Claude, Gemini)
  5. Analyze results and generate figures

Usage:
  python -m src.main                   # Run full pipeline
  python -m src.main --step corpus     # Run single step
  python -m src.main --step degrade
  python -m src.main --step quality
  python -m src.main --step llm
  python -m src.main --step analysis
"""

import argparse
import sys

import yaml

from src import corpus, degradation, quality, llm_scoring, analysis


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, step: str | None = None):
    """Execute the pipeline, optionally starting from a specific step."""

    steps = ["corpus", "degrade", "quality", "llm", "analysis"]
    if step and step not in steps:
        print(f"Unknown step: {step}. Valid steps: {steps}")
        sys.exit(1)

    run_all = step is None

    # Step 1: Corpus
    if run_all or step == "corpus":
        print("=" * 60)
        print("STEP 1: Corpus Acquisition")
        print("=" * 60)
        articles = corpus.run(config)
        print(f"  → {len(articles)} articles ready\n")
    else:
        articles = corpus.load_corpus(config["corpus"]["output_dir"])

    # Step 2: Degradation
    if run_all or step == "degrade":
        print("=" * 60)
        print("STEP 2: Text Degradation")
        print("=" * 60)
        samples = degradation.run(config, articles)
        print(f"  → {len(samples)} degraded samples generated\n")
    else:
        import json
        from pathlib import Path
        deg_path = Path(config["degradation"]["output_dir"]) / "degraded_samples.json"
        with open(deg_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    # Step 3: Quality Scoring
    if run_all or step == "quality":
        print("=" * 60)
        print("STEP 3: Quality Function (Q) Computation")
        print("=" * 60)
        scored_samples = quality.run(config, samples)
        print(f"  → {len(scored_samples)} samples scored\n")
    else:
        import json
        from pathlib import Path
        q_path = Path(config["quality"]["output_dir"]) / "scored_samples.json"
        with open(q_path, "r", encoding="utf-8") as f:
            scored_samples = json.load(f)

    # Step 4: LLM Scoring
    if run_all or step == "llm":
        print("=" * 60)
        print("STEP 4: LLM Scoring")
        print("=" * 60)
        llm_results = llm_scoring.run(config, scored_samples)
        print(f"  → {len(llm_results)} LLM ratings collected\n")
    else:
        import json
        from pathlib import Path
        llm_path = Path(config["llm_scoring"]["output_dir"]) / "llm_scores.json"
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)

    # Step 5: Analysis
    if run_all or step == "analysis":
        print("=" * 60)
        print("STEP 5: Analysis & Visualization")
        print("=" * 60)
        analysis.run(config, scored_samples, llm_results)
        print()

    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Condemned to B — LLM Score Compression Research Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--step", default=None,
        choices=["corpus", "degrade", "quality", "llm", "analysis"],
        help="Run a single pipeline step instead of the full pipeline",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, step=args.step)


if __name__ == "__main__":
    main()
