"""
Step 3 — Quality Function Q
Computes objective quality scores for each degraded sample using model-free metrics.

Q = grammar^w₁ × coherence^w₂ × (1 − deletion)^w₃ × lexical^w₄

Components:
  - Grammar:     LanguageTool error density (inverted)
  - Coherence:   Sentence-level semantic similarity to original (SentenceTransformers)
  - Information:  1 − (fraction of text deleted)
  - Lexical:     Type-token ratio relative to original

All components are normalized to [0, 1]. Multiplicative composition ensures
any axis hitting zero kills the overall score.
"""

import json
from pathlib import Path

import language_tool_python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ── Component Scorers ────────────────────────────────────────────

class QualityScorer:
    """Computes the four quality components and composite Q."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self._lang_tool = None
        self._st_model = None

    @property
    def lang_tool(self):
        if self._lang_tool is None:
            self._lang_tool = language_tool_python.LanguageTool("en-US")
        return self._lang_tool

    @property
    def st_model(self):
        if self._st_model is None:
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model

    def score_grammar(self, text: str) -> float:
        """Grammar quality: 1 − (error_count / word_count). Clamped to [0, 1]."""
        words = text.split()
        if not words:
            return 0.0
        matches = self.lang_tool.check(text)
        error_rate = len(matches) / len(words)
        return max(0.0, min(1.0, 1.0 - error_rate))

    def score_coherence(self, degraded: str, original: str) -> float:
        """Coherence quality: cosine similarity between sentence embeddings
        of degraded vs original text. Measures preservation of local meaning."""
        if not degraded.strip() or not original.strip():
            return 0.0
        emb = self.st_model.encode([degraded, original])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return max(0.0, min(1.0, float(sim)))

    def score_information(self, degraded: str, original: str) -> float:
        """Information preservation: 1 − deletion_fraction.
        Measured by character length ratio."""
        if not original:
            return 0.0
        ratio = len(degraded) / len(original)
        return max(0.0, min(1.0, ratio))

    def score_lexical(self, degraded: str, original: str) -> float:
        """Lexical diversity: type-token ratio of degraded relative to original.
        TTR = unique_words / total_words. Returned as ratio of degraded TTR to
        original TTR, clamped to [0, 1]."""
        def ttr(text: str) -> float:
            words = text.lower().split()
            if not words:
                return 0.0
            return len(set(words)) / len(words)

        ttr_orig = ttr(original)
        if ttr_orig == 0:
            return 0.0
        ttr_deg = ttr(degraded)
        return max(0.0, min(1.0, ttr_deg / ttr_orig))

    def compute_q(self, grammar: float, coherence: float,
                  information: float, lexical: float) -> float:
        """Multiplicative quality function:
        Q = grammar^w₁ × coherence^w₂ × (1−deletion)^w₃ × lexical^w₄

        Note: information score already represents (1 − deletion).
        """
        w = self.weights
        q = (grammar ** w["grammar"]
             * coherence ** w["coherence"]
             * information ** w["information"]
             * lexical ** w["lexical"])
        return q

    def score_sample(self, degraded_text: str, original_text: str) -> dict:
        """Compute all components and composite Q for one sample."""
        g = self.score_grammar(degraded_text)
        c = self.score_coherence(degraded_text, original_text)
        i = self.score_information(degraded_text, original_text)
        l = self.score_lexical(degraded_text, original_text)
        q = self.compute_q(g, c, i, l)

        return {
            "grammar_score": round(g, 4),
            "coherence_score": round(c, 4),
            "information_score": round(i, 4),
            "lexical_score": round(l, 4),
            "Q": round(q, 4),
        }


# ── Batch Processing ────────────────────────────────────────────

def run(config: dict, samples: list[dict]) -> list[dict]:
    """Score all degraded samples. Returns samples with quality scores attached."""
    q_cfg = config["quality"]
    output_dir = Path(q_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "scored_samples.json"
    if output_file.exists():
        print("[Quality] Loading existing scored samples...")
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    scorer = QualityScorer(weights=q_cfg["weights"])
    scored = []

    for sample in tqdm(samples, desc="Computing Q scores"):
        scores = scorer.score_sample(
            sample["degraded_text"], sample["original_text"]
        )
        entry = {**sample, **scores}
        scored.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"[Quality] Saved {len(scored)} scored samples to {output_file}")

    # Cleanup LanguageTool server
    try:
        scorer.lang_tool.close()
    except Exception:
        pass

    return scored
