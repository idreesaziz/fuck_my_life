"""
Sanity check: verify that degradation metrics trend in the expected direction
as level increases, for each axis. Uses only lightweight string-based metrics.
"""
import json
from collections import defaultdict

with open("data/degraded/degraded_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Group by (axis, level)
groups = defaultdict(list)
for s in samples:
    groups[(s["axis"], s["level"])].append(s)

axes = ["grammar", "coherence", "information", "lexical"]
levels = [0.0, 0.2, 0.4, 0.6, 0.8]


def char_ratio(s):
    """len(degraded) / len(original)"""
    orig = len(s["original_text"])
    return len(s["degraded_text"]) / orig if orig else 1.0


def word_ratio(s):
    """word count ratio"""
    orig = len(s["original_text"].split())
    return len(s["degraded_text"].split()) / orig if orig else 1.0


def ttr(text):
    """type-token ratio"""
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0


def ttr_ratio(s):
    """TTR of degraded / TTR of original"""
    t_orig = ttr(s["original_text"])
    return ttr(s["degraded_text"]) / t_orig if t_orig else 0.0


def edit_distance_ratio(s):
    """Rough char-level difference: chars changed / total chars.
    Uses set difference of character bigrams as a fast proxy."""
    def bigrams(t):
        return set(t[i:i+2] for i in range(len(t)-1))
    b_orig = bigrams(s["original_text"])
    b_deg = bigrams(s["degraded_text"])
    if not b_orig:
        return 0.0
    # Jaccard distance
    intersection = b_orig & b_deg
    union = b_orig | b_deg
    return 1.0 - len(intersection) / len(union) if union else 0.0


def sentence_count_ratio(s):
    """Number of sentences in degraded vs original (proxy for information deletion)."""
    import re
    def count_sents(t):
        return len(re.split(r'[.!?]+', t.strip()))
    orig = count_sents(s["original_text"])
    return count_sents(s["degraded_text"]) / orig if orig else 1.0


# ── Print results ──
print("=" * 85)
print("SANITY CHECK — Degradation Metrics by Axis × Level")
print("=" * 85)

for axis in axes:
    print(f"\n{'─' * 85}")
    print(f"  AXIS: {axis.upper()}")
    print(f"{'─' * 85}")

    if axis == "grammar":
        print(f"  {'Level':>6}  {'Bigram Dist':>12}  {'Char Ratio':>11}  {'Word Ratio':>11}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*11}  {'─'*11}")
        for lvl in levels:
            g = groups[(axis, lvl)]
            bd = sum(edit_distance_ratio(s) for s in g) / len(g)
            cr = sum(char_ratio(s) for s in g) / len(g)
            wr = sum(word_ratio(s) for s in g) / len(g)
            print(f"  {lvl:>6.1f}  {bd:>12.4f}  {cr:>11.4f}  {wr:>11.4f}")
        print(f"  Expected: bigram dist ↑, char/word ratio ≈ stable")

    elif axis == "coherence":
        print(f"  {'Level':>6}  {'Bigram Dist':>12}  {'Char Ratio':>11}  {'Word Ratio':>11}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*11}  {'─'*11}")
        for lvl in levels:
            g = groups[(axis, lvl)]
            bd = sum(edit_distance_ratio(s) for s in g) / len(g)
            cr = sum(char_ratio(s) for s in g) / len(g)
            wr = sum(word_ratio(s) for s in g) / len(g)
            print(f"  {lvl:>6.1f}  {bd:>12.4f}  {cr:>11.4f}  {wr:>11.4f}")
        print(f"  Expected: bigram dist ↑ (shuffled), char/word ratio = 1.0 (no deletion)")

    elif axis == "information":
        print(f"  {'Level':>6}  {'Char Ratio':>11}  {'Word Ratio':>11}  {'Sent Ratio':>11}")
        print(f"  {'─'*6}  {'─'*11}  {'─'*11}  {'─'*11}")
        for lvl in levels:
            g = groups[(axis, lvl)]
            cr = sum(char_ratio(s) for s in g) / len(g)
            wr = sum(word_ratio(s) for s in g) / len(g)
            sr = sum(sentence_count_ratio(s) for s in g) / len(g)
            print(f"  {lvl:>6.1f}  {cr:>11.4f}  {wr:>11.4f}  {sr:>11.4f}")
        print(f"  Expected: all ratios ↓ (text getting shorter)")

    elif axis == "lexical":
        print(f"  {'Level':>6}  {'TTR Ratio':>10}  {'Bigram Dist':>12}  {'Word Ratio':>11}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*11}")
        for lvl in levels:
            g = groups[(axis, lvl)]
            tr = sum(ttr_ratio(s) for s in g) / len(g)
            bd = sum(edit_distance_ratio(s) for s in g) / len(g)
            wr = sum(word_ratio(s) for s in g) / len(g)
            print(f"  {lvl:>6.1f}  {tr:>10.4f}  {bd:>12.4f}  {wr:>11.4f}")
        print(f"  Expected: TTR ratio ↓ (less diverse), word ratio ≈ 1.0")


# ── Show example snippets ──
print(f"\n\n{'=' * 85}")
print("EXAMPLE SNIPPETS (first 300 chars of degraded text)")
print("=" * 85)

# Pick one article 
example_title = samples[0]["source_title"]
for axis in axes:
    print(f"\n{'─' * 85}")
    print(f"  AXIS: {axis.upper()}  |  Article: {example_title}")
    print(f"{'─' * 85}")
    for lvl in [0.0, 0.4, 0.8]:
        matching = [s for s in samples 
                    if s["source_title"] == example_title 
                    and s["axis"] == axis 
                    and s["level"] == lvl 
                    and s["repetition"] == 0]
        if matching:
            snippet = matching[0]["degraded_text"][:300]
            print(f"\n  Level {lvl}:")
            print(f"    {snippet}...")
