"""Show side-by-side degradation examples for each axis and level."""
import json
import nltk

with open("data/degraded/degraded_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

title = samples[0]["source_title"]
orig = samples[0]["original_text"]
sents = nltk.sent_tokenize(orig)
passage = " ".join(sents[:3])

print(f"ARTICLE: {title}")
print(f"PASSAGE: first 3 sentences ({len(passage)} chars)")
print()
print("ORIGINAL:")
print(passage)

for axis in ["grammar", "coherence", "information", "lexical"]:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"AXIS: {axis.upper()}")
    print(sep)

    axis_samples = [
        s for s in samples
        if s["axis"] == axis
        and s["source_title"] == title
        and s["repetition"] == 0
    ]

    for s in axis_samples:
        lv = s["level"]
        if lv == 0.0:
            continue

        deg_sents = nltk.sent_tokenize(s["degraded_text"])
        if axis == "coherence":
            deg_passage = " ".join(deg_sents[:5])
        else:
            deg_passage = " ".join(deg_sents[:3])

        print(f"\n--- Level {lv} ---")
        print(deg_passage)

print(f"\n{'=' * 90}")
print("SECOND ARTICLE EXAMPLES (for variety)")
print("=" * 90)

# Find a different article
other_title = None
for s in samples:
    if s["source_title"] != title:
        other_title = s["source_title"]
        break

other_orig = [s for s in samples if s["source_title"] == other_title][0]["original_text"]
other_sents = nltk.sent_tokenize(other_orig)
other_passage = " ".join(other_sents[:3])

print(f"\nARTICLE: {other_title}")
print(f"\nORIGINAL:")
print(other_passage)

for axis in ["grammar", "coherence", "information", "lexical"]:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"AXIS: {axis.upper()}")
    print(sep)

    axis_samples = [
        s for s in samples
        if s["axis"] == axis
        and s["source_title"] == other_title
        and s["repetition"] == 0
    ]

    for s in axis_samples:
        lv = s["level"]
        if lv == 0.0:
            continue

        deg_sents = nltk.sent_tokenize(s["degraded_text"])
        if axis == "coherence":
            deg_passage = " ".join(deg_sents[:5])
        else:
            deg_passage = " ".join(deg_sents[:3])

        print(f"\n--- Level {lv} ---")
        print(deg_passage)
