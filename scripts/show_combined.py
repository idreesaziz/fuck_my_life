"""Show combined degradation: all 4 axes applied to the same text, with every error annotated."""
import json
import random
import difflib
import nltk
from src.degradation import (
    degrade_grammar, degrade_coherence, degrade_information, degrade_lexical
)

with open("data/corpus/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Find an article with richer vocabulary — search for academic words
target_words = {"significant", "prominent", "fundamental", "considerable",
                "numerous", "establish", "substantial", "remarkable",
                "subsequently", "however", "furthermore", "constitute"}

best_article = None
best_score = 0
for art in corpus:
    text_lower = art["text"].lower()
    score = sum(1 for w in target_words if w in text_lower)
    if score > best_score:
        best_score = score
        best_article = art

article = best_article
sents = nltk.sent_tokenize(article["text"])
original = " ".join(sents[:4])  # 4 sentences for richer content

print("ARTICLE:", article["title"])
print()
print("ORIGINAL TEXT:")
print(original)
print()

levels = [0.2, 0.4, 0.6, 0.8]

for level in levels:
    print("=" * 90)
    print(f"ALL AXES COMBINED — LEVEL {level}")
    print("=" * 90)

    seed = 42

    # Apply all 4 axes sequentially
    text = original
    text = degrade_information(text, level, random.Random(seed))
    text = degrade_grammar(text, level, random.Random(seed + 1))
    text = degrade_lexical(text, level, random.Random(seed + 2))
    text = degrade_coherence(text, level, random.Random(seed + 3))
    final = text

    print()
    print("DEGRADED TEXT:")
    print(final)
    print()

    # Word-by-word diff to annotate changes
    orig_words = original.split()
    final_words = final.split()
    
    changes = []
    sm = difflib.SequenceMatcher(None, orig_words, final_words)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            old_chunk = " ".join(orig_words[i1:i2])
            new_chunk = " ".join(final_words[j1:j2])
            changes.append(f'  "{old_chunk}" → "{new_chunk}"')
        elif op == "delete":
            old_chunk = " ".join(orig_words[i1:i2])
            changes.append(f'  "{old_chunk}" → [DELETED]')
        elif op == "insert":
            new_chunk = " ".join(final_words[j1:j2])
            changes.append(f'  [INSERTED] → "{new_chunk}"')

    print("EVERY CHANGE:")
    for c in changes:
        print(c)
    print(f"\n  Word count: {len(orig_words)} → {len(final_words)}")
    print()
