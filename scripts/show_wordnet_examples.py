"""Show lexical collapse examples — WordNet + wordfreq."""
import json, random, difflib, nltk
from src.degradation import degrade_lexical

with open("data/corpus/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

for idx in [0, 30, 60, 80]:
    art = corpus[idx]
    sents = nltk.sent_tokenize(art["text"])
    text = " ".join(sents[:5])

    print(f"=== {art['title']} ===")
    print(f"ORIGINAL:\n{text[:500]}")
    print()

    for level in [0.4, 0.8]:
        result = degrade_lexical(text, level, random.Random(42))

        orig_words = text.split()
        new_words = result.split()
        sm = difflib.SequenceMatcher(None, orig_words, new_words)
        changes = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                o = " ".join(orig_words[i1:i2])
                n = " ".join(new_words[j1:j2])
                if o.replace("'", " '") != n:  # skip tokenization artifacts
                    changes.append(f"  {o}  -->  {n}")

        print(f"LEVEL {level}: {len(changes)} changes")
        for c in changes[:20]:
            print(c)
        print()

    print("=" * 70)
    print()
