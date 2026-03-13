"""Quick test: check lexical dose-response using TTR (alignment-independent)."""
import json, random, nltk
from src.degradation import degrade_lexical, _WORDNET_SKIP, _WORDNET_POS_MAP

with open("data/corpus/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

def ttr(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0

def unique_word_count(text):
    return len(set(text.lower().split()))

print(f"{'Article':40} {'CW':>4}  {'Orig':>6}  {'0.2':>6}  {'0.4':>6}  {'0.6':>6}  {'0.8':>6}")
print(f"{'':40} {'':>4}  {'TTR':>6}  {'TTR':>6}  {'TTR':>6}  {'TTR':>6}  {'TTR':>6}")
print("-" * 82)

allowed = {"NN","NNS","VB","VBD","VBG","VBN","VBP","VBZ","JJ","JJR","JJS"}

ttr_drops = {0.2: [], 0.4: [], 0.6: [], 0.8: []}
for art in corpus[:15]:
    text = art["text"]
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    cw = len([1 for w, t in tagged
              if t in allowed and len(w) >= 4
              and w.isalpha() and w.lower() not in _WORDNET_SKIP])

    orig_ttr = ttr(text)
    ttrs = []
    for lvl in [0.2, 0.4, 0.6, 0.8]:
        deg = degrade_lexical(text, lvl, random.Random(42))
        t = ttr(deg)
        ttrs.append(t)
        ttr_drops[lvl].append(orig_ttr - t)

    title = art["title"][:40]
    print(f"{title:40} {cw:>4}  {orig_ttr:.4f}  {ttrs[0]:.4f}  {ttrs[1]:.4f}  {ttrs[2]:.4f}  {ttrs[3]:.4f}")

# Also show unique word reduction
print(f"\n{'Article':40} {'Orig':>6}  {'0.2':>6}  {'0.4':>6}  {'0.6':>6}  {'0.8':>6}")
print(f"{'':40} {'uniq':>6}  {'uniq':>6}  {'uniq':>6}  {'uniq':>6}  {'uniq':>6}")
print("-" * 72)

for art in corpus[:15]:
    text = art["text"]
    orig_u = unique_word_count(text)
    uniqs = []
    for lvl in [0.2, 0.4, 0.6, 0.8]:
        deg = degrade_lexical(text, lvl, random.Random(42))
        uniqs.append(unique_word_count(deg))

    title = art["title"][:40]
    print(f"{title:40} {orig_u:>6}  {uniqs[0]:>6}  {uniqs[1]:>6}  {uniqs[2]:>6}  {uniqs[3]:>6}")

print(f"\nAvg TTR drop by level:")
for lvl in [0.2, 0.4, 0.6, 0.8]:
    avg_drop = sum(ttr_drops[lvl]) / len(ttr_drops[lvl])
    print(f"  Level {lvl}: -{avg_drop:.4f}")
