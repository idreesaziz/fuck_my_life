import json

with open("scripts/fa_lengths.json", encoding="utf-8") as f:
    arts = json.load(f)

# Show "Other" articles with their Wikipedia categories
with open("scripts/fa_raw_cache.json", encoding="utf-8") as f:
    cache = json.load(f)

import requests, time
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CondemnedToB/1.0 (research)"})
API_URL = "https://en.wikipedia.org/w/api.php"

others = [a for a in arts if a["category"] == "Other"][:50]
titles = [a["title"] for a in others]

# Fetch categories for these
batch_size = 50
cats_map = {}
for i in range(0, len(titles), batch_size):
    batch = titles[i:i+batch_size]
    params = {
        "action": "query", "titles": "|".join(batch),
        "prop": "categories", "cllimit": "50", "clshow": "!hidden", "format": "json",
    }
    resp = SESSION.get(API_URL, params=params, timeout=30)
    data = resp.json()
    for pid, pinfo in data["query"]["pages"].items():
        if int(pid) < 0:
            continue
        cat_names = [c["title"].replace("Category:", "") for c in pinfo.get("categories", [])]
        cats_map[pinfo["title"]] = cat_names

print("First 50 'Other' articles and their categories:")
print("-" * 80)
for a in others:
    cats = cats_map.get(a["title"], [])
    print(f"\n{a['title']} ({a['bytes']:,} bytes)")
    for c in cats[:8]:
        print(f"  - {c}")
