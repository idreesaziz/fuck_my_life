"""
Select 150 short Featured Articles with good category diversity.
Improved classifier + diversity-aware greedy selection.
"""

import json
import re
import requests
import time

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CondemnedToB/1.0 (research)"})
API_URL = "https://en.wikipedia.org/w/api.php"

# Load cached data
with open("scripts/fa_raw_cache.json", encoding="utf-8") as f:
    cache = json.load(f)

sizes = cache["sizes"]

# Sort all articles by size
sorted_articles = sorted(sizes.items(), key=lambda x: x[1])

# Take the shortest 500 as candidates
candidates = sorted_articles[:500]
candidate_titles = [t for t, s in candidates]

print(f"Fetching categories for {len(candidate_titles)} shortest articles...")

# Fetch categories in batches
all_cats = {}
batch_size = 50
for i in range(0, len(candidate_titles), batch_size):
    batch = candidate_titles[i:i + batch_size]
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
        all_cats[pinfo["title"]] = cat_names
    time.sleep(0.05)

print(f"Got categories for {len(all_cats)} articles")


# -- Improved classifier --
def classify(title: str, categories: list[str]) -> str:
    t = title.lower()
    c = " ".join(categories).lower()
    text = t + " " + c

    # Latin binomials / species names
    if re.match(r'^[A-Z][a-z]+ [a-z]+$', title):
        return "Biology"

    # Explicit category checks first (most reliable)
    cat_rules = [
        ("Biology", ["mammal", "bird", "fish", "insect", "reptile", "amphibian",
                      "flora", "fauna", "fungus", "fungi", "species", "rodent",
                      "animal", "moth", "butterfly", "primate", "bat", "endemic",
                      "iucn", "organism", "mouse", "rat", "dove", "swallow",
                      "shrew", "lemur", "zoolog", "botan", "ecolog", "taxa named",
                      "plants described", "mammals described", "birds described",
                      "animals described"]),
        ("Military", ["warship", "battleship", "destroyer", "frigate", "cruiser",
                       "submarine", "aircraft carrier", "torpedo", "navy",
                       "military", "regiment", "brigade", "battalion", "squadron",
                       "victoria cross", "medal of honor", "war memorial",
                       "conflicts in", "battles of", "wars of", "military history",
                       "naval battles", "gunboat", "corvette", "sloop"]),
        ("Weather", ["hurricane", "typhoon", "cyclone", "tropical storm",
                      "tropical depression", "atlantic hurricane", "pacific hurricane",
                      "effects of hurricane"]),
        ("History", ["century in", "history of", "historical", "dynasty",
                      "ancient", "medieval", "byzantine", "roman empire",
                      "anglo-saxon", "viking", "ottoman", "colonial"]),
        ("Film & TV", ["film", "movie", "television", "anime", "episode",
                        "cinema", "indonesian film", "silent film", "horror film",
                        "comedy film", "drama film", "documentary film"]),
        ("Geography", ["geography", "island", "mountain", "river", "route",
                        "state route", "interstate", "highway", "road",
                        "national park", "county", "province", "peninsula"]),
        ("Music", ["album", "song", "single", "discograph", "musician",
                    "singer", "band", "composer", "symphony", "concert",
                    "record chart"]),
        ("Literature", ["novel", "book", "poem", "magazine", "pulp magazine",
                         "fiction magazine", "literary", "publication", "writer",
                         "comic", "manga"]),
        ("Sports", ["sport", "football", "baseball", "basketball", "cricket",
                     "tennis", "olympic", "championship", "tournament",
                     "rugby", "racing", "boxing", "horse racing", "jockey",
                     "thoroughbred", "racehorse", "stakes", "stallion", "gelding",
                     "filly", "mare"]),
        ("Science", ["physics", "chemistry", "element", "molecule", "mineral",
                      "geology", "asteroid", "planet", "star", "meteorite",
                      "mathematics", "theorem", "equation"]),
        ("Arts", ["painting", "sculpture", "architecture", "portrait",
                   "cathedral", "church", "mosque", "temple", "palace",
                   "castle", "monument", "museum", "gallery", "fresco",
                   "artwork", "art movement"]),
        ("Transport", ["railway", "railroad", "locomotive", "station",
                        "bridge", "tunnel", "canal", "tramway", "tram",
                        "transit", "airline", "airport"]),
        ("Video Games", ["video game", "game boy", "nintendo", "playstation",
                          "xbox", "sega", "atari"]),
        ("Religion", ["saint", "bishop", "pope", "monastery", "abbess", "abbot",
                       "diocese", "archbishop", "religious", "theology",
                       "christian", "islam", "buddhis", "hindu"]),
        ("Politics", ["president", "election", "governor", "senator",
                       "parliament", "political", "prime minister",
                       "constitution"]),
        ("Technology", ["spacecraft", "satellite", "rocket", "computer",
                         "software", "engineering", "ship", "vessel"]),
        ("Medicine", ["disease", "medical", "health", "surgery", "hospital",
                       "treatment", "diagnosis", "epidemic", "vaccine",
                       "syndrome"]),
        ("Law", ["legal", "court", "judge", "trial", "rights", "criminal",
                  "justice", "siege", "law"]),
        ("Culture", ["culture", "festival", "tradition", "cuisine",
                      "mythology", "legend", "folk", "heritage"]),
        ("Education", ["university", "college", "school", "education",
                        "academic"]),
    ]

    for topic, keywords in cat_rules:
        for kw in keywords:
            if kw in text:
                return topic

    # Title-based patterns
    title_patterns = [
        (r'\bUSS\b|\bCSS\b|\bHMS\b|\bSMS\b', "Military"),
        (r'battleship|destroyer|frigate', "Military"),
        (r'Action of \d', "Military"),
        (r'Battle of', "Military"),
        (r'War Memorial', "Military"),
        (r'hurricane|typhoon|cyclone|tropical storm', "Weather"),
        (r'State Route|Interstate \d', "Geography"),
        (r'\(\d{4} film\)', "Film & TV"),
        (r'magazine\)$', "Literature"),
    ]
    for pattern, topic in title_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return topic

    return "Other"


# Classify all candidates
classified = []
for title, size in candidates:
    cats = all_cats.get(title, [])
    topic = classify(title, cats)
    classified.append({
        "title": title,
        "category": topic,
        "bytes": size,
    })

# Show improved distribution
cat_counts = {}
for a in classified:
    cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1
print("\nImproved category distribution (500 candidates):")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {cnt}")

# -- Diversity-aware selection --
# Goal: pick 150 articles, trying to get at least 5 per major category
TARGET = 150

# We want categories with at least some representation
# Sort classified by size
classified.sort(key=lambda x: x["bytes"])

# Greedy: round-robin by category, picking shortest available from each
# First, organize by category
by_cat = {}
for a in classified:
    by_cat.setdefault(a["category"], []).append(a)

# Exclude "Other" from priority round-robin (fill later)
priority_cats = [c for c in by_cat if c != "Other"]

selected = []
selected_titles = set()

# Phase 1: Take up to 8 per category (round-robin, shortest first)
for rnd in range(8):
    for cat in priority_cats:
        pool = by_cat[cat]
        for a in pool:
            if a["title"] not in selected_titles:
                selected.append(a)
                selected_titles.add(a["title"])
                break
        if len(selected) >= TARGET:
            break
    if len(selected) >= TARGET:
        break

print(f"\nAfter phase 1 (round-robin): {len(selected)} selected")

# Phase 2: Fill remaining slots with shortest unselected articles
remaining = [a for a in classified if a["title"] not in selected_titles]
remaining.sort(key=lambda x: x["bytes"])
for a in remaining:
    if len(selected) >= TARGET:
        break
    selected.append(a)
    selected_titles.add(a["title"])

print(f"After phase 2 (fill): {len(selected)} selected")

# Sort final selection by size
selected.sort(key=lambda x: x["bytes"])

# Final category distribution
final_cats = {}
for a in selected:
    final_cats[a["category"]] = final_cats.get(a["category"], 0) + 1

print(f"\nFinal selection: {len(selected)} articles")
print(f"Size range: {selected[0]['bytes']:,} - {selected[-1]['bytes']:,} bytes")
print(f"\nFinal category distribution:")
for cat, cnt in sorted(final_cats.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {cnt}")

# Save selection
with open("scripts/selected_150.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print(f"\nSaved to scripts/selected_150.json")

# Print full list
print(f"\n{'='*80}")
print(f"Selected 150 articles:")
print(f"{'='*80}")
for i, a in enumerate(selected):
    print(f"  {i+1:3d}. [{a['category']:<20s}] {a['title']:<55s} {a['bytes']:>7,}")
