"""
Reclassify the selected 150 articles with a fixed classifier,
then generate the config.yaml articles section.
"""
import json
import re
import requests
import time

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CondemnedToB/1.0 (research)"})
API_URL = "https://en.wikipedia.org/w/api.php"

with open("scripts/selected_150.json", encoding="utf-8") as f:
    selected = json.load(f)

titles = [a["title"] for a in selected]

# Fetch categories for all
all_cats = {}
batch_size = 50
for i in range(0, len(titles), batch_size):
    batch = titles[i:i + batch_size]
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


def classify(title: str, categories: list[str]) -> str:
    t = title.lower()
    c_text = " ".join(categories).lower()

    # -- Title-based rules (highest priority, most reliable) --

    # Military: ship names with prefixes
    if re.match(r'^(USS|CSS|HMS|SMS|HMHS|HMAS|HMCS|INS|KMS) ', title):
        return "Military"
    # Military: explicit patterns
    if re.search(r'Action of \d|Battle of|War Memorial|battleship|destroyer|frigate|torpedo boat|gunboat|class battleship', title, re.I):
        return "Military"
    # Military: regiment/squadron etc
    if re.search(r'Regiment|Squadron|Brigade|Battalion', title):
        return "Military"
    # Military: operations (military)
    if re.search(r'^Operation \w', title) and 'military' in c_text:
        return "Military"

    # Weather
    if re.search(r'Hurricane|Typhoon|Cyclone|Tropical Storm|Tropical Depression|subtropical storm', title, re.I):
        return "Weather"

    # Film patterns
    if re.search(r'\(\d{4} film\)|\(\d{4}s? film\)', title):
        return "Film & TV"

    # Geography: route/highway
    if re.search(r'State Route|Interstate \d|U\.S\. Route|North Road|Highway', title, re.I):
        return "Geography"

    # -- Category-based rules (ordered from most specific to least) --

    # Military from categories
    mil_kws = ["warship", "navy", "military history", "naval battle", "victoria cross",
               "medal of honor", "war memorial", "conflicts in", "battles of",
               "wars involving", "military award", "military operation"]
    if any(kw in c_text for kw in mil_kws):
        return "Military"

    # Weather from categories
    if any(kw in c_text for kw in ["hurricane", "typhoon", "cyclone", "tropical storm"]):
        return "Weather"

    # Sports from categories (before biology to catch horse racing)
    sport_kws = ["horse racing", "racehorse", "thoroughbred", "jockey", "stallion",
                 "gelding", "filly", "stakes winner", "sport", "football", "baseball",
                 "basketball", "cricket", "tennis", "olympic", "championship",
                 "rugby", "boxing", "wrestling", "athlete", "racing"]
    if any(kw in c_text for kw in sport_kws):
        return "Sports"

    # Film/TV from categories
    film_kws = ["film", "cinema", "movie", "television", "anime",
                "silent short", "lost film", "indonesian film"]
    if any(kw in c_text for kw in film_kws):
        return "Film & TV"

    # Biology from categories (specific terms only)
    bio_kws = ["mammals", "birds", "fish", "insect", "reptile", "amphibian",
               "rodent", "primate", "lemur", "carnivora", "chiroptera",
               "passerine", "parrot", "raptor", "dove", "pigeon", "swallow",
               "shrew", "bat species", "flora of", "fauna of", "fungi",
               "fungus", "mushroom", "plant", "species described",
               "iucn red list", "endemic", "zoolog", "taxon",
               "taxa named", "vertebrate", "invertebrate", "moth", "butterfly",
               "beetle", "marine animal"]
    if any(kw in c_text for kw in bio_kws):
        return "Biology"
    # Latin binomial title = biology
    if re.match(r'^[A-Z][a-z]+ [a-z]+$', title):
        return "Biology"

    # Literature from categories
    lit_kws = ["magazine", "pulp magazine", "fiction magazine", "novel",
               "book", "literary", "publication", "comic", "manga",
               "writer", "author", "poet"]
    if any(kw in c_text for kw in lit_kws):
        return "Literature"

    # Music
    music_kws = ["album", "song", "single", "discograph", "musician",
                 "singer", "band", "composer", "symphony", "concert",
                 "record chart", "music"]
    if any(kw in c_text for kw in music_kws):
        return "Music"

    # Arts
    art_kws = ["painting", "sculpture", "portrait", "artwork", "fresco",
               "architecture", "architect", "gallery", "museum",
               "stone circle", "stone monument", "megalith"]
    if any(kw in c_text for kw in art_kws):
        return "Arts"

    # History
    hist_kws = ["century", "ancient", "medieval", "byzantine", "roman empire",
                "anglo-saxon", "viking", "ottoman", "colonial", "dynasty",
                "historical", "history of", "monarch", "emperor", "empress",
                "king of", "queen of", "duke of", "earl of"]
    if any(kw in c_text for kw in hist_kws):
        return "History"

    # Geography
    geo_kws = ["geography", "island", "mountain", "river", "national park",
               "county", "province", "peninsula", "landform", "earthquake",
               "cape ", "lake ", "valley"]
    if any(kw in c_text for kw in geo_kws):
        return "Geography"

    # Transport
    trans_kws = ["railway", "railroad", "locomotive", "tramway", "tram",
                 "transit", "airline", "airport", "bridge", "tunnel",
                 "canal", "turnpike"]
    if any(kw in c_text for kw in trans_kws):
        return "Transport"

    # Video Games
    if any(kw in c_text for kw in ["video game", "game boy", "nintendo", "playstation"]):
        return "Video Games"

    # Religion
    rel_kws = ["pope", "bishop", "saint", "monastery", "abbey", "diocese",
               "archbishop", "religious", "theology", "church history",
               "christian", "islam", "buddhis", "hinduism", "catholic"]
    if any(kw in c_text for kw in rel_kws):
        return "Religion"

    # Science
    sci_kws = ["physics", "chemistry", "element", "mineral", "geology",
               "asteroid", "planet", "star ", "mathematics",
               "theorem", "meteorite", "fossil", "paleontolog"]
    if any(kw in c_text for kw in sci_kws):
        return "Science"

    # Technology
    tech_kws = ["spacecraft", "satellite", "rocket", "computer", "software",
                "engineering", "invention"]
    if any(kw in c_text for kw in tech_kws):
        return "Technology"

    # Politics
    pol_kws = ["president", "election", "governor", "senator", "parliament",
               "political", "prime minister", "constitution", "government"]
    if any(kw in c_text for kw in pol_kws):
        return "Politics"

    # Medicine
    med_kws = ["disease", "medical", "health", "surgery", "hospital",
               "treatment", "syndrome", "vaccine"]
    if any(kw in c_text for kw in med_kws):
        return "Medicine"

    # Law
    law_kws = ["legal", "court", "judge", "trial", "rights", "criminal",
               "justice", "siege", "law enforcement"]
    if any(kw in c_text for kw in law_kws):
        return "Law"

    # Culture
    cult_kws = ["culture", "festival", "tradition", "cuisine", "mythology",
                "legend", "folklore", "folk", "heritage", "hoax", "prank"]
    if any(kw in c_text for kw in cult_kws):
        return "Culture"

    # Education
    if any(kw in c_text for kw in ["university", "college", "school", "academic"]):
        return "Education"

    # Last resort: check title for common patterns
    if re.search(r'ditch|mound|hill|viaduct|fort\b', t):
        return "Geography"
    if re.search(r'covenant|treaty', t):
        return "History"

    return "Other"


# Reclassify
for a in selected:
    cats = all_cats.get(a["title"], [])
    a["category"] = classify(a["title"], cats)

# Category distribution
cat_counts = {}
for a in selected:
    cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1

print("Reclassified category distribution:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {cnt}")

other_articles = [a for a in selected if a["category"] == "Other"]
print(f"\n'Other' articles ({len(other_articles)}):")
for a in other_articles:
    cats = all_cats.get(a["title"], [])
    print(f"  {a['title']}")
    for c in cats[:5]:
        print(f"    - {c}")

# Save updated
with open("scripts/selected_150.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

# Generate config entries
print("\n\n# CONFIG YAML ENTRIES:")
print("articles:")
for a in selected:
    safe_title = a["title"].replace('"', '\\"')
    print(f'  - title: "{safe_title}"')
    print(f'    category: "{a["category"]}"')
