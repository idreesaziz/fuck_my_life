"""
Scan Wikipedia Featured Articles and find the shortest ones across categories.
Uses the MediaWiki API to get page sizes efficiently (bytes, not full text).
Then we fetch actual text length for the shortest candidates.
"""

import json
import time
from pathlib import Path

import requests
import wikipediaapi

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CondemnedToB/1.0 (research; idrees@example.com)"
})
API_URL = "https://en.wikipedia.org/w/api.php"

wiki = wikipediaapi.Wikipedia(
    user_agent="CondemnedToB/1.0 (research project)",
    language="en",
)


def get_all_fa_titles() -> list[str]:
    """
    Get ALL Featured Article titles by enumerating Category:Featured articles
    using the MediaWiki API with continuation.
    """
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Featured articles",
        "cmlimit": "500",
        "cmtype": "page",
        "cmnamespace": "0",  # Main namespace only
        "format": "json",
    }
    while True:
        resp = SESSION.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for member in data["query"]["categorymembers"]:
            titles.append(member["title"])
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(0.1)
        else:
            break
    return titles


def get_page_sizes(titles: list[str]) -> dict[str, int]:
    """Get page sizes in bytes for batches of titles using the API."""
    sizes = {}
    # API allows up to 50 titles per request
    batch_size = 50
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "info",
            "format": "json",
        }
        resp = SESSION.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for page_id, page_info in data["query"]["pages"].items():
            if int(page_id) < 0:
                continue
            sizes[page_info["title"]] = page_info.get("length", 0)
        time.sleep(0.05)
    return sizes


def get_page_categories(titles: list[str]) -> dict[str, list[str]]:
    """Get categories for batches of titles. We use this to assign topic labels."""
    cats = {}
    batch_size = 50
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "categories",
            "cllimit": "50",
            "clshow": "!hidden",
            "format": "json",
        }
        resp = SESSION.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for page_id, page_info in data["query"]["pages"].items():
            if int(page_id) < 0:
                continue
            cat_names = [
                c["title"].replace("Category:", "")
                for c in page_info.get("categories", [])
            ]
            cats[page_info["title"]] = cat_names
        time.sleep(0.05)
    return cats


# Topic classification based on Wikipedia categories
TOPIC_KEYWORDS = {
    "Biology": ["animal", "bird", "fish", "plant", "species", "insect", "mammal",
                 "reptile", "amphibian", "fungus", "flora", "fauna", "organism",
                 "dinosaur", "dog breed", "cat breed", "horse", "bacteria", "virus",
                 "ecology", "evolution", "anatomy", "botany", "zoology", "biology",
                 "moth", "butterfly", "beetle", "primate", "marine"],
    "History": ["history", "battle", "war ", "ancient", "medieval", "century",
                "dynasty", "empire", "historical", "reign", "conquest"],
    "Geography": ["geography", "island", "mountain", "river", "lake", "city",
                  "town", "country", "region", "province", "state", "county",
                  "continent", "ocean", "volcano", "peninsula"],
    "Science": ["physics", "chemistry", "science", "element", "molecule", "atom",
                "quantum", "thermodynamic", "mineral", "geology", "asteroid",
                "planet", "star ", "comet", "meteor", "moon", "astronomy",
                "mathematics"],
    "Technology": ["technology", "computer", "software", "internet", "spacecraft",
                   "aircraft", "engineering", "rocket", "satellite", "robot",
                   "ship", "locomotive", "automobile", "vehicle"],
    "Arts": ["art", "painting", "sculpture", "architecture", "building",
             "cathedral", "church", "mosque", "temple", "palace", "castle",
             "monument", "museum", "gallery", "architect", "photograph"],
    "Music": ["music", "album", "song", "band", "singer", "composer",
              "symphony", "opera", "concert", "musical"],
    "Literature": ["novel", "book", "poem", "author", "writer", "literary",
                   "fiction", "literature", "comics", "manga", "publication"],
    "Sports": ["sport", "football", "baseball", "basketball", "cricket",
               "tennis", "olympic", "championship", "tournament", "athlete",
               "rugby", "soccer", "racing", "marathon", "boxing", "wrestling",
               "hockey", "golf"],
    "Film & TV": ["film", "movie", "television", "tv series", "anime",
                  "episode", "actor", "actress", "director", "cinema",
                  "documentary"],
    "Video Games": ["video game", "game ", "nintendo", "playstation", "xbox",
                    "gaming"],
    "Politics": ["politic", "president", "election", "government", "democracy",
                 "parliament", "senator", "governor", "prime minister",
                 "constitution", "political"],
    "Military": ["military", "navy", "army", "air force", "warship", "tank",
                 "weapon", "missile", "submarine", "regiment", "brigade",
                 "squadron", "frigate", "destroyer", "battleship", "cruiser",
                 "carrier"],
    "Religion": ["religion", "christian", "islam", "buddhis", "hindu", "jewish",
                 "church", "mosque", "temple", "saint", "pope", "bishop",
                 "monastery", "bible", "quran", "theology"],
    "Medicine": ["medical", "medicine", "disease", "health", "surgery", "drug",
                 "hospital", "symptom", "treatment", "diagnosis", "epidemic",
                 "pandemic", "vaccine"],
    "Transport": ["railway", "railroad", "train", "station", "highway", "road",
                  "bridge", "tunnel", "canal", "port", "airport", "airline",
                  "route", "transit"],
    "Culture": ["culture", "festival", "tradition", "cuisine", "fashion",
                "language", "ethnic", "indigenous", "folk", "heritage",
                "mythology", "legend"],
    "Law": ["law", "legal", "court", "judge", "trial", "supreme court",
            "constitution", "rights", "criminal", "justice"],
    "Education": ["university", "college", "school", "education", "academic",
                  "professor"],
    "Weather": ["hurricane", "typhoon", "cyclone", "tornado", "storm",
                "flood", "drought", "weather", "climate", "meteorolog",
                "tropical"],
    "People": ["biography", "born", "people", "person"],
}


def classify_article(title: str, categories: list[str]) -> str:
    """Assign a topic label based on title and categories."""
    text = (title + " " + " ".join(categories)).lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scores[topic] = score
    if scores:
        return max(scores, key=scores.get)
    return "Other"


def main():
    output_dir = Path("scripts")
    output_dir.mkdir(exist_ok=True)

    cache_file = output_dir / "fa_raw_cache.json"

    # Step 1: Get all FA titles
    if cache_file.exists():
        print("Loading cached FA data...")
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        all_titles = cached["titles"]
        sizes = cached["sizes"]
        print(f"Loaded {len(all_titles)} titles from cache")
    else:
        print("Fetching all Featured Article titles...")
        all_titles = get_all_fa_titles()
        print(f"Found {len(all_titles)} Featured Articles")

        # Step 2: Get page sizes (bytes) for all of them
        print("Fetching page sizes...")
        sizes = get_page_sizes(all_titles)
        print(f"Got sizes for {len(sizes)} pages")

        # Cache results
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"titles": all_titles, "sizes": sizes}, f, ensure_ascii=False)

    # Step 3: Sort by size and get categories for the shortest ~400
    sorted_titles = sorted(sizes.keys(), key=lambda t: sizes[t])
    candidates = sorted_titles[:400]

    print(f"\nFetching categories for {len(candidates)} shortest articles...")
    cats = get_page_categories(candidates)

    # Step 4: Classify and build results
    all_articles = []
    for title in candidates:
        topic = classify_article(title, cats.get(title, []))
        all_articles.append({
            "title": title,
            "category": topic,
            "bytes": sizes[title],
        })

    all_articles.sort(key=lambda x: x["bytes"])

    # Save results
    with open(output_dir / "fa_lengths.json", "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Total FAs: {len(sizes)}")
    print(f"Candidates (shortest 400): {len(all_articles)}")
    print(f"Shortest: {all_articles[0]['title']} ({all_articles[0]['bytes']:,} bytes)")
    print(f"Longest candidate: {all_articles[-1]['title']} ({all_articles[-1]['bytes']:,} bytes)")

    # Show distribution
    brackets = [5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
    print("\nLength distribution (ALL FAs, bytes):")
    all_sizes = sorted(sizes.values())
    prev = 0
    for b in brackets:
        count = sum(1 for s in all_sizes if prev <= s < b)
        print(f"  {prev:>7,} - {b:>7,} bytes: {count} articles")
        prev = b
    count = sum(1 for s in all_sizes if s >= brackets[-1])
    print(f"  {brackets[-1]:>7,}+          bytes: {count} articles")

    # Show category distribution of shortest candidates
    print("\nCategory distribution (shortest 400):")
    cat_counts = {}
    for a in all_articles:
        cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25s}: {count}")

    # Show shortest 250
    print(f"\n\nShortest 250 Featured Articles:")
    print("-" * 80)
    for i, a in enumerate(all_articles[:250]):
        print(f"  {i+1:3d}. [{a['category']:<20s}] {a['title']:<55s} {a['bytes']:>7,} bytes")


if __name__ == "__main__":
    main()
