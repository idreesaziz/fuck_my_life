"""Generate the updated config.yaml with new short articles."""
import json
import yaml

with open("scripts/selected_150_final.json", encoding="utf-8") as f:
    articles = json.load(f)

# Sort by category then title
articles.sort(key=lambda x: (x["category"], x["title"]))

# Build article entries
article_entries = []
for a in articles:
    article_entries.append({
        "title": a["title"],
        "category": a["category"],
    })

config = {
    "corpus": {
        "max_chars": 0,
        "output_dir": "data/corpus",
        "articles": article_entries,
    },
    "degradation": {
        "levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "samples_per_level": 3,
        "axes": {
            "grammar": {"enabled": True},
            "coherence": {"enabled": True},
            "information": {"enabled": True},
            "lexical": {"enabled": True},
        },
        "output_dir": "data/degraded",
    },
    "quality": {
        "weights": {
            "grammar": 1.0,
            "coherence": 1.0,
            "information": 1.0,
            "lexical": 1.0,
        },
        "output_dir": "data/scores",
    },
    "llm_scoring": {
        "models": [
            {"name": "gpt-4.1-nano", "provider": "openai", "model_id": "gpt-4.1-nano"},
            {"name": "gemini-2.5-flash-lite", "provider": "google", "model_id": "gemini-2.5-flash-lite"},
        ],
        "conditions": ["isolated", "batched"],
        "scale_min": 1,
        "scale_max": 9,
        "repetitions": 1,
        "api_key_rotation": {
            "google": [{"env": "GOOGLE_API_KEY_1"}, {"env": "GOOGLE_API_KEY_2"}],
            "openai": [{"env": "OPENAI_API_KEY"}],
        },
        "output_dir": "data/llm_scores",
    },
    "analysis": {
        "output_dir": "results",
        "figures_dir": "results/figures",
    },
}

# Custom YAML dumper to handle special chars in titles
class QuotedStr(str):
    pass

def quoted_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

yaml.add_representer(QuotedStr, quoted_str_representer)

# Quote article titles/categories
for entry in config["corpus"]["articles"]:
    entry["title"] = QuotedStr(entry["title"])
    entry["category"] = QuotedStr(entry["category"])

header = """# ============================================================
# Condemned to B — Configuration
# ============================================================
# 150 shortest Wikipedia Featured Articles across 16 categories
# Size range: ~7.5K - 22.5K bytes (full text, no truncation)
# ============================================================

"""

with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(header)
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

print("config.yaml updated successfully!")
print(f"Articles: {len(article_entries)}")
cats = {}
for a in article_entries:
    cats[a["category"]] = cats.get(a["category"], 0) + 1
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")
