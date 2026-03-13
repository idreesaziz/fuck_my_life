import json

with open("scripts/fa_lengths.json", encoding="utf-8") as f:
    arts = json.load(f)

cats = {}
for a in arts:
    cats[a["category"]] = cats.get(a["category"], 0) + 1

print("Category distribution (shortest 400):")
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {count}")

print(f"\nShortest: {arts[0]['title']} ({arts[0]['bytes']:,} bytes)")
print(f"Article 150: {arts[149]['title']} ({arts[149]['bytes']:,} bytes)")
print(f"Article 200: {arts[199]['title']} ({arts[199]['bytes']:,} bytes)")

# Count under-represented categories
print("\n\nCategories with fewer than 5 articles (need more diverse picks):")
for cat, count in sorted(cats.items(), key=lambda x: x[1]):
    if count < 5:
        print(f"  {cat}: {count}")
