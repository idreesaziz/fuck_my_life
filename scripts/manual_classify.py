"""
Manual category mapping for the 150 selected short FAs.
Then generate config.yaml article entries.
"""
import json

with open("scripts/selected_150.json", encoding="utf-8") as f:
    articles = json.load(f)

# Manual overrides for every article
MANUAL = {
    "2005 Azores subtropical storm": "Weather",
    "Miss Meyers": "Film & TV",
    "Constantine (son of Theophilos)": "History",
    "Nico Ditch": "Geography",
    "Hurricane Irene (2005)": "Weather",
    "How Brown Saw the Baseball Game": "Film & TV",
    "Operation Hardboiled": "Military",
    "Si Ronda": "Film & TV",
    "Tropical Storm Brenda (1960)": "Weather",
    "Hurricane Daniel (2006)": "Weather",
    "Petter's big-footed mouse": "Biology",
    "Fir Clump Stone Circle": "Arts",
    "Si Tjonat": "Film & TV",
    "Pope Sisinnius": "Religion",
    "Mini scule": "Biology",
    "Ignace Tonené": "History",
    "North Road, Manchester": "Geography",
    "Tales of Wonder (magazine)": "Literature",
    "Katsudō Shashin": "Film & TV",
    "Pipistrellus raceyi": "Biology",
    "Meurig ab Arthfael": "History",
    "Operation Copperhead": "Military",
    "Ashcan comic": "Literature",
    "Scoops (magazine)": "Literature",
    "Lost Luggage (video game)": "Video Games",
    "Grass Fight": "History",
    "She Shoulda Said No!": "Film & TV",
    "Lavanify": "Biology",
    "Ghost Stories (magazine)": "Literature",
    "USS Constellation vs La Vengeance": "Military",
    "Gagak Item": "Film & TV",
    "Miniopterus aelleni": "Biology",
    "Fort Southerland": "Military",
    "Amazing Stories Quarterly": "Literature",
    "Design A-150 battleship": "Military",
    "Ælfwynn, wife of Æthelstan Half-King": "History",
    "Lightning Bar": "Sports",
    "Irish Thoroughbred": "Sports",
    "Action of 1 August 1801": "Military",
    "Hurricane Grace (1991)": "Weather",
    "U.S. Route 45 in Michigan": "Geography",
    "Myriostoma": "Biology",
    "Imaginative Tales": "Literature",
    "Durrell's vontsira": "Biology",
    "Pah Wongso Pendekar Boediman": "Film & TV",
    "Lyon-class battleship": "Military",
    "Thomasomys ucucha": "Biology",
    "Soeara Berbisa": "Film & TV",
    "2007–2008 Nazko earthquakes": "Science",
    "Manchester Mummy": "Culture",
    "Science Fiction Adventures (1956 magazine)": "Literature",
    "Martinus (son of Heraclius)": "History",
    "Elizabeth Needham": "History",
    "Science Fiction Monthly": "Literature",
    "Euryoryzomys emmonsae": "Biology",
    "Snoring rail": "Biology",
    "Saturn (magazine)": "Literature",
    "Al-Altan": "History",
    "Fantastic Novels": "Literature",
    "Barbara L": "Sports",
    "Berners Street hoax": "Culture",
    "Uncanny Tales (Canadian pulp magazine)": "Literature",
    "Victoria Cross for New Zealand": "Military",
    "Interstate 296": "Geography",
    "4th Missouri Infantry Regiment (Confederate)": "Military",
    "Northolt siege": "Law",
    "Action of 1 January 1800": "Military",
    "Christopher Lekapenos": "History",
    "Thomcord": "Biology",
    "Brochfael ap Meurig": "History",
    "Inocybe saliceticola": "Biology",
    "Marvel Science Stories": "Literature",
    "Xeromphalina setulipes": "Biology",
    "Tjioeng Wanara": "Film & TV",
    "Battle Birds": "Literature",
    "Galton Bridge": "Transport",
    "Science-Fiction Plus": "Literature",
    "Seorsumuscardinus": "Biology",
    "Saline Valley salt tram": "Transport",
    "Heptamegacanthus": "Biology",
    "Madman's Drum": "Literature",
    "The Carpet from Bagdad": "Literature",
    "Abuwtiyuw": "Culture",
    "D-Day naval deceptions": "Military",
    "This Dust Was Once the Man": "Literature",
    "Virgin and Child Enthroned": "Arts",
    "Prince Alfred of Great Britain": "History",
    "Malagasy mountain mouse": "Biology",
    "Oriental Stories": "Literature",
    "Fantasy Book": "Literature",
    "White-headed fruit dove": "Biology",
    "Yugoslav gunboat Beli Orao": "Military",
    "Luo Yixiu": "History",
    "New York State Route 373": "Geography",
    "Dirty Dick": "Culture",
    "Fragment of a Crucifixion": "Arts",
    "Miniopterus griveaudi": "Biology",
    "CSS Baltic": "Military",
    "Ancestry of the Godwins": "History",
    "Wolverton Viaduct": "Transport",
    "Super-Science Fiction": "Literature",
    "Cloud (video game)": "Video Games",
    "Southern Cross (wordless novel)": "Literature",
    "Uskok-class torpedo boat": "Military",
    "Vision in White": "Literature",
    "Blue men of the Minch": "Culture",
    "Portrait of Maria Portinari": "Arts",
    "Donald Forrester Brown": "Military",
    "Kedok Ketawa": "Film & TV",
    "Yugoslav destroyer Zagreb": "Military",
    "CSS General Earl Van Dorn": "Military",
    "Oryzomys peninsulae": "Biology",
    "Double Seven Day scuffle": "History",
    "Gravity Bone": "Video Games",
    "Oryzomys dimidiatus": "Biology",
    "Nothing to My Name": "Music",
    "Northampton War Memorial": "Arts",
    "Freedom of Worship (painting)": "Arts",
    "Mells War Memorial": "Arts",
    "Subway Sadie": "Film & TV",
    "Battle of Babylon Hill": "History",
    "Eremoryzomys": "Biology",
    "Interstate 15 in Arizona": "Geography",
    "Sorga Ka Toedjoe": "Film & TV",
    "Charles Domery": "Culture",
    "La Salute è in voi": "Film & TV",
    "Michael Francis Egan": "Religion",
    "Battle of Oroscopa": "History",
    "Pennatomys": "Biology",
    "Operation Ironside": "Military",
    "Withypool Stone Circle": "Arts",
    "Ambondro mahabo": "Biology",
    "Wihtred of Kent": "History",
    "Cryptoprocta spelea": "Biology",
    "Battle of Marshall's Elm": "History",
    "Portrait of a Young Girl (Christus)": "Arts",
    "Hurricane Ismael": "Weather",
    "Harta Berdarah": "Film & TV",
    "Baljuna Covenant": "History",
    "Hurricane Kiko (1989)": "Weather",
    "Russian battleship Dvenadsat Apostolov": "Military",
    "Hurricane Vince": "Weather",
    "Francesco Caracciolo-class battleship": "Military",
    "Battle of San Patricio": "History",
    "Sutton Hoo Helmet (sculpture)": "Arts",
    "Dish-bearers and butlers in Anglo-Saxon England": "Culture",
    "Satellite Science Fiction": "Literature",
    "After the Deluge (painting)": "Arts",
    "Hellingly Hospital Railway": "Transport",
    "Big Boys (song)": "Music",
    "Paulinus of York": "Religion",
    "Saint Vincent Beer": "Culture",
    "The Temple at Thatch": "Literature",
    "Horncastle boar's head": "Arts",
    "Siege of Guînes (1352)": "Military",
    "It Is the Law": "Film & TV",
    "Elizabeth Lyon (criminal)": "History",
    "Margaret Abbott": "Sports",
    "Coenred of Mercia": "History",
    "Overdrawn at the Memory Bank": "Film & TV",
    "1955 MacArthur Airport United Air Lines crash": "Transport",
    "Slug (song)": "Music",
    "Laborintus II (album)": "Music",
    "Zombie Nightmare": "Film & TV",
    "Siege of Hennebont": "Military",
    "200 (Stargate SG-1)": "Film & TV",
    "Duckport Canal": "Geography",
    "St Botolph's Church, Quarrington": "Arts",
    "Westcott railway station": "Transport",
    "Norfolk, Virginia, Bicentennial half dollar": "Culture",
    "CMLL World Middleweight Championship": "Sports",
    "Sinistar: Unleashed": "Video Games",
    "Flotilla (video game)": "Video Games",
    "Wandsworth Bridge": "Transport",
    "The Guardian of Education": "Literature",
    "Confusion (album)": "Music",
    "Brill railway station": "Transport",
    "Cædwalla": "History",
}

# Apply manual + auto classification
for a in articles:
    if a["title"] in MANUAL:
        a["category"] = MANUAL[a["title"]]

# Final distribution
cat_counts = {}
for a in articles:
    cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1

print("Final category distribution:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25s}: {cnt}")

others = [a for a in articles if a["category"] == "Other"]
if others:
    print(f"\nRemaining 'Other' ({len(others)}):")
    for a in others:
        print(f"  - {a['title']}")

# Save
with open("scripts/selected_150_final.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

# Generate YAML
print("\n\n# === YAML FOR config.yaml ===")
print("articles:")
articles.sort(key=lambda x: x["category"])  # sort by category for readability
for a in articles:
    safe = a["title"].replace('"', '\\"')
    print(f'  - title: "{safe}"')
    print(f'    category: "{a["category"]}"')
