from wordfreq import word_frequency as wf

pairs = [
    ("rain", "rainfall"),
    ("painting", "picture"),
    ("position", "place"),
    ("companion", "comrade"),
    ("underage", "minor"),
    ("influence", "effect"),
    ("elaborate", "complex"),
    ("exhibited", "showed"),
    ("periodical", "magazine"),
    ("proffered", "offered"),
    ("vermilion", "red"),
    ("turpentine", "oil"),
    ("embalmed", "preserved"),
    ("pathological", "extreme"),
    ("guardianship", "care"),
]

for a, b in pairs:
    fa, fb = wf(a, "en"), wf(b, "en")
    verdict = "REPLACE" if fb > fa else "KEEP"
    print(f"  {a:18s}({fa:.2e}) vs {b:18s}({fb:.2e})  ->  {verdict}")
