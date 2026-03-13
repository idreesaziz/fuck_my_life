"""
Step 2 — Degradation Engine
Corrupts clean texts along four independent, orthogonal axes at controlled
intensity levels.

Axes
────
1. Grammar — Rich grammatical corruption:
   • Keyboard typos (character-level)
   • Subject-verb agreement errors ("the dogs runs")
   • Article misuse ("a elephants", dropped/extra articles)
   • Tense inconsistency (past↔present swaps)
   • Preposition errors ("interested for", "depend of")
   • Hyphenation errors ("well-known" → "well known" and vice versa)
   • Homophones/confusables (its/it's, their/there, affect/effect)
   • Double negatives, wrong comparatives ("more better")
   • Mild word-order disruption within sentences

2. Coherence — Sentence-order shuffling:
   • Preserves grammar, vocabulary, and information completely
   • Breaks logical flow, coreference chains, and argument structure
   • Low levels: swap adjacent sentence pairs
   • High levels: near-random permutation

3. Information — Word/phrase/clause deletion (NOT sentence deletion):
   • Removes modifiers (adjectives, adverbs) at low levels
   • Removes parenthetical/appositive phrases at mid levels
   • Removes subordinate clauses and content words at high levels
   • Text stays readable but progressively loses substance

4. Lexical — Vocabulary flattening:
   • Replaces diverse/academic vocabulary with simple common words
   • Uses WordNet synonym collapse at higher levels
   • Reduces lexical variety without changing grammar or meaning
"""

import json
import random
import re
from pathlib import Path

import nltk
from tqdm import tqdm

# Ensure NLTK data is available
_NLTK_RESOURCES = [
    ("tokenizers", "punkt"),
    ("tokenizers", "punkt_tab"),
    ("taggers", "averaged_perceptron_tagger"),
    ("taggers", "averaged_perceptron_tagger_eng"),
    ("corpora", "wordnet"),
]
for folder, name in _NLTK_RESOURCES:
    try:
        nltk.data.find(f"{folder}/{name}")
    except LookupError:
        nltk.download(name, quiet=True)


# ═══════════════════════════════════════════════════════════════════
# Axis 1: Grammar Corruption
# ═══════════════════════════════════════════════════════════════════

# --- Confusable word pairs (homophones, near-homophones) ---
_CONFUSABLES = {
    "its": "it's", "it's": "its",
    "their": "there", "there": "their",
    "they're": "their", "your": "you're", "you're": "your",
    "affect": "effect", "effect": "affect",
    "then": "than", "than": "then",
    "who's": "whose", "whose": "who's",
    "to": "too", "too": "to",
    "accept": "except", "except": "accept",
    "lose": "loose", "loose": "lose",
    "complement": "compliment", "compliment": "complement",
    "principal": "principle", "principle": "principal",
    "stationary": "stationery", "stationery": "stationary",
    "weather": "whether", "whether": "weather",
    "cite": "site", "site": "cite",
}

# --- Wrong prepositions ---
_WRONG_PREPS = {
    "interested in": "interested for",
    "depend on": "depend of",
    "capable of": "capable to",
    "consist of": "consist from",
    "according to": "according with",
    "result in": "result to",
    "different from": "different than",
    "similar to": "similar with",
    "aware of": "aware about",
    "based on": "based off",
    "responsible for": "responsible of",
    "refer to": "refer at",
    "related to": "related with",
    "arrived at": "arrived to",
    "independent of": "independent from",
}

# --- Agreement error patterns ---
_AGREEMENT_SUBS = [
    (r'\b(he|she|it) (was)\b', r'\1 were'),
    (r'\b(they) (were)\b', r'\1 was'),
    (r'\b(he|she|it) (has)\b', r'\1 have'),
    (r'\b(they) (have)\b', r'\1 has'),
    (r'\b(he|she|it) (does)\b', r'\1 do'),
    (r'\b(they) (do)\b', r'\1 does'),
    (r'\b(he|she|it) (is)\b', r'\1 are'),
    (r'\b(they) (are)\b', r'\1 is'),
]

# --- Tense swap patterns ---
_TENSE_SWAPS = [
    (r'\b(was)\b', 'is'), (r'\b(were)\b', 'are'),
    (r'\b(had)\b', 'has'), (r'\b(did)\b', 'does'),
    (r'\b(went)\b', 'goes'), (r'\b(came)\b', 'comes'),
    (r'\b(took)\b', 'takes'), (r'\b(made)\b', 'makes'),
    (r'\b(said)\b', 'says'), (r'\b(became)\b', 'becomes'),
    (r'\b(began)\b', 'begins'), (r'\b(gave)\b', 'gives'),
    (r'\b(found)\b', 'finds'), (r'\b(built)\b', 'builds'),
    (r'\b(led)\b', 'leads'), (r'\b(wrote)\b', 'writes'),
    (r'\b(ran)\b', 'runs'), (r'\b(held)\b', 'holds'),
    (r'\b(brought)\b', 'brings'), (r'\b(fought)\b', 'fights'),
]

# --- Wrong comparatives ---
_BAD_COMPARATIVES = {
    "better": "more better", "worse": "more worse",
    "bigger": "more bigger", "smaller": "more smaller",
    "faster": "more faster", "larger": "more larger",
    "easier": "more easier", "harder": "more harder",
}

# --- Hyphenation errors ---
_HYPHEN_SPLITS = [
    "well-known", "long-term", "high-quality", "so-called", "self-made",
    "non-profit", "full-time", "part-time", "large-scale", "small-scale",
    "well-established", "short-lived", "long-standing", "wide-ranging",
    "far-reaching", "hard-working", "old-fashioned", "open-ended",
    "state-of-the-art", "up-to-date", "first-hand", "second-hand",
    "man-made", "hand-made", "re-elected", "co-authored",
]
_HYPHEN_JOINS = [
    ("every day", "everyday"), ("any time", "anytime"),
    ("every one", "everyone"), ("some times", "sometimes"),
    ("in to", "into"), ("on to", "onto"),
]


def _apply_keyboard_typos(text: str, level: float, rng: random.Random) -> str:
    """Character-level typos: adjacent key substitutions, doubled chars,
    dropped chars."""
    char_p = min(level * 0.02, 0.03)
    adjacent_keys = {
        'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
        'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
        'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'bm', 'o': 'ip',
        'p': 'o', 'q': 'w', 'r': 'et', 's': 'ad', 't': 'ry',
        'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
        'z': 'x',
    }
    result = []
    for ch in text:
        if ch.isalpha() and rng.random() < char_p:
            action = rng.choice(["swap", "double", "drop"])
            lower = ch.lower()
            if action == "swap" and lower in adjacent_keys:
                replacement = rng.choice(adjacent_keys[lower])
                result.append(replacement if ch.islower() else replacement.upper())
            elif action == "double":
                result.append(ch)
                result.append(ch)
            else:  # drop
                pass
        else:
            result.append(ch)
    return "".join(result)


def _apply_agreement_errors(text: str, level: float, rng: random.Random) -> str:
    """Subject-verb agreement corruption."""
    for pattern, replacement in _AGREEMENT_SUBS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if rng.random() < level * 0.3:
                new = re.sub(pattern, replacement, m.group(0), flags=re.IGNORECASE)
                text = text[:m.start()] + new + text[m.end():]
    return text


def _apply_tense_swaps(text: str, level: float, rng: random.Random) -> str:
    """Swap past↔present tense verbs."""
    for pattern, replacement in _TENSE_SWAPS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if rng.random() < level * 0.15:
                original = m.group(0)
                new = replacement
                if original[0].isupper():
                    new = new.capitalize()
                text = text[:m.start()] + new + text[m.end():]
    return text


def _apply_article_errors(text: str, level: float, rng: random.Random) -> str:
    """Misuse articles: wrong a/an, drop articles, insert spurious ones."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        lower = w.lower().rstrip(".,;:!?")

        if lower in ("a", "an") and rng.random() < level * 0.3:
            new = "an" if lower == "a" else "a"
            if w[0].isupper():
                new = new.capitalize()
            trail = w[len(lower):]
            result.append(new + trail)
        elif lower in ("the", "a", "an") and rng.random() < level * 0.15:
            pass  # drop article
        elif (w[0].isupper() and i > 0 and not words[i-1].endswith(".")
              and rng.random() < level * 0.05):
            result.append("the")
            result.append(w)
        else:
            result.append(w)
        i += 1
    return " ".join(result)


def _apply_confusables(text: str, level: float, rng: random.Random) -> str:
    """Swap homophones and commonly confused words."""
    words = text.split()
    result = []
    for w in words:
        stripped = w.lower().rstrip(".,;:!?\"')")
        trail = w[len(stripped):]
        if stripped in _CONFUSABLES and rng.random() < level * 0.25:
            replacement = _CONFUSABLES[stripped]
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement + trail)
        else:
            result.append(w)
    return " ".join(result)


def _apply_preposition_errors(text: str, level: float, rng: random.Random) -> str:
    """Replace correct preposition phrases with wrong ones."""
    for correct, wrong in _WRONG_PREPS.items():
        if correct in text.lower() and rng.random() < level * 0.4:
            idx = text.lower().find(correct)
            if idx >= 0:
                text = text[:idx] + wrong + text[idx + len(correct):]
    return text


def _apply_comparative_errors(text: str, level: float, rng: random.Random) -> str:
    """Insert double comparatives ("more better")."""
    for correct, wrong in _BAD_COMPARATIVES.items():
        pattern = rf'\b{correct}\b'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if rng.random() < level * 0.5:
                text = text[:m.start()] + wrong + text[m.end():]
    return text


def _apply_hyphenation_errors(text: str, level: float, rng: random.Random) -> str:
    """Break hyphens or wrongly join words."""
    for compound in _HYPHEN_SPLITS:
        if compound in text.lower() and rng.random() < level * 0.4:
            idx = text.lower().find(compound)
            if idx >= 0:
                original = text[idx:idx + len(compound)]
                replacement = original.replace("-", " ")
                text = text[:idx] + replacement + text[idx + len(compound):]

    for separate, joined in _HYPHEN_JOINS:
        if separate in text.lower() and rng.random() < level * 0.3:
            idx = text.lower().find(separate)
            if idx >= 0:
                text = text[:idx] + joined + text[idx + len(separate):]

    return text


def _apply_word_order_disruption(text: str, level: float,
                                 rng: random.Random) -> str:
    """Mild within-sentence word swaps (adjacent pair transpositions)."""
    if level < 0.3:
        return text

    sentences = nltk.sent_tokenize(text)
    result = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 4:
            result.append(sent)
            continue
        n_swaps = max(0, int(len(words) * level * 0.08))
        for _ in range(n_swaps):
            idx = rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        result.append(" ".join(words))
    return " ".join(result)


def degrade_grammar(text: str, level: float, rng: random.Random) -> str:
    """Rich grammatical corruption combining multiple error types.
    Each sub-corruption is applied with probability scaled by level."""
    if level <= 0:
        return text

    text = _apply_keyboard_typos(text, level, rng)
    text = _apply_agreement_errors(text, level, rng)
    text = _apply_tense_swaps(text, level, rng)
    text = _apply_article_errors(text, level, rng)
    text = _apply_confusables(text, level, rng)
    text = _apply_preposition_errors(text, level, rng)
    text = _apply_comparative_errors(text, level, rng)
    text = _apply_hyphenation_errors(text, level, rng)
    text = _apply_word_order_disruption(text, level, rng)

    return text


# ═══════════════════════════════════════════════════════════════════
# Axis 2: Coherence — Sentence-Order Shuffling
# ═══════════════════════════════════════════════════════════════════

def degrade_coherence(text: str, level: float, rng: random.Random) -> str:
    """Shuffle sentence order to break logical flow and coreference chains.
    Preserves grammar, vocabulary, and information completely.

    level 0.2: swap a few adjacent sentence pairs
    level 0.4: displace some sentences by a few positions
    level 0.6: significant reordering
    level 0.8: near-random permutation
    """
    if level <= 0:
        return text

    sentences = nltk.sent_tokenize(text)
    n = len(sentences)
    if n <= 2:
        return text

    n_swaps = max(1, int(n * level * 0.8))
    max_displacement = max(1, int(n * level * 0.6))

    indices = list(range(n))
    for _ in range(n_swaps):
        i = rng.randint(0, n - 1)
        j_min = max(0, i - max_displacement)
        j_max = min(n - 1, i + max_displacement)
        j = rng.randint(j_min, j_max)
        indices[i], indices[j] = indices[j], indices[i]

    shuffled = [sentences[i] for i in indices]
    return " ".join(shuffled)


# ═══════════════════════════════════════════════════════════════════
# Axis 3: Information — Word/Phrase/Clause Deletion
# ═══════════════════════════════════════════════════════════════════

_MODIFIER_TAGS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}
_CONTENT_TAGS = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG",
                 "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

_PARENTHETICAL_RE = re.compile(r'\([^)]{5,80}\)')
_SUBORD_CLAUSE_RE = re.compile(
    r'\b(?:which|who|whom|that|although|though|because|since|while|whereas|'
    r'unless|whenever|wherever|if|when|after|before|until)\b[^.;]{10,80}[,.]',
    re.IGNORECASE
)
_PREP_PHRASE_RE = re.compile(
    r'\b(?:in|on|at|by|for|with|from|about|during|through|between|among|'
    r'under|above|after|before|without|within|across|behind|beyond)\s+'
    r'(?:the\s+|a\s+|an\s+)?(?:\w+\s*){1,5}',
    re.IGNORECASE
)


def _delete_modifiers(text: str, level: float, rng: random.Random) -> str:
    """Remove adjectives and adverbs."""
    sentences = nltk.sent_tokenize(text)
    result_sents = []
    for sent in sentences:
        try:
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        except Exception:
            result_sents.append(sent)
            continue
        kept = []
        for word, tag in tagged:
            if tag in _MODIFIER_TAGS and rng.random() < level * 0.5:
                continue
            kept.append(word)
        result_sents.append(" ".join(kept))
    return " ".join(result_sents)


def _delete_parentheticals(text: str, level: float,
                           rng: random.Random) -> str:
    """Remove parenthetical expressions like (born 1923)."""
    def maybe_remove(m):
        if rng.random() < level * 0.6:
            return ""
        return m.group(0)
    return _PARENTHETICAL_RE.sub(maybe_remove, text)


def _delete_subordinate_clauses(text: str, level: float,
                                rng: random.Random) -> str:
    """Remove subordinate/relative clauses."""
    def maybe_remove(m):
        if rng.random() < level * 0.3:
            ending = m.group(0)[-1]
            return ending if ending in ".," else ""
        return m.group(0)
    return _SUBORD_CLAUSE_RE.sub(maybe_remove, text)


def _delete_content_words(text: str, level: float,
                          rng: random.Random) -> str:
    """At high levels, drop random content words (nouns, verbs)."""
    if level < 0.5:
        return text

    sentences = nltk.sent_tokenize(text)
    result_sents = []
    for sent in sentences:
        try:
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        except Exception:
            result_sents.append(sent)
            continue
        kept = []
        for word, tag in tagged:
            if (tag in _CONTENT_TAGS and tag not in _MODIFIER_TAGS
                    and rng.random() < (level - 0.4) * 0.2):
                continue
            kept.append(word)
        if kept:
            result_sents.append(" ".join(kept))
    return " ".join(result_sents)


def _delete_prep_phrases(text: str, level: float,
                         rng: random.Random) -> str:
    """Remove prepositional phrases at mid-high levels."""
    if level < 0.3:
        return text

    def maybe_remove(m):
        if rng.random() < (level - 0.2) * 0.25:
            return ""
        return m.group(0)
    return _PREP_PHRASE_RE.sub(maybe_remove, text)


def degrade_information(text: str, level: float, rng: random.Random) -> str:
    """Progressive information loss through word/phrase/clause deletion.

    Low levels (0.2):  drop modifiers and parentheticals
    Mid levels (0.4):  also remove subordinate clauses, some prep phrases
    High levels (0.6+): also remove content words
    """
    if level <= 0:
        return text

    text = _delete_parentheticals(text, level, rng)
    text = _delete_modifiers(text, level, rng)
    text = _delete_prep_phrases(text, level, rng)
    text = _delete_subordinate_clauses(text, level, rng)
    text = _delete_content_words(text, level, rng)

    # Clean up artifacts
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Axis 4: Lexical Resource Collapse
# ═══════════════════════════════════════════════════════════════════

_WORDNET_POS_MAP = {
    "NN": "n", "NNS": "n",
    "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
    "JJ": "a", "JJR": "a", "JJS": "a",
    "RB": "r", "RBR": "r", "RBS": "r",
}

# Words that should never be replaced
_WORDNET_SKIP = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "not", "no", "it", "its",
    "he", "she", "they", "we", "you", "i", "me", "him", "her",
    "them", "us", "my", "his", "our", "your", "their", "this",
    "that", "these", "those", "which", "who", "whom", "whose",
    "what", "where", "when", "how", "why", "if", "then", "than",
    "as", "so", "very", "also", "just", "only", "more", "most",
    "much", "many", "some", "any", "all", "each", "every", "both",
    "few", "other", "such", "own", "same", "first", "last", "new",
    "old", "good", "great", "little", "big", "long", "after",
    "before", "between", "over", "under", "about", "up", "down",
    "out", "into", "through", "during", "without", "within",
    # Prone to bad cross-POS replacements
    "short", "hard", "fast", "well", "right", "left", "light",
    "close", "late", "early", "still", "even", "back", "round",
    "like", "near", "open", "free", "fine", "clear", "full",
    "high", "low", "true", "real", "sure", "dead", "live",
}

# Irregular verb tables for morphology transfer
_IRREGULAR_PAST = {
    "get": "got", "say": "said", "make": "made", "know": "knew",
    "come": "came", "take": "took", "give": "gave", "find": "found",
    "tell": "told", "show": "showed", "go": "went", "see": "saw",
    "run": "ran", "build": "built", "hold": "held", "write": "wrote",
    "keep": "kept", "lead": "led", "meet": "met", "pay": "paid",
    "leave": "left", "bring": "brought", "begin": "began",
    "grow": "grew", "draw": "drew", "break": "broke",
    "speak": "spoke", "drive": "drove", "rise": "rose",
    "choose": "chose", "fall": "fell", "bear": "bore",
    "think": "thought", "feel": "felt", "send": "sent",
    "stand": "stood", "lose": "lost", "cut": "cut",
    "put": "put", "set": "set", "let": "let",
    "do": "did", "have": "had", "be": "was",
    "control": "controlled", "admit": "admitted", "occur": "occurred",
    "refer": "referred", "permit": "permitted", "submit": "submitted",
    "prefer": "preferred", "commit": "committed", "omit": "omitted",
}
_IRREGULAR_PARTICIPLE = {
    "get": "gotten", "say": "said", "make": "made", "know": "known",
    "come": "come", "take": "taken", "give": "given", "find": "found",
    "tell": "told", "show": "shown", "go": "gone", "see": "seen",
    "run": "run", "build": "built", "hold": "held", "write": "written",
    "keep": "kept", "lead": "led", "meet": "met", "pay": "paid",
    "leave": "left", "bring": "brought", "begin": "begun",
    "grow": "grown", "draw": "drawn", "break": "broken",
    "speak": "spoken", "drive": "driven", "rise": "risen",
    "choose": "chosen", "fall": "fallen", "bear": "borne",
    "think": "thought", "feel": "felt", "send": "sent",
    "stand": "stood", "lose": "lost", "cut": "cut",
    "put": "put", "set": "set", "let": "let",
    "do": "done", "have": "had", "be": "been",
    "control": "controlled", "admit": "admitted", "occur": "occurred",
    "refer": "referred", "permit": "permitted", "submit": "submitted",
    "prefer": "preferred", "commit": "committed", "omit": "omitted",
}


def _needs_doubling(base: str) -> bool:
    """Check if final consonant should double (CVC pattern, monosyllabic)."""
    if len(base) < 3:
        return False
    vowels = set("aeiou")
    groups = 0
    in_vowel = False
    for c in base:
        if c in vowels:
            if not in_vowel:
                groups += 1
                in_vowel = True
        else:
            in_vowel = False
    if groups != 1:
        return False
    return (base[-1] not in vowels and base[-1] not in "wxy"
            and base[-2] in vowels
            and base[-3] not in vowels)


def _transfer_morphology(original: str, base_replacement: str, tag: str) -> str:
    """Transfer verb/noun morphology from original word to replacement."""
    repl_lower = base_replacement.lower()

    if tag in ("NN", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"):
        return base_replacement
    if tag == "NNS":
        if not base_replacement.endswith("s"):
            return base_replacement + "s"
        return base_replacement

    if tag == "VBD":
        if repl_lower in _IRREGULAR_PAST:
            return _IRREGULAR_PAST[repl_lower]
        if repl_lower.endswith("e"):
            return base_replacement + "d"
        if _needs_doubling(repl_lower):
            return base_replacement + base_replacement[-1] + "ed"
        return base_replacement + "ed"
    elif tag == "VBN":
        if repl_lower in _IRREGULAR_PARTICIPLE:
            return _IRREGULAR_PARTICIPLE[repl_lower]
        if repl_lower.endswith("e"):
            return base_replacement + "d"
        if _needs_doubling(repl_lower):
            return base_replacement + base_replacement[-1] + "ed"
        return base_replacement + "ed"
    elif tag == "VBG":
        if repl_lower.endswith("e"):
            return base_replacement[:-1] + "ing"
        if _needs_doubling(repl_lower):
            return base_replacement + base_replacement[-1] + "ing"
        return base_replacement + "ing"
    elif tag == "VBZ":
        if repl_lower.endswith(("s", "x", "z", "ch", "sh")):
            return base_replacement + "es"
        return base_replacement + "s"
    else:
        return base_replacement


def _get_lemma(word: str, tag: str) -> str:
    """Get the base/lemma form of a word for synonym lookup."""
    from nltk.stem import WordNetLemmatizer
    _wnl = WordNetLemmatizer()
    wn_pos = _WORDNET_POS_MAP.get(tag, "n")
    return _wnl.lemmatize(word.lower(), pos=wn_pos)


def degrade_lexical(text: str, level: float, rng: random.Random) -> str:
    """Collapse vocabulary by unifying synonyms to the simplest form.

    Algorithm:
    1. Tokenize + POS-tag the entire text.
    2. Walk content words (noun/verb/adjective). For each:
       a. Lemmatize it, get ALL its WordNet synsets.
       b. Collect single-word synonyms from those synsets.
       c. Find text positions whose lemma matches any of those synonyms.
       d. Pick the simplest synonym (highest wordfreq). Replace others.
    3. Stop once the target percentage of content words has been replaced.
       No transitive chaining — each word's group is independent.
    """
    if level <= 0:
        return text

    from nltk.corpus import wordnet as wn
    from wordfreq import word_frequency

    target_frac = level * 0.7

    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        return text

    _ALLOWED_POS = {
        "NN", "NNS",
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "JJ", "JJR", "JJS",
    }

    # --- Phase 1: identify content-word positions and their lemmas ---
    content_indices = []
    lemma_at = {}
    pos_at = {}

    for i, (word, tag) in enumerate(tagged):
        if (tag in _ALLOWED_POS
                and len(word) >= 4
                and word.isalpha()
                and word.lower() not in _WORDNET_SKIP):
            content_indices.append(i)
            lemma = _get_lemma(word, tag)
            lemma_at[i] = lemma
            pos_at[i] = _WORDNET_POS_MAP.get(tag)

    total_content = len(content_indices)
    if total_content == 0:
        return text

    max_replacements = int(total_content * target_frac)
    if max_replacements == 0:
        return text

    # --- Phase 2: walk content words, collect synonym groups ---
    new_tokens = list(tokens)
    replaced = set()
    processed_lemmas = set()
    replacements_done = 0

    for i in content_indices:
        if replacements_done >= max_replacements:
            break
        if i in replaced:
            continue

        word, tag = tagged[i]
        lemma = lemma_at[i]

        if lemma in processed_lemmas:
            continue
        processed_lemmas.add(lemma)

        wn_pos = pos_at[i]
        if wn_pos is None:
            continue

        # Get ALL synsets for this lemma and collect synonym lemmas
        synsets = wn.synsets(lemma, pos=wn_pos)
        if not synsets:
            continue

        synonym_lemmas = set()
        for ss in synsets:
            # Only include synsets where our lemma is explicitly listed
            ss_lemma_names = {l.name().lower() for l in ss.lemmas()}
            if lemma not in ss_lemma_names:
                continue
            for lem in ss.lemmas():
                name = lem.name().lower()
                if name.isalpha() and len(name) >= 3 and "_" not in lem.name():
                    synonym_lemmas.add(name)

        if len(synonym_lemmas) < 2:
            continue

        # Find content positions whose lemma is in this synonym group
        group_positions = []
        for j in content_indices:
            if j in replaced:
                continue
            if lemma_at[j] in synonym_lemmas:
                group_positions.append(j)

        if len(group_positions) < 2:
            continue

        # Pick the simplest synonym (highest wordfreq frequency)
        best_word = None
        best_freq = -1
        for syn in synonym_lemmas:
            freq = word_frequency(syn, "en")
            if freq > best_freq:
                best_freq = freq
                best_word = syn

        if best_word is None:
            continue

        # Replace positions not already at the simplest form
        for j in group_positions:
            if replacements_done >= max_replacements:
                break
            if lemma_at[j] == best_word:
                continue

            orig_word, orig_tag = tagged[j]
            replacement = _transfer_morphology(orig_word, best_word, orig_tag)
            if orig_word[0].isupper():
                replacement = replacement.capitalize()
            if orig_word.isupper():
                replacement = replacement.upper()

            new_tokens[j] = replacement
            replaced.add(j)
            replacements_done += 1

        # Mark all synonym lemmas as processed
        processed_lemmas.update(synonym_lemmas)

    # --- Phase 3: reconstruct text ---
    rebuilt = []
    for tok in new_tokens:
        if tok in ".,;:!?)}]'\"" and rebuilt:
            rebuilt[-1] += tok
        elif tok in "({[\"'" and rebuilt:
            rebuilt.append(tok)
        else:
            rebuilt.append(tok)
    return " ".join(rebuilt)


# ═══════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════

AXIS_FUNCTIONS = {
    "grammar": degrade_grammar,
    "coherence": degrade_coherence,
    "information": degrade_information,
    "lexical": degrade_lexical,
}


def degrade_text(text: str, axis: str, level: float, seed: int = 42) -> str:
    """Apply a single degradation axis at a given level to text."""
    rng = random.Random(seed)
    fn = AXIS_FUNCTIONS[axis]
    return fn(text, level, rng)


# ═══════════════════════════════════════════════════════════════════
# Batch Processing
# ═══════════════════════════════════════════════════════════════════

def run(config: dict, corpus: list[dict]) -> list[dict]:
    """Generate all degraded samples. Returns list of sample dicts."""
    deg_cfg = config["degradation"]
    levels = deg_cfg["levels"]
    samples_per_level = deg_cfg["samples_per_level"]
    output_dir = Path(deg_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "degraded_samples.json"
    if output_file.exists():
        print("[Degradation] Loading existing degraded samples...")
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    axes = [a for a, cfg in deg_cfg["axes"].items() if cfg.get("enabled", True)]
    all_samples = []
    sample_id = 0

    total = len(corpus) * len(axes) * len(levels) * samples_per_level
    pbar = tqdm(total=total, desc="Degrading texts")

    for article in corpus:
        for axis in axes:
            for level in levels:
                for rep in range(samples_per_level):
                    seed = hash((article["title"], axis, level, rep)) % (2**31)
                    degraded_text = degrade_text(
                        article["text"], axis, level, seed=seed
                    )
                    sample = {
                        "id": sample_id,
                        "source_title": article["title"],
                        "category": article.get("category", "uncategorized"),
                        "axis": axis,
                        "level": level,
                        "repetition": rep,
                        "seed": seed,
                        "original_text": article["text"],
                        "degraded_text": degraded_text,
                    }
                    all_samples.append(sample)
                    sample_id += 1
                    pbar.update(1)

    pbar.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"[Degradation] Saved {len(all_samples)} samples to {output_file}")

    return all_samples
