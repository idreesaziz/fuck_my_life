# Condemned to B

**Nobody Gets an A: Distributional Bias in LLM Text Scoring**

An empirical investigation into score compression in large language model text evaluation — the systematic tendency of LLMs to avoid extreme ratings, compressing scores toward the interior of any provided scale regardless of true quality.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Experimental Design](#experimental-design)
3. [Corpus Construction](#corpus-construction)
4. [Degradation Engine](#degradation-engine)
   - [Axis 1: Grammar Corruption](#axis-1-grammar-corruption)
   - [Axis 2: Coherence Disruption](#axis-2-coherence-disruption)
   - [Axis 3: Information Deletion](#axis-3-information-deletion)
   - [Axis 4: Lexical Collapse](#axis-4-lexical-collapse)
5. [Deterministic Reproducibility](#deterministic-reproducibility)
6. [Scoring Protocol](#scoring-protocol)
7. [Statistical Analysis](#statistical-analysis)
8. [Repository Structure](#repository-structure)
9. [Running the Pipeline](#running-the-pipeline)

---

## Motivation

When asked to rate text quality on a numeric scale (e.g., 0–10), LLMs rarely assign scores at the extremes. A pristine, publication-quality text that a human expert would comfortably rate 9 or 10 consistently receives a 7 or 8. Conversely, heavily corrupted text that is borderline unreadable rarely drops below 2 or 3. The empirical score distribution concentrates in a narrow band, regardless of the true quality range of the input.

This project constructs a controlled experiment to measure this compression effect rigorously. By starting from texts of known high quality and applying *precisely controlled, mathematically defined* degradations along independent axes, we create a ground-truth quality gradient and measure how faithfully LLMs reproduce it.

---

## Experimental Design

### Overview

The experiment operates as a factorial design:

$$N = |\mathcal{C}| \times |\mathcal{A}| \times |\mathcal{L}| \times R$$

where:
- $\mathcal{C}$: set of corpus texts ($|\mathcal{C}| = 150$)
- $\mathcal{A}$: set of degradation axes ($|\mathcal{A}| = 4$: grammar, coherence, information, lexical)
- $\mathcal{L}$: set of degradation levels ($|\mathcal{L}| = 5$: $\{0.0, 0.2, 0.4, 0.6, 0.8\}$)
- $R$: repetitions per condition ($R = 3$)

This yields $150 \times 4 \times 5 \times 3 = 9{,}000$ degraded samples.

### Axis Isolation

Each sample is degraded along exactly *one* axis. This is critical: by corrupting grammar while holding coherence, information, and vocabulary constant (and vice versa), we can attribute any scoring biases to specific quality dimensions rather than confounded combinations.

### Scale

All LLM scores are elicited on an integer scale from 0 to 10.

---

## Corpus Construction

### Source Material

The corpus consists of 150 Wikipedia Featured Articles (FAs) — articles that have passed Wikipedia's rigorous peer review process (FA candidacy) and represent the highest quality tier of encyclopedic writing. FAs are selected because they provide:

1. **Known high quality**: Each FA has been reviewed and promoted by experienced editors, establishing a credible "near-ceiling" baseline.
2. **Topical diversity**: Articles are drawn from 16 categories (Arts, Biology, Culture, Film & TV, Geography, History, Mathematics, Media, Military, Music, Philosophy, Politics, Science, Sport, Technology, Transport) to avoid domain-specific scoring biases.
3. **Availability**: Full text is freely accessible via the Wikipedia API.

### Selection Criteria

From the pool of all English Wikipedia FAs, the 150 shortest articles (by raw character count) are selected. This serves two purposes:

- **Token economy**: Shorter texts reduce API costs without sacrificing quality diversity.
- **Full-text integrity**: No truncation is applied ($\texttt{max\_chars} = 0$). Each article is used in its entirety, preserving the natural structure, argument flow, and coherence that FAs are scored on.

The resulting corpus spans approximately 7,500–22,500 characters per article, with articles stored as JSON records containing title, category, and full text.

### Fetching

Corpus acquisition uses the `wikipedia-api` library with a custom user agent. Articles that do not exist or fall below 500 characters are excluded, with warnings logged.

---

## Degradation Engine

The degradation engine (`src/degradation.py`) corrupts clean texts along four independent, orthogonal axes at controlled intensity levels. Each degradation function takes:

- `text`: the original article text (a string)
- `level`: degradation intensity $\lambda \in [0, 1]$ (in practice $\{0.0, 0.2, 0.4, 0.6, 0.8\}$)
- `rng`: a seeded `random.Random` instance for reproducibility

At $\lambda = 0$, the original text is returned unchanged. As $\lambda$ increases, corruption severity increases monotonically.

---

### Axis 1: Grammar Corruption

Grammar degradation introduces surface-level mechanical errors while preserving meaning, vocabulary, and information content. It is implemented as a pipeline of nine independent sub-transformations, applied sequentially:

$$\text{degrade\_grammar}(t, \lambda) = (f_9 \circ f_8 \circ \cdots \circ f_1)(t, \lambda)$$

Each sub-transformation applies its errors stochastically, with individual error probabilities scaled by $\lambda$.

#### Sub-transformation 1: Keyboard Typos

Simulates mechanical typing errors. For each alphabetic character $c$ in the text, with probability:

$$p_{\text{typo}} = \min(\lambda \cdot 0.02,\ 0.03)$$

one of three mutations is applied (chosen uniformly at random):

| Mutation | Effect |
|----------|--------|
| **Swap** | Replace $c$ with a random character from its adjacent keys on a QWERTY keyboard |
| **Double** | Duplicate $c$ (e.g., `t` → `tt`) |
| **Drop** | Delete $c$ entirely |

The adjacency map covers all 26 lowercase letters with their QWERTY neighbors.

#### Sub-transformation 2: Agreement Errors

Introduces subject-verb agreement violations using 8 regex substitution patterns. For each pattern match, the error is applied with probability:

$$p_{\text{agree}} = \lambda \cdot 0.3$$

Examples:
- *"he was"* → *"he were"*
- *"they have"* → *"they has"*
- *"it is"* → *"it are"*

#### Sub-transformation 3: Tense Swaps

Replaces past-tense verb forms with present-tense equivalents using 20 regex patterns. For each match:

$$p_{\text{tense}} = \lambda \cdot 0.15$$

Examples: *"was"* → *"is"*, *"wrote"* → *"writes"*, *"brought"* → *"brings"*.

#### Sub-transformation 4: Article Errors

Three types of article manipulation, applied per word:

| Error | Probability | Effect |
|-------|-------------|--------|
| a/an swap | $\lambda \cdot 0.3$ | *"a elephant"*, *"an car"* |
| Article deletion | $\lambda \cdot 0.15$ | Drop *"the"*, *"a"*, *"an"* |
| Spurious insertion | $\lambda \cdot 0.05$ | Insert *"the"* before capitalized words mid-sentence |

#### Sub-transformation 5: Confusable Homophones

Substitutes common confused word pairs from a 30-entry lookup table (15 bidirectional pairs). For each word matching a confusable:

$$p_{\text{confuse}} = \lambda \cdot 0.25$$

Pairs include: *its/it's*, *their/there*, *affect/effect*, *then/than*, *lose/loose*, *complement/compliment*, *principal/principle*, *weather/whether*, etc.

#### Sub-transformation 6: Preposition Errors

Replaces correct preposition collocations with incorrect ones from a 15-entry table. For each collocation found in the text:

$$p_{\text{prep}} = \lambda \cdot 0.4$$

Examples: *"interested in"* → *"interested for"*, *"depend on"* → *"depend of"*, *"based on"* → *"based off"*.

#### Sub-transformation 7: Double Comparative Errors

Transforms synthetic comparatives into pleonastic double comparatives from an 8-entry table:

$$p_{\text{comp}} = \lambda \cdot 0.5$$

Examples: *"better"* → *"more better"*, *"faster"* → *"more faster"*.

#### Sub-transformation 8: Hyphenation Errors

Two types of hyphen corruption:

1. **Compound splitting** (26 entries): Removes hyphens from compound modifiers (*"well-known"* → *"well known"*) with probability $\lambda \cdot 0.4$.
2. **Spurious joining** (6 entries): Merges separate words into compounds (*"every day"* → *"everyday"*) with probability $\lambda \cdot 0.3$.

#### Sub-transformation 9: Word-Order Disruption

Active only when $\lambda \geq 0.3$. Within each sentence, performs local adjacent-word swaps:

$$n_{\text{swaps}} = \max\!\left(0,\ \left\lfloor |W| \cdot \lambda \cdot 0.08 \right\rfloor\right)$$

where $|W|$ is the word count of the sentence. Each swap exchanges two adjacent words.

---

### Axis 2: Coherence Disruption

Coherence degradation disrupts the logical flow and coreference chains of a text by shuffling sentence order, while *perfectly preserving* all content, grammar, and vocabulary. This isolates coherence as an independent quality dimension.

#### Algorithm

Given a text tokenized into $n$ sentences $S = (s_1, s_2, \ldots, s_n)$:

1. Compute the number of swap operations:
$$n_{\text{swaps}} = \max\!\left(1,\ \left\lfloor n \cdot \lambda \cdot 0.8 \right\rfloor\right)$$

2. Compute the maximum displacement distance:
$$d_{\max} = \max\!\left(1,\ \left\lfloor n \cdot \lambda \cdot 0.6 \right\rfloor\right)$$

3. For each of the $n_{\text{swaps}}$ iterations:
   - Select a random index $i \sim \mathcal{U}\{0, n-1\}$
   - Select a swap target $j \sim \mathcal{U}\{\max(0, i - d_{\max}),\ \min(n-1, i + d_{\max})\}$
   - Swap $S[i] \leftrightarrow S[j]$

The bounded displacement ensures that at low $\lambda$, sentences drift only slightly from their original positions (local disorder), while at high $\lambda$, sentences can be displaced across the entire text (global disorder).

**Guard**: Texts with $n \leq 2$ sentences are returned unchanged (shuffling is meaningless).

#### Properties

- **Content preservation**: Every original sentence appears exactly once — no additions or deletions.
- **Grammar preservation**: Individual sentences are unmodified.
- **Monotonic severity**: Higher $\lambda$ produces more swaps over larger distances, strictly increasing disorder.

---

### Axis 3: Information Deletion

Information degradation progressively removes content from the text through a five-phase pipeline, targeting increasingly important linguistic constituents. Each phase is probability-gated by $\lambda$, and phases targeting more essential content activate only at higher degradation levels.

$$\text{degrade\_information}(t, \lambda) = \text{clean}\!\left((g_5 \circ g_4 \circ g_3 \circ g_2 \circ g_1)(t, \lambda)\right)$$

#### Phase 1: Parenthetical Deletion

Removes parenthetical expressions (content within parentheses, 5–80 characters) via regex:

$$p_{\text{paren}} = \lambda \cdot 0.6$$

Pattern: `\([^)]{5,80}\)` — captures supplementary details, citations, and clarifications.

#### Phase 2: Modifier Deletion

POS-tags each sentence using NLTK's averaged perceptron tagger and removes adjectives and adverbs (tags: JJ, JJR, JJS, RB, RBR, RBS):

$$p_{\text{mod}} = \lambda \cdot 0.5$$

This strips descriptive richness while leaving core predicate-argument structure intact.

#### Phase 3: Prepositional Phrase Deletion

Activates only when $\lambda \geq 0.3$. Matches prepositional phrases via regex (20 prepositions followed by optional determiner and 1–5 content words):

$$p_{\text{pp}} = (\lambda - 0.2) \cdot 0.25$$

This removes locative, temporal, and instrumental adjuncts.

#### Phase 4: Subordinate Clause Deletion

Matches subordinate clauses introduced by conjunctions and relative pronouns (which, who, whom, that, although, because, since, while, etc.) spanning 10–80 characters:

$$p_{\text{subord}} = \lambda \cdot 0.3$$

Preserving the clause-final punctuation prevents sentence-boundary artifacts.

#### Phase 5: Content Word Deletion

Activates only when $\lambda \geq 0.5$. POS-tags each sentence and removes content words (nouns, verbs — excluding modifiers already handled in Phase 2):

$$p_{\text{content}} = (\lambda - 0.4) \cdot 0.2$$

This is the most aggressive phase, applied only at high degradation levels, stripping core semantic content.

#### Post-Processing

After all phases, artifacts are cleaned:
- Collapsed whitespace (`/  +/` → single space)
- Detached punctuation (`/ ([.,;:!?])/` → attached)
- Empty parentheses, double punctuation removed

---

### Axis 4: Lexical Collapse

Lexical degradation flattens vocabulary diversity by replacing content words with their more common synonyms, simulating the kind of vocabulary simplification that distinguishes basic from sophisticated writing. Domain-specific terminology and technical vocabulary are explicitly protected from substitution.

#### Synonym Generation

Synonym candidates are generated using pre-trained word embeddings (GloVe 840B 300d, 2.2M vectors). For each eligible content word $w$:

1. Retrieve the $k = 30$ nearest neighbors of $w$ by cosine similarity in the embedding space.
2. Filter candidates by:
   - **Cosine threshold**: $\cos(w, w') \geq 0.65$
   - **Frequency threshold**: $\text{freq}(w') \geq 3 \times 10^{-6}$ (via `wordfreq` library, covering the ~100K most common English words)
   - **POS constraint**: candidate must belong to the same broad POS class (noun, verb, adjective) as the source word
   - **Form**: single alphabetic token, length $\geq 4$, not in a stop-word skip list of ~150 function words

The frequency threshold ensures that technical terms (e.g., *"mitochondria"*, *"topography"*) are never substituted — only general-vocabulary content words participate.

Results are cached by $(w, \text{pos\_group})$ so each unique word is looked up exactly once across all 9,000 samples.

#### Two-Pass Replacement Algorithm

The replacement uses a two-pass approach to create controlled vocabulary collapse:

**Pass 1 — Canonical Registration (right-to-left):**

Scanning the text from right to left, the *last* occurrence of each synonym group becomes the canonical form. For each eligible content word:

1. Lemmatize via NLTK's WordNet lemmatizer: $\ell = \text{lemmatize}(w, \text{pos})$
2. Retrieve embedding synonyms $\text{syn}(w)$
3. For all members $m \in \text{syn}(w) \cup \{w\}$ not yet assigned a canonical form, set $\text{canonical}(m) = \ell$

Storing the *base lemma* (not the surface form) as canonical prevents double-inflection errors in Pass 2.

**Pass 2 — Probabilistic Replacement (left-to-right):**

Scanning left to right through content-word positions:

1. Look up $\text{canonical}(w)$ (trying surface form, then lemma)
2. Skip if canonical lemma equals the word's own lemma (no change needed)
3. Skip if the canonical word already appears elsewhere in the same sentence (prevents awkward repetition within a sentence)
4. With probability $\lambda$, replace $w$ with $\text{canonical}(w)$, inflected to match the original word's morphology

$$P(\text{replace } w_i) = \lambda$$

There is no hard ceiling — the replacement probability scales linearly with $\lambda$, guaranteeing a smooth gradient across severity levels.

#### Morphological Transfer

When replacing a word, its surface morphology must be preserved. The function `_transfer_morphology(original, base, tag)` inflects the canonical base lemma to match the original word's POS tag:

| Tag | Transformation |
|-----|---------------|
| NN, JJ, RB | Identity (base form) |
| NNS | Pluralize ($+$s) |
| VBD | Past tense (irregular table or $+$ed, with consonant doubling) |
| VBN | Past participle (irregular table or $+$ed) |
| VBG | Progressive ($-$e$+$ing, with consonant doubling) |
| VBZ | Third person ($+$s or $+$es) |

Irregular past tenses and participles are handled via lookup tables (~50 common irregular verbs). Consonant doubling follows the standard English rule: double the final consonant if the word has one vowel group, ends in a consonant (not w, x, y), and the penultimate character is a vowel.

#### Case Preservation

The replacement preserves the capitalization pattern of the original word:
- Initial capital → capitalize replacement
- All uppercase → uppercase replacement
- Otherwise → lowercase

---

## Deterministic Reproducibility

All randomness in the degradation engine is deterministic. For each sample, a seed is computed via:

$$\text{seed} = \text{MD5}\!\left(\texttt{title} \mathbin\Vert \texttt{axis} \mathbin\Vert \texttt{level} \mathbin\Vert \texttt{rep}\right) \bmod 2^{31}$$

where $\mathbin\Vert$ denotes string concatenation with underscore separators.

This was a deliberate fix over Python's built-in `hash()`, which is session-randomized via `PYTHONHASHSEED` since Python 3.3. MD5 hashing produces identical seeds across machines, operating systems, and Python versions, ensuring bitwise-identical degraded texts on any rerun.

A fresh `random.Random(seed)` instance is created per sample, so degradation of one sample never affects another.

---

## Scoring Protocol

### Prompt Design

Each degraded text is presented to the LLM in isolation with a minimal prompt designed to avoid anchoring effects:

- **System message**: `"Rate the quality of the following text from 0 to 10. Respond with ONLY the number."`
- **User message**: `"Please rate the quality of the following text:\n\n---\n{text}\n---"`

The prompt provides no rubric, no examples, and no reference texts. This is intentional: we are measuring the LLM's *intrinsic* calibration on an unanchored scale, not its ability to follow a grading rubric.

### Models

| Model | Provider | Reasoning | Temperature |
|-------|----------|-----------|-------------|
| GPT-5 mini | OpenAI | Yes (reasoning model) | 1.0 (only supported value) |
| Gemini 3 Flash | Google | Minimal (`thinking_level="minimal"`) | 0.0 |

GPT-5 mini is a reasoning model that internally allocates "thinking tokens" before producing a visible response. With `max_completion_tokens=1024`, approximately 200–320 tokens are used for internal reasoning and ~10 for the visible output. Temperature cannot be set to 0 for this model (only the default of 1.0 is supported).

Gemini 3 Flash uses `thinking_level="minimal"` to reduce reasoning overhead while keeping temperature at 0.0 for maximum determinism.

### Response Parsing

The raw LLM response is parsed by extracting the first integer match 0–10 via:

$$\texttt{re.search(r'\textbackslash b(10|\textbackslash d)\textbackslash b',\ \text{response})}$$

If parsing fails (empty response, non-numeric output), the sample is retried up to 5 times with exponential backoff.

### Checkpointing

Scores are checkpointed to a JSONL file every 5 samples, allowing the scoring run to resume from the last checkpoint on interruption. Completed checkpoints are consolidated into a single JSON output file.

---

## Statistical Analysis

### Dose-Response Curves

For each axis $a$ and model $m$, the mean LLM score is plotted as a function of degradation level $\lambda$:

$$\bar{S}_m(a, \lambda) = \frac{1}{|\mathcal{C}| \cdot R} \sum_{c \in \mathcal{C}} \sum_{r=1}^{R} S_m(c, a, \lambda, r)$$

with 95% confidence intervals computed via the standard error of the mean.

### Axis Sensitivity

Linear regression is fitted per axis to quantify how responsive each model is to degradation:

$$\bar{S}_m(a, \lambda) = \beta_0 + \beta_1 \lambda + \varepsilon$$

The slope $\beta_1$ indicates sensitivity (steeper = more responsive to quality changes), and $R^2$ indicates how well degradation level predicts the LLM score.

### Distribution Analysis

Score distributions are examined for:

- **Range compression**: What fraction of the 0–10 scale is actually used?
- **Boundary avoidance**: How often do scores of 0 or 10 appear?
- **Central tendency**: Where does the mode of the undegraded ($\lambda = 0$) distribution fall?

---

## Repository Structure

```
Condemned_to_B/
├── config.yaml              # Full experiment configuration (articles, axes, levels, models)
├── requirements.txt         # Python dependencies
├── .env.example             # Template for API keys
│
├── src/
│   ├── main.py              # Pipeline orchestrator (runs steps 1–5)
│   ├── corpus.py            # Step 1: Wikipedia FA fetcher
│   ├── degradation.py       # Step 2: Four-axis degradation engine
│   ├── quality.py           # Step 3: Objective quality function Q
│   ├── llm_scoring.py       # Step 4: LLM scoring with checkpointing
│   └── analysis.py          # Step 5: Statistical analysis & figures
│
├── scripts/
│   ├── generate_graphs.py   # Standalone figure generation from raw data
│   └── sanity_check.py      # Quick sanity checks on degradation output
│
├── data/
│   ├── corpus/               # (gitignored — regenerate with --step corpus)
│   │   └── corpus.json      # 150 Wikipedia FA texts
│   ├── degraded/             # (gitignored — regenerate with --step degrade)
│   │   └── degraded_samples.json  # 9,000 degraded samples (~117 MB)
│   └── scores/               # committed
│       └── gpt5_mini_scores.json  # 9,000 GPT-5 mini scores
│
└── output/
    └── figures/              # Generated plots and statistics
        ├── dose_response_per_axis.png
        ├── cross_axis_comparison.png
        ├── score_distribution.png
        ├── boxplots_per_axis.png
        ├── violin_compression.png
        ├── heatmap_scores.png
        ├── undegraded_distribution.png
        ├── axis_sensitivity_slopes.png
        ├── distribution_per_axis.png
        └── summary_statistics.csv
```

---

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

The lexical axis requires pre-trained GloVe word vectors (~2.2 GB):
- Download [GloVe 840B 300d](https://nlp.stanford.edu/data/glove.840B.300d.zip)
- Extract `glove.840B.300d.txt` into the `src/` directory

NLTK resources are downloaded automatically on first run.

### API Keys

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=...
```

### Execution

```bash
# Full pipeline
python -m src.main

# Individual steps
python -m src.main --step corpus     # Fetch Wikipedia articles
python -m src.main --step degrade    # Generate 9,000 degraded samples
python -m src.main --step llm        # Score with LLMs
python -m src.main --step analysis   # Generate figures

# Standalone graph generation (from existing data)
python scripts/generate_graphs.py
```
