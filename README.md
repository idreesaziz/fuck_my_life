# Condemned to B

**Nobody Gets an A: Distributional Bias in LLM Text Scoring**

Research pipeline investigating score compression bias in LLM text evaluation — the systematic tendency to overrate poor texts and underrate excellent ones, compressing everything toward the middle.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# 2. Set up API keys
cp .env.example .env
# Edit .env with your OpenAI key and two Google API keys (for free-tier throughput)

# 3. Run the full pipeline
python -m src.main

# Or run individual steps
python -m src.main --step corpus
python -m src.main --step degrade
python -m src.main --step quality
python -m src.main --step llm
python -m src.main --step analysis
```

## Experiment Scale

- **150 Wikipedia Featured Articles** across 25 topic categories (physics, biology, history, art, computing, …)
- **4 degradation axes** × **9 intensity levels** × **3 samples** = **16,200 degraded samples**
- **2 LLM models**: GPT-4.1 nano (OpenAI) + Gemini 2.5 Flash-Lite (Google)
- **2 conditions**: isolated + batched = **~19,440 API calls per model**
- **Dual Google API key rotation** for free-tier throughput doubling

## Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/corpus.py` | Fetches 150 Wikipedia Featured Articles as Band 9 baselines |
| 2 | `src/degradation.py` | Corrupts texts along 4 independent axes at 9 intensity levels |
| 3 | `src/quality.py` | Computes objective quality scores using model-free metrics |
| 4 | `src/llm_scoring.py` | Sends texts to GPT-4.1 nano & Gemini 2.5 Flash-Lite (isolated & batched) |
| 5 | `src/analysis.py` | Generates figures and statistical summaries |

## Degradation Axes

| Axis | Method | What It Breaks |
|------|--------|---------------|
| **Grammar** | Character-level noise (keyboard typos, swaps) via nlpaug | Spelling, punctuation, syntax |
| **Coherence** | Local word-order shuffling within sliding windows | Sentence flow, readability |
| **Information** | Random sentence deletion | Content completeness |
| **Lexical** | Synonym collapse + vocabulary flattening | Vocabulary diversity |

Each axis is degraded independently at levels `[0.0, 0.1, 0.2, ..., 0.8]` with 3 samples per level.

## Quality Function

$$Q = \text{grammar}^{w_1} \times \text{coherence}^{w_2} \times (1 - \text{deletion})^{w_3} \times \text{lexical}^{w_4}$$

- **Multiplicative** — any component hitting zero kills the score
- **Weights** configurable in `config.yaml` (default: all 1.0)
- Grounded in IELTS criteria taxonomy and Grice's Maxims

### Component Metrics (all model-free)

| Component | Metric | Tool |
|-----------|--------|------|
| Grammar | 1 − (error_count / word_count) | LanguageTool |
| Coherence | Cosine similarity of sentence embeddings vs original | SentenceTransformers |
| Information | Character length ratio (degraded / original) | Direct computation |
| Lexical | Type-token ratio relative to original | Direct computation |

## Research Questions

1. Do LLMs systematically compress scores toward the center?
2. Is compression consistent across models or model-specific?
3. Does batched vs. isolated rating affect compression?
4. Are LLMs differentially blind to specific degradation types?

## Outputs

```
results/
├── figures/
│   ├── dose_response_{axis}.png     # Per-axis degradation curves
│   ├── cross_axis_{model}.png       # Cross-axis sensitivity per model
│   ├── batched_vs_isolated.png      # Condition comparison
│   └── q_vs_llm_scurve.png         # The S-curve (core finding)
├── merged_data.csv                   # All data points merged
├── summary_statistics.csv            # Compression ratios, correlations
└── axis_sensitivity.csv              # Per-axis regression slopes
```

## Configuration

All parameters are in `config.yaml`:
- Corpus articles and size limits
- Degradation levels, axes, samples per level
- Quality function weights
- LLM models, conditions, repetitions
- Output directories

## Tool Stack

| Category | Tools |
|----------|-------|
| Noise generation | nlpaug, TextAttack |
| NLP processing | spaCy, NLTK |
| Grammar scoring | LanguageTool |
| Semantic similarity | SentenceTransformers (all-MiniLM-L6-v2) |
| Readability | textstat |
| Statistics | NumPy, SciPy, scikit-learn, pandas |
| Visualization | matplotlib, seaborn |
| LLM APIs | OpenAI, Google GenAI |

## API Keys & Cost

| Model | Provider | Cost per 1M tokens (in/out) | Free Tier |
|-------|----------|----------------------------|-----------|
| GPT-4.1 nano | OpenAI | $0.10 / $0.40 | No |
| Gemini 2.5 Flash-Lite | Google | $0.10 / $0.40 | Yes (15 RPM/key) |

The pipeline supports **dual Google API key rotation** (`GOOGLE_API_KEY_1` + `GOOGLE_API_KEY_2`) to double free-tier throughput to ~30 RPM. Configure keys in `.env`.

**Estimated cost**: ~$1–2.50 total (OpenAI only; Gemini on free tier = $0).
