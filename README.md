# Financial Sentiment Analysis Engine NLP Dashboard

An end-to-end financial NLP application built in Python that automatically fetches corporate earnings call transcripts, extracts granular sentence-level sentiment metrics using domain-adapted transformer models, generates concise executive summaries, and visualizes financial telemetry through an interactive Streamlit dashboard.

---

## Key Features

* **Live Earnings Call Integration:** Fetches quarterly earnings call transcripts by ticker symbol using the `earningscall` API with an integrated manual text paste fallback.
* **Domain-Specific Sentiment Analysis:** Utilizes `ProsusAI/finbert` (a BERT model fine-tuned on financial corpora) to classify sentence-level sentiment (Positive, Negative, Neutral) and compute net aggregate sentiment scores.
* **Section-Aware Abstractive Summarization:** Employs `facebook/bart-large-cnn` via PyTorch and Hugging Face Transformers (`AutoModelForSeq2SeqLM`) to distill lengthy transcripts into executive summaries, after stripping safe-harbor boilerplate and separating prepared remarks from Q&A.
* **Word-Level Loughran-McDonald Scoring:** Scores text against the Loughran-McDonald Master Dictionary across eight financial categories (Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, ModalStrong, ModalWeak).
* **Fine-Tuning and Evaluation Harness:** Trains FinBERT on Financial PhraseBank and compares it against the off-the-shelf model on an identical held-out split, alongside a summarization harness that measures the guided pipeline against a legacy truncation baseline.
* **Sentence Tokenization:** Uses SpaCy (`en_core_web_sm`) for natural language boundary parsing.
* **Interactive Financial Telemetry:** Visualizes sentence sentiment distribution, score distributions, and narrative trajectory using Plotly charts inside a Streamlit web app.
* **Modular Codebase:** Organized using Object-Oriented Design (OOD) principles with distinct separation between data fetchers, preprocessing pipelines, model inference, and UI rendering.

---

## Architecture Overview

```text
financial-sentiment-analyzer/
│
├── app.py                      # Streamlit web dashboard & UI layout
├── eval.py                     # Sentiment evaluation (off-the-shelf vs fine-tuned)
├── eval_lm_word_level.py       # Word-level Loughran-McDonald evaluation
├── eval_summarization.py       # Summarization harness (guided vs legacy truncation)
├── finetune_sentiment.py       # FinBERT fine-tuning on Financial PhraseBank
├── requirements.txt            # Core dependencies & versions
├── .gitignore                  # Git exclusion rules for environments/cache
├── README.md                   # Project documentation
│
├── data/                       # Local dataset snapshots (regenerated per setup)
│   ├── financial_phrasebank.parquet
│   └── loughran_mcdonald.parquet
├── eval_results/               # Generated reports & confusion matrices
├── models/                     # Saved fine-tuned model artifacts
│
└── src/                        # Core source code modules
    ├── __init__.py
    ├── fetcher.py              # Fetches earnings call transcripts via API
    ├── preprocessing.py        # SpaCy NLP sentence tokenization & cleaning
    ├── modeling.py             # FinBERT sentiment classification engine
    ├── loughran_mcdonald.py    # LM Master Dictionary scorer (8 categories)
    ├── summarizer.py           # Section-aware, LM-guided BART summarizer
    ├── logger_config.py        # Logging configuration (file + console levels)
    └── pipeline.py             # Orchestrates fetch, preprocess, model, & summary

```

## Getting Started

### Prerequisites
Ensure you have **Python 3.10 or newer** installed. The evaluation scripts use modern
type syntax (`str | None`), which is not valid on 3.9 and earlier.

### Installation
Clone the Repository:

```bash
git clone https://github.com/your-username/financial-sentiment-analyzer
cd financial-sentiment-analyzer
```

### Create and Activate a Virtual Environment:

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Download SpaCy Language Model:

```bash
python -m spacy download en_core_web_sm
```

---

## Datasets

Two local snapshots are required. Both live under `data/`, and the commands below rebuild
them from source.

| File | Used by | Purpose |
|---|---|---|
| `data/loughran_mcdonald.parquet` | app, summarizer, `eval_lm_word_level.py`, `eval_summarization.py` | Loughran-McDonald Master Dictionary |
| `data/financial_phrasebank.parquet` | `eval.py`, `finetune_sentiment.py` | Labelled sentiment corpus |

`data/loughran_mcdonald.parquet` is **not only an evaluation asset**. `TranscriptSummarizer`
loads the LM scorer to rank sentences during chunk selection, so the Streamlit app cannot
produce a summary without it.

### Loughran-McDonald Master Dictionary

Convert the master CSV once:

```bash
python -c "import pandas as pd; df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv'); df.to_parquet('data/loughran_mcdonald.parquet')"
```

### Financial PhraseBank

Build the labelled corpus from the original archive:

```bash
python - <<'PY'
import zipfile, pandas as pd
from huggingface_hub import hf_hub_download

# Sentences_50Agree: 4,846 sentences, kept when at least half the annotators agree.
path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
raw = zipfile.ZipFile(path).read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin-1")

labels = {"negative": 0, "neutral": 1, "positive": 2}
rows = []
for line in raw.splitlines():
    if line.strip():
        sentence, _, label = line.rpartition("@")
        rows.append({"sentence": sentence.strip(), "label": labels[label.strip()]})

pd.DataFrame(rows).to_parquet("data/financial_phrasebank.parquet")
print("saved 4,846 sentences with columns: sentence, label")
PY
```

The resulting file has two columns — `sentence` and `label` (0 = negative, 1 = neutral,
2 = positive) — which is what `eval.py` and `finetune_sentiment.py` both expect.

---

## Running the Application

### Launch the Streamlit web interface locally:

```bash
streamlit run app.py
```
The application will open in your browser at http://localhost:8501.

The app runs two analysis paths side by side, each with its own checkbox:

- **FinBERT + BART pipeline:** sentence-level sentiment and abstractive summarization
- **Loughran-McDonald word-level scoring:** category composition, category flow, and sentence-level LM scores for Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, ModalStrong, and ModalWeak

A full earnings call takes 1-2 minutes to summarize on CPU.

---

## Model Evaluation

The project has three evaluation scripts: `eval.py` for sentiment, `eval_lm_word_level.py`
for lexical scoring, and `eval_summarization.py` for summaries.

### Sentence-Level Sentiment

```bash
python eval.py
```

`eval.py` takes **no command-line arguments**. It reads `data/financial_phrasebank.parquet`,
shuffles with a fixed seed, and builds a 3,600 / 400 / 800 train / validation / test split.
Evaluation is on the 800 held-out sentences only.

Output:

- per-class precision, recall and F1 for negative / neutral / positive
- accuracy, macro precision, macro recall and macro F1
- BART ROUGE-1/2/L on the small built-in summarization benchmark, in the same run

Macro F1 is the headline number rather than accuracy, because 62% of the test split is
neutral: a constant "neutral" predictor would score 62% accuracy but only about 0.25 macro
F1.

Artifacts written to `eval_results/`:

| File | Contents |
|---|---|
| `finbert_confusion_matrix.png` | off-the-shelf confusion matrix |
| `finbert_financial_sentiment_ft_confusion_matrix.png` | fine-tuned confusion matrix (only when a checkpoint exists) |
| `sentiment_comparison.json` | accuracy and macro precision/recall/F1 for both models, plus deltas |

If `models/finbert_financial_sentiment_ft/` exists, the fine-tuned model is included
automatically and a side-by-side table is printed — see [Fine-Tuning](#fine-tuning). A run
with the comparison enabled takes roughly 4 minutes on CPU, versus 2 minutes for the
off-the-shelf model alone.

### Word-Level Loughran-McDonald

```bash
python eval_lm_word_level.py
```

Prints per-sentence and aggregated category shares against the eight-category schema
(Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, ModalStrong,
ModalWeak). This is a dictionary lookup, so there is no accuracy figure — the output is a
category composition, not a prediction.

It also supports fetching transcripts with the same earningscall flow as the rest of the app:

```bash
python eval_lm_word_level.py --ticker AAPL --year 2024 --quarter 3
```

### Summarization

See [Summarization Evaluation](#summarization-evaluation) below.

---

## Fine-Tuning

`finetune_sentiment.py` fine-tunes `ProsusAI/finbert` on Financial PhraseBank and saves a
checkpoint that the rest of the project loads directly.

```bash
python finetune_sentiment.py
python finetune_sentiment.py --epochs 5 --batch-size 32
python finetune_sentiment.py --cpu
```

| Option | Default | Meaning |
|---|---|---|
| `--sample-size` | 4800 | Rows taken from the corpus before splitting |
| `--test-size` | 800 | Held-out sentences; must match `eval.py` |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--max-length` | 64 | Token truncation length |
| `--seed` | 42 | Seed for the shuffle and the split |
| `--output-dir` | `models/finbert_financial_sentiment_ft` | Where the checkpoint is written |
| `--cpu` | off | Force CPU training |

The split is imported from `eval.load_phrasebank_test_split`, so the model trains on the same
3,600 sentences and is measured on exactly the 800 that `eval.py` scores. Using a separate
split here would leak training sentences into the test set and inflate the reported accuracy.

Outputs under `models/finbert_financial_sentiment_ft/`: model weights, tokenizer, a
`config.json` carrying `id2label`, and `fine_tune_metrics.json` with the test metrics, split
sizes and hyperparameters. Training takes about 42 minutes on CPU (3 epochs, batch size 16,
`max_length=64`).

**Expected result: fine-tuning does not improve the score here.** On this project's run the
fine-tuned model reached 0.8538 accuracy / 0.8329 macro F1 against the off-the-shelf model's
0.8825 / 0.8723 — a drop of roughly 3 points. That is expected rather than a bug:
`ProsusAI/finbert` was itself trained on Financial PhraseBank, so fine-tuning on a subset of
the same corpus and evaluating on the same distribution cannot beat it. A genuine gain would
require out-of-domain data such as earnings-call sentences.

---

## Summarization Pipeline

`src/summarizer.py` holds the project's single summarizer, `TranscriptSummarizer`. It no
longer truncates the transcript to the model window and summarizes the opening 750
words (which, for an earnings call, is the safe-harbor disclaimer). It runs five stages:

1. **Boilerplate stripping** - drops forward-looking-statement / safe-harbor sentences and
   opening call logistics ("my name is...", "this call is being recorded"), capped at 35%
   of sentences so a heuristic can never gut the transcript.
2. **Section detection** - separates prepared remarks from the Q&A session. The opening
   agenda ("after that, we'll open the call to questions") is ignored, so the split lands on
   the real hand-over rather than the first mention of questions.
3. **Chunking** - token-budgeted chunks (default 900 tokens) on sentence boundaries.
4. **LM-guided selection** - each sentence is scored with the Loughran-McDonald dictionary
   (positive/negative density) plus numeric density (revenue/margin/guidance figures), and
   penalized for litigious/constraining/uncertainty hedging. Chunks are selected within a
   4800-token input budget: prepared remarks get up to 60% of it (the Q&A is usually several
   times longer, while the prepared block carries the results and guidance), the Q&A gets the
   rest, and every section keeps at least one chunk.
5. **Sectioned map-reduce generation** - each selected chunk is summarized, then the chunk
   summaries of each section are reduced separately and labelled ("Prepared remarks: ..." /
   "Q&A: ..."). A single global reduce was measured to collapse onto the first topic and drop
   guidance and the entire Q&A, because BART is trained on news leads; per-section synthesis
   keeps both halves in the output.

Behavior is configured through `SummarizerConfig`; `legacy_summarizer_config()` reproduces
the old truncation behavior for comparison. The pipeline returns `summary_meta` alongside
the summary text (boilerplate removed, chunks used, tokens fed to the model, compression
ratio, per-section coverage), which the app displays under the Executive Summary tab along
with the boilerplate sentences that were dropped.

Expect a full earnings call to take 1-2 minutes on CPU: it is one BART pass per selected
chunk (typically 4-6) plus one synthesis pass per section, instead of a single truncated
pass. Short transcripts still summarize in a few seconds.

### Summarization Evaluation

```bash
python eval_summarization.py
python eval_summarization.py --dataset data/summarization_benchmark.jsonl --ablations
python eval_summarization.py --model sshleifer/distilbart-cnn-6-6 --max-examples 5
```

The harness compares the guided pipeline against legacy truncation (and, with
`--ablations`, strip-only and salience-only variants) and reports:

- ROUGE-1/2/L against reference summaries
- **LM sentiment retention** - cosine similarity between transcript and summary LM category shares; reported as undefined (with a coverage figure) when a summary contains no dictionary words at all, because "not measurable" is not the same as "no retention"
- **Disclaimer suppression** - Litigious/Constraining hits per 1,000 tokens in the summary versus the transcript
- **Hedging density of the model input** - what each arm actually fed the model, which is where boilerplate stripping shows up
- compression ratio, input tokens, chunks used, and latency

Benchmark format is JSONL (or a JSON list), one object per line:

```json
{"transcript": "...", "reference": "human written summary"}
```

Without `--dataset` the harness runs a two-example built-in smoke benchmark, which is a
wiring check, not a benchmark result: both examples fit in a single chunk and contain no
boilerplate, so every delta is exactly zero. Results are written to
`eval_results/summarization_eval.json`.

---

## License
This project is licensed under the MIT License.
