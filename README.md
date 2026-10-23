# Financial Sentiment Analysis Engine NLP Dashboard

An end-to-end financial NLP application built in Python that automatically fetches corporate earnings call transcripts, extracts granular sentence-level sentiment metrics using domain-adapted transformer models, generates concise executive summaries, and visualizes financial telemetry through an interactive Streamlit dashboard.

---

## Key Features

* **Live Earnings Call Integration:** Fetches quarterly earnings call transcripts by ticker symbol using the `earningscall` API with an integrated manual text paste fallback.
* **Domain-Specific Sentiment Analysis:** Utilizes `ProsusAI/finbert` (a BERT model fine-tuned on financial corpora) to classify sentence-level sentiment (Positive, Negative, Neutral) and compute net aggregate sentiment scores.
* **Abstractive Summarization:** Employs `facebook/bart-large-cnn` via PyTorch and Hugging Face Transformers (`AutoModelForSeq2SeqLM`) to distill lengthy transcripts into executive summaries.
* **Sentence Tokenization:** Uses SpaCy (`en_core_web_sm`) for natural language boundary parsing.
* **Interactive Financial Telemetry:** Visualizes sentence sentiment distribution, score distributions, and narrative trajectory using Plotly charts inside a Streamlit web app.
* **Modular Codebase:** Organized using Object-Oriented Design (OOD) principles with distinct separation between data fetchers, preprocessing pipelines, model inference, and UI rendering.

---

## Architecture Overview

```text
financial-sentiment-analyzer/
│
├── app.py                  # Streamlit web dashboard & UI layout
├── requirements.txt        # Core dependencies & versions
├── .gitignore              # Git exclusion rules for environments/cache
├── README.md               # Project documentation
│
└── src/                    # Core source code modules
    ├── __init__.py
    ├── fetcher.py          # Fetches earnings call transcripts via API
    ├── preprocessing.py    # SpaCy NLP sentence tokenization & cleaning
    ├── modeling.py         # FinBERT sentiment classification engine
    ├── summarizer.py       # BART abstractive summarization engine
    └── pipeline.py         # Orchestrates fetch, preprocess, model, & summary

```

## Getting Started
### Prerequisites
Ensure you have Python 3.9+ installed.

### Installation
Clone the Repository:

``` bash
git clone [https://github.com/your-username/financial-sentiment-analyzer](https://github.com/your-username/financial-sentiment-analyzer)
cd financial-sentiment-engine
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

## Running the Application
### Launch the Streamlit web interface locally:

```bash
streamlit run app.py
```
The application will open in your browser at http://localhost:8501.

The app now runs two analysis paths side by side:

- **FinBERT + BART pipeline:** sentence-level sentiment and abstractive summarization
- **Loughran-McDonald word-level scoring:** category composition, category flow, and sentence-level LM scores for Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, ModalStrong, and ModalWeak

---

## Model Evaluation

The project now has two evaluation paths:

- **Sentence-level FinBERT evaluation:** uses `data/loughran_mcdonald.parquet` as the local dataset backing and is run with:

  ```bash
  python eval.py
  ```

  This updates performance reports and saves artifacts (such as finbert_confusion_matrix.png and summary_metrics.json) into the eval_results/ directory.

- **Word-level Loughran-McDonald evaluation:** uses the Master Dictionary to score text by financial sentiment categories:

  ```bash
  python eval_lm_word_level.py
  ```

  Category schema: Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, ModalStrong, ModalWeak.

  It also supports fetching transcripts with the same earningscall flow as the rest of the app:

  ```bash
  python eval_lm_word_level.py --ticker AAPL --year 2024 --quarter 3
  ```


If you have the CSV file (`Loughran-McDonald_MasterDictionary_1993-2025.csv`), convert it once:

```bash
python -c "import pandas as pd; from pathlib import Path; df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv'); df.to_parquet('data/loughran_mcdonald.parquet')"
```

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
wiring check, not a benchmark result. Results are written to
`eval_results/summarization_eval.json`.

---

## License
This project is licensed under the MIT License.
