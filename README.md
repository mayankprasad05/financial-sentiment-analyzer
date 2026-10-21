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

---
```
## Model Evaluation
To run the offline benchmark suite and generate model metrics (Financial PhraseBank evaluation for FinBERT and ROUGE scores for BART):

```bash
python eval.py


This updates performance reports and saves artifacts (such as finbert_confusion_matrix.png and summary_metrics.json) into the eval_results/ directory.

---
```
## License
This project is licensed under the MIT License.
