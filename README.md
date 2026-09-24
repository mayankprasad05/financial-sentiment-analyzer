# 📈 Financial Sentiment & Automated Summarization Engine

An end-to-end NLP pipeline designed to analyze earnings call transcripts. The system segments raw text into executive remarks and analyst Q&A sections, evaluates domain-specific sentiment using **FinBERT**, extracts abstractive summaries using **BART-Large-CNN**, and presents key visual indicators via a **Streamlit** dashboard.

---

## ✨ Features

* **Segmented Transcript Parsing:** Automatically splits raw transcripts into Executive Remarks and Q&A sessions using SpaCy NLP sentence extraction.
* **Domain-Specific Sentiment Analysis:** Utilizes `ProsusAI/finbert` to calculate net sentiment scores ($P_{\text{positive}} - P_{\text{negative}}$) across individual sentences.
* **Sentiment Shift Tracking:** Measures divergence between prepared executive statements and unscripted Q&A interactions.
* **Abstractive Summarization:** Generates concise executive summaries using `facebook/bart-large-cnn`.
* **Interactive Dashboard:** Visualizes sentiment distributions, narrative flows, and raw sentence-level predictions.
* **Offline Model Benchmarking:** Includes an evaluation pipeline (`eval.py`) to measure performance against standard financial datasets (Financial PhraseBank and ROUGE metrics).

---

## 📁 Repository Structure

```text
.
├── src/
│   ├── logger_config.py     # Application-wide logging configuration
│   ├── modeling.py          # FinBERT and BART model class wrappers
│   ├── pipeline.py          # Pipeline orchestration logic
│   └── preprocessing.py     # SpaCy transcript segmentation and extraction
├── eval_results/            # Generated benchmark metrics and visual artifacts
├── app.py                   # Interactive Streamlit frontend UI
├── eval.py                  # Offline evaluation suite (F1-score, ROUGE metrics)
├── requirements.txt         # Project dependencies
└── README.md

```
---

## 🚀 Getting Started
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

## 🖥️ Running the Application
### Launch the Streamlit web interface locally:

```bash
streamlit run app.py
```
The application will open in your browser at http://localhost:8501.

---

## 🧪 Model Evaluation
To run the offline benchmark suite and generate model metrics (Financial PhraseBank evaluation for FinBERT and ROUGE scores for BART):

```bash
python eval.py
```

This updates performance reports and saves artifacts (such as finbert_confusion_matrix.png and summary_metrics.json) into the eval_results/ directory.

---

## 📄 License
This project is licensed under the MIT License.
