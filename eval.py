import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from datasets import load_dataset
import evaluate

from src.modeling import FinancialSentimentAnalyzer, ExecutiveSummarizer
from src.logger_config import setup_logging

logger = logging.getLogger(__name__)

# Directory setup for artifacts
ARTIFACTS_DIR = Path("eval_results")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def evaluate_finbert(analyzer: FinancialSentimentAnalyzer, sample_size: int = 300) -> Dict[str, Any]:
    """
    Evaluates FinBERT against the Financial PhraseBank dataset.
    Computes Precision, Recall, Macro-F1, and generates a Confusion Matrix plot.
    """
    logger.info("--- Starting FinBERT Sentiment Analysis Evaluation ---")
    
    # Financial PhraseBank uses 0: negative, 1: neutral, 2: positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    logger.info("Loading Financial PhraseBank dataset from Hugging Face...")
    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        split=f"train[:{sample_size}]"
    )

    sentences = dataset["sentence"]
    y_true_labels = [label_map[label_id] for label_id in dataset["label"]]

    logger.info(f"Running inference on {len(sentences)} test samples...")
    results = analyzer.analyze_sentences(sentences, batch_size=16)
    y_pred_labels = [res["label"] for res in results]

    # Metrics calculation
    target_names = ["negative", "neutral", "positive"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=target_names, average="macro"
    )

    logger.info("Calculating detailed classification report...")
    report = classification_report(
        y_true_labels, y_pred_labels, labels=target_names, output_dict=True
    )
    
    # Print formatted metrics to console
    print("\n================ FINBERT EVALUATION METRICS ================")
    print(classification_report(y_true_labels, y_pred_labels, labels=target_names))
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {f1:.4f}")
    print("============================================================")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=target_names)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title("FinBERT Confusion Matrix (Financial PhraseBank)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    cm_path = ARTIFACTS_DIR / "finbert_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    return {
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
        "classification_report": report
    }


def evaluate_bart(summarizer: ExecutiveSummarizer) -> Dict[str, float]:
    """
    Evaluates BART-Large-CNN on financial text summaries using ROUGE metrics.
    Uses reference earnings call snippets with ground-truth human summaries.
    """
    logger.info("--- Starting BART-Large-CNN Summarization Evaluation ---")

    # Benchmark test pairs (Financial transcript excerpts -> Ground Truth human summaries)
    eval_benchmark = [
        {
            "transcript": (
                "We delivered exceptional revenue growth of 24% year-over-year, hitting $1.2B in Q4. "
                "Gross margins expanded by 150 basis points due to operational efficiency and strong Enterprise adoption. "
                "However, supply chain bottlenecks created mild cost headwinds in logistics. "
                "The inflation impact on freight cost is real and lingering, though demand remains robust."
            ),
            "reference": (
                "Q4 revenue grew 24% YoY to $1.2B with gross margins expanding 150 bps. "
                "Despite strong demand, lingering supply chain inflation and logistics costs presented mild headwinds."
            )
        },
        {
            "transcript": (
                "Total net sales increased 12% to $850 million compared to the prior year period. "
                "Net income was $95 million, down from $110 million due to elevated operating expenses "
                "and R&D investments in next-generation platforms. Free cash flow generation was strong at $140 million."
            ),
            "reference": (
                "Net sales increased 12% to $850M, but net income dropped to $95M driven by higher operating "
                "and R&D expenses. Free cash flow stood solid at $140M."
            )
        }
    ]

    logger.info("Loading ROUGE evaluation metric...")
    rouge_metric = evaluate.load("rouge")

    predictions = []
    references = []

    logger.info("Generating predictions using ExecutiveSummarizer...")
    for idx, item in enumerate(eval_benchmark, start=1):
        generated_summary = summarizer.summarize(item["transcript"], max_length=100, min_length=20)
        predictions.append(generated_summary)
        references.append(item["reference"])
        logger.debug(f"Sample {idx} - Generated: {generated_summary}")

    # Compute ROUGE Scores
    scores = rouge_metric.compute(predictions=predictions, references=references)

    print("\n================ BART ROUGE EVALUATION METRICS ================")
    print(f"ROUGE-1 : {scores['rouge1']:.4f}")
    print(f"ROUGE-2 : {scores['rouge2']:.4f}")
    print(f"ROUGE-L : {scores['rougeL']:.4f}")
    print("===============================================================")

    return {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"]
    }


def main():
    setup_logging(log_level=logging.INFO)
    logger.info("Starting Offline Model Evaluation Suite")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluation executing on hardware device: {device}")

    # 1. FinBERT Sentiment Model Evaluation
    try:
        sentiment_analyzer = FinancialSentimentAnalyzer(device=device)
        finbert_metrics = evaluate_finbert(sentiment_analyzer, sample_size=300)
    except Exception as e:
        print(f"\n❌ FinBERT Evaluation Failed: {e}\n")  # Surface error to terminal
        logger.error(f"FinBERT evaluation failed: {e}", exc_info=True)

    # 2. BART Summarization Model Evaluation
    try:
        summarizer = ExecutiveSummarizer(device=device)
        bart_metrics = evaluate_bart(summarizer)
    except Exception as e:
        print(f"\n❌ BART Evaluation Failed: {e}\n")
        logger.error(f"BART evaluation failed: {e}", exc_info=True)

    logger.info("Model evaluation pipeline finished. Results saved in 'eval_results/'.")