import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Must be set before transformers / huggingface_hub are imported, otherwise progress
# bars and hub warnings get written into the terminal alongside the metric tables.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from datasets import load_dataset
from src.modeling import FinancialSentimentAnalyzer
from src.summarizer import TranscriptSummarizer
from src.logger_config import setup_logging, silence_third_party

logger = logging.getLogger(__name__)

# Directory setup for artifacts
ARTIFACTS_DIR = Path("eval_results")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Financial PhraseBank label space: 0: negative, 1: neutral, 2: positive
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TARGET_NAMES = ["negative", "neutral", "positive"]


def load_phrasebank_test_split(
    sample_size: int = 1200,
    test_size: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Load Financial PhraseBank from a local parquet snapshot and return a fixed
    train/val/test split.

    This is shared by both fine-tuning and comparison evaluation so the two runs
    are directly comparable on the same held-out test data.
    """
    logger.info(
        "Loading Financial PhraseBank (sentences_50agree) sample_size=%d, test_size=%d",
        sample_size,
        test_size,
    )
    parquet_path = Path("data/financial_phrasebank.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            "Local dataset snapshot not found at data/financial_phrasebank.parquet.\n\n"
            "Download the Financial PhraseBank once from HuggingFace and save it as a parquet:\n\n"
            "    python - <<'PY'\n"
            "    from datasets import load_dataset\n"
            "    load_dataset('financial_phrasebank', split='sentences_50agree', trust_remote_code=True)\n"
            "        .to_parquet('data/financial_phrasebank.parquet')\n"
            "    print('saved to data/financial_phrasebank.parquet')\n"
            "    PY\n\n"
            "The current capstone setup uses a one-time local parquet copy instead.\n"
        )

    df = pd.read_parquet(parquet_path)
    # Shuffle before slicing so the capped sample is representative of the whole file
    # (the raw PhraseBank file is not ordered randomly).
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.head(min(len(df), max(sample_size, test_size + 50)))

    logger.info("Loaded Financial PhraseBank snapshot with %d rows", len(df))

    from datasets import Dataset

    ds = Dataset.from_pandas(df)
    actual_test_size = min(test_size, max(1, int(len(ds) * 0.2)))
    ds = ds.train_test_split(test_size=actual_test_size, seed=seed)

    train_split = ds["train"]
    test_split = ds["test"]

    train_split = train_split.train_test_split(
        test_size=max(1, int(len(train_split) * 0.1)), seed=seed
    )

    return {
        "train": train_split["train"],
        "val": train_split["test"],
        "test": test_split,
    }


def artifact_name(model_source: str) -> str:
    """Filesystem-safe stem for a model source.

    Handles both Hub ids ("ProsusAI/finbert") and local paths
    ("models/finbert_financial_sentiment_ft" on Windows, with backslashes), so the
    confusion-matrix filename never contains a path separator.
    """
    stem = Path(str(model_source)).name or str(model_source)
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return safe.strip("_") or "model"


def evaluate_model_on_test(
    analyzer: FinancialSentimentAnalyzer,
    test_split,
    batch_size: int = 16,
    max_length: int = 512,
    model_name: str = "model",
    sentence_column: str = "sentence",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Evaluate one sentiment model on the same test split and return metrics + artifacts.
    """
    logger.info("Evaluating %s on %d test samples...", model_name, len(test_split))

    sentences = list(test_split[sentence_column])
    y_true = [LABEL_MAP[label_id] for label_id in test_split[label_column]]

    results = analyzer.analyze_sentences(sentences, batch_size=batch_size, max_length=max_length)
    y_pred = [res["label"] for res in results]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_NAMES, average="macro", zero_division=0
    )
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    report = classification_report(
        y_true, y_pred, labels=TARGET_NAMES, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_NAMES)

    print(
        f"\n================ {model_name.upper()} EVALUATION METRICS ================"
    )
    print(classification_report(y_true, y_pred, labels=TARGET_NAMES, zero_division=0))
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {f1:.4f}")
    print("===============================================================")

    cm_path = ARTIFACTS_DIR / f"{model_name.replace('/', '_')}_confusion_matrix.png"
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.title(f"{model_name} Confusion Matrix (Financial PhraseBank)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    logger.info("Saved confusion matrix for %s to %s", model_name, cm_path)

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_sentiment_comparison(
    off_the_shelf_source: str = "ProsusAI/finbert",
    fine_tuned_source: Optional[str] = None,
    sample_size: int = 1200,
    test_size: int = 200,
    device: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Direct apples-to-apples comparison of off-the-shelf vs fine-tuned FinBERT.
    Both models are evaluated on the same held-out test split.
    """
    logger.info("--- Starting off-the-shelf vs fine-tuned sentiment comparison ---")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    splits = load_phrasebank_test_split(sample_size=sample_size, test_size=test_size)
    test_split = splits["test"]

    results: Dict[str, Any] = {}

    # 1. Off-the-shelf model
    oos_analyzer = FinancialSentimentAnalyzer(model_source=off_the_shelf_source, device=device)
    oos_name = artifact_name(off_the_shelf_source)
    results["off_the_shelf"] = evaluate_model_on_test(
        oos_analyzer,
        test_split,
        batch_size=batch_size,
        max_length=max_length,
        model_name=oos_name,
    )

    # 2. Fine-tuned model (optional)
    ft_name = None
    if fine_tuned_source:
        ft_analyzer = FinancialSentimentAnalyzer(model_source=fine_tuned_source, device=device)
        ft_name = artifact_name(fine_tuned_source)
        ft_sentences = list(test_split["sentence"])
        ft_y_true = [LABEL_MAP[label_id] for label_id in test_split["label"]]
        ft_results = ft_analyzer.analyze_sentences(ft_sentences, batch_size=batch_size, max_length=max_length)
        ft_y_pred = [res["label"] for res in ft_results]
        ft_precision, ft_recall, ft_f1, _ = precision_recall_fscore_support(
            ft_y_true, ft_y_pred, labels=TARGET_NAMES, average="macro", zero_division=0
        )
        ft_accuracy = float(np.mean(np.array(ft_y_true) == np.array(ft_y_pred)))
        ft_report = classification_report(
            ft_y_true, ft_y_pred, labels=TARGET_NAMES, output_dict=True, zero_division=0
        )
        ft_cm = confusion_matrix(ft_y_true, ft_y_pred, labels=TARGET_NAMES)
        ft_cm_path = ARTIFACTS_DIR / f"{ft_name}_confusion_matrix.png"
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            ft_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_NAMES,
            yticklabels=TARGET_NAMES,
        )
        plt.title(f"{ft_name} Confusion Matrix (Financial PhraseBank)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(ft_cm_path)
        plt.close()
        logger.info("Saved confusion matrix for %s to %s", ft_name, ft_cm_path)

        print(
            f"\n================ FINETUNED EVALUATION METRICS ================"
        )
        print(classification_report(ft_y_true, ft_y_pred, labels=TARGET_NAMES, zero_division=0))
        print(f"Accuracy        : {ft_accuracy:.4f}")
        print(f"Macro Precision : {ft_precision:.4f}")
        print(f"Macro Recall    : {ft_recall:.4f}")
        print(f"Macro F1-Score  : {ft_f1:.4f}")
        print("===============================================================")

        results["fine_tuned"] = {
            "accuracy": ft_accuracy,
            "macro_precision": ft_precision,
            "macro_recall": ft_recall,
            "macro_f1": ft_f1,
            "classification_report": ft_report,
            "confusion_matrix": ft_cm.tolist(),
            "y_true": ft_y_true,
            "y_pred": ft_y_pred,
        }

    # 3. Side-by-side summary
    if ft_name:
        oos_f1 = results["off_the_shelf"]["macro_f1"]
        ft_f1 = results["fine_tuned"]["macro_f1"]
        oos_acc = results["off_the_shelf"]["accuracy"]
        ft_acc = results["fine_tuned"]["accuracy"]

        print(
            "\n================ SIDE-BY-SIDE COMPARISON ================"
        )
        print(f"{'Metric':<22} {'Off-the-shelf':>16} {'Fine-tuned':>16} {'Delta':>10}")
        print("-" * 64)
        print(f"{'Accuracy':<22} {oos_acc:>16.4f} {ft_acc:>16.4f} {ft_acc - oos_acc:>+10.4f}")
        print(f"{'Macro Precision':<22} {results['off_the_shelf']['macro_precision']:>16.4f} {results['fine_tuned']['macro_precision']:>16.4f} {results['fine_tuned']['macro_precision'] - oos_f1:>+10.4f}")
        print(f"{'Macro Recall':<22} {results['off_the_shelf']['macro_recall']:>16.4f} {results['fine_tuned']['macro_recall']:>16.4f} {results['fine_tuned']['macro_recall'] - oos_f1:>+10.4f}")
        print(f"{'Macro F1':<22} {oos_f1:>16.4f} {ft_f1:>16.4f} {ft_f1 - oos_f1:>+10.4f}")
        print("===============================================================")

        comparison = {
            "off_the_shelf": oos_name,
            "fine_tuned": ft_name,
            "test_samples": len(test_split),
            "accuracy": {
                "off_the_shelf": oos_acc,
                "fine_tuned": ft_acc,
                "delta": ft_acc - oos_acc,
            },
            "macro_precision": {
                "off_the_shelf": results["off_the_shelf"]["macro_precision"],
                "fine_tuned": results["fine_tuned"]["macro_precision"],
                "delta": results["fine_tuned"]["macro_precision"] - results["off_the_shelf"]["macro_precision"],
            },
            "macro_recall": {
                "off_the_shelf": results["off_the_shelf"]["macro_recall"],
                "fine_tuned": results["fine_tuned"]["macro_recall"],
                "delta": results["fine_tuned"]["macro_recall"] - results["off_the_shelf"]["macro_recall"],
            },
            "macro_f1": {
                "off_the_shelf": oos_f1,
                "fine_tuned": ft_f1,
                "delta": ft_f1 - oos_f1,
            },
        }
    else:
        # No fine-tuned model available: still record the off-the-shelf scores so the
        # artifact contains the numbers instead of only a note.
        oos = results["off_the_shelf"]
        comparison = {
            "off_the_shelf": oos_name,
            "fine_tuned": None,
            "test_samples": len(test_split),
            "accuracy": {"off_the_shelf": oos["accuracy"], "fine_tuned": None, "delta": None},
            "macro_precision": {
                "off_the_shelf": oos["macro_precision"],
                "fine_tuned": None,
                "delta": None,
            },
            "macro_recall": {
                "off_the_shelf": oos["macro_recall"],
                "fine_tuned": None,
                "delta": None,
            },
            "macro_f1": {"off_the_shelf": oos["macro_f1"], "fine_tuned": None, "delta": None},
            "note": (
                "Only the off-the-shelf model was evaluated. Pass fine_tuned_source "
                "to enable the comparison."
            ),
        }

    # Save comparison artifacts
    compare_path = ARTIFACTS_DIR / "sentiment_comparison.json"
    with compare_path.open("w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved comparison report to %s", compare_path)

    return {
        "splits": {k: len(v) for k, v in splits.items()},
        "results": results,
        "comparison": comparison,
    }


def evaluate_finbert(analyzer: FinancialSentimentAnalyzer, sample_size: int = 300) -> Dict[str, Any]:
    """
    Legacy single-model evaluation against Financial PhraseBank.
    Kept for backwards compatibility; new comparisons should use run_sentiment_comparison().
    """
    logger.info("--- Starting FinBERT Sentiment Analysis Evaluation ---")
    splits = load_phrasebank_test_split(sample_size=sample_size, test_size=200)
    return evaluate_model_on_test(analyzer, splits["test"], model_name="finbert")


def _rouge_fscore(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-1/2/L f-measures via the installed ``rouge_score`` package.

    This project's environment has ``rouge_score`` installed but the HuggingFace
    ``evaluate.load('rouge')`` path is not available, so both eval scripts use this
    helper instead of ``evaluate``.
    """
    from rouge_score import rouge_scorer as _rs

    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=prediction)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def evaluate_bart(summarizer: TranscriptSummarizer) -> Dict[str, float]:
    """
    Evaluates BART-Large-CNN on financial text summaries using ROUGE metrics.
    Uses reference earnings call snippets with ground-truth human summaries.
    """
    logger.info("--- Starting BART-Large-CNN Summarization Evaluation ---")

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

    logger.info("Generating predictions using TranscriptSummarizer...")
    predictions: list[str] = []
    references: list[str] = []

    for idx, item in enumerate(eval_benchmark, start=1):
        generated_summary = summarizer.summarize(item["transcript"], max_length=100, min_length=20)
        predictions.append(generated_summary)
        references.append(item["reference"])
        logger.debug("Sample %d - Generated: %s", idx, generated_summary)

    rouge1 = float(np.mean([_rouge_fscore(p, r)["rouge1"] for p, r in zip(predictions, references)]))
    rouge2 = float(np.mean([_rouge_fscore(p, r)["rouge2"] for p, r in zip(predictions, references)]))
    rougeL = float(np.mean([_rouge_fscore(p, r)["rougeL"] for p, r in zip(predictions, references)]))

    print("\n================ BART ROUGE EVALUATION METRICS ================")
    print(f"ROUGE-1 : {rouge1:.4f}")
    print(f"ROUGE-2 : {rouge2:.4f}")
    print(f"ROUGE-L : {rougeL:.4f}")
    print("===============================================================")

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


def main():
    # Console shows only warnings/errors, so the terminal holds just the metric tables.
    # logs/app.log still receives every INFO record.
    setup_logging(log_level=logging.INFO, console_level=logging.WARNING)
    silence_third_party()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting Offline Model Evaluation Suite on device: %s", device)

    default_fine_tuned = Path("models/finbert_financial_sentiment_ft")
    fine_tuned_source = str(default_fine_tuned) if default_fine_tuned.exists() else None

    # 1. Sentiment comparison (off-the-shelf vs fine-tuned, if available)
    try:
        run_sentiment_comparison(
            off_the_shelf_source="ProsusAI/finbert",
            fine_tuned_source=fine_tuned_source,
            sample_size=4800,
            test_size=800,
            device=device,
            batch_size=16,
            max_length=512,
        )
    except Exception as e:
        import traceback as _tb

        print(f"\n!! Sentiment comparison failed: {e}\n")
        print(_tb.format_exc())
        logger.error("Sentiment comparison failed: %s", e, exc_info=True)

    # 2. BART Summarization Model Evaluation
    try:
        summarizer = TranscriptSummarizer(device=device)
        evaluate_bart(summarizer)
    except Exception as e:
        import traceback as _tb

        print(f"\n!! BART Evaluation Failed: {e}\n")
        print(_tb.format_exc())
        logger.error("BART evaluation failed: %s", e, exc_info=True)

    logger.info("Model evaluation pipeline finished. Results saved in 'eval_results/'.")


if __name__ == "__main__":
    main()
