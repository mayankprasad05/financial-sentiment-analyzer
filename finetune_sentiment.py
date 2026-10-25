"""
Lightweight FinBERT fine-tuning for sentence-level financial sentiment.

Fine-tunes ProsusAI/finbert on the Financial PhraseBank (sentences_50agree) and saves a
local model + tokenizer that the rest of the project loads directly.

The train/val/test split is taken from ``eval.load_phrasebank_test_split`` so the
fine-tuned model is measured on exactly the held-out sentences ``eval.py`` uses. Using a
separate split here would leak training sentences into eval.py's test set and inflate the
reported accuracy.

Saved artifacts (under models/finbert_financial_sentiment_ft):
- model weights + config (id2label set to the PhraseBank order)
- tokenizer
- fine_tune_metrics.json (test metrics plus the training configuration)

Usage:
    python finetune_sentiment.py
    python finetune_sentiment.py --epochs 5 --batch-size 32
    python finetune_sentiment.py --cpu
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Set before transformers is imported so progress bars and hub warnings stay out of the
# way of the metrics output.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.logger_config import setup_logging, silence_third_party

logger = logging.getLogger(__name__)

MODEL_SOURCE = "ProsusAI/finbert"
LOCAL_MODEL_DIR = Path("models/finbert_financial_sentiment_ft")

# Financial PhraseBank label space: 0 -> negative, 1 -> neutral, 2 -> positive
LABEL_NAMES = ["negative", "neutral", "positive"]


class PhraseBankDataset(Dataset):
    """Tiny wrapper turning pre-tokenized tensors into a torch Dataset."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: List[int],
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_datasets(
    sample_size: int = 4800,
    test_size: int = 800,
    max_length: int = 64,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """Load and tokenize the shared PhraseBank train/val/test split.

    The split is imported from ``eval.py`` on purpose: both scripts must agree on which
    sentences are held out, otherwise the fine-tuned model's test accuracy is measured on
    data it was trained on.
    """
    from eval import load_phrasebank_test_split

    splits = load_phrasebank_test_split(
        sample_size=sample_size, test_size=test_size, seed=seed
    )
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

    def build(split) -> PhraseBankDataset:
        encoded = tokenizer(
            list(split["sentence"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return PhraseBankDataset(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=list(split["label"]),
        )

    train_ds, val_ds, test_ds = build(train_split), build(val_split), build(test_split)

    logger.info(
        "Prepared datasets -> train=%d, val=%d, test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds, tokenizer


def compute_metrics(eval_prediction) -> Dict[str, float]:
    """Accuracy and macro-averaged precision/recall/F1 over the three classes."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def main(
    sample_size: int = 4800,
    test_size: int = 800,
    max_length: int = 64,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: Path = LOCAL_MODEL_DIR,
    use_gpu: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    setup_logging(log_level=logging.INFO, console_level=logging.WARNING)
    silence_third_party()
    logger.info("Starting lightweight FinBERT financial sentiment fine-tuning")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    logger.info("Using device: %s", device)

    train_ds, val_ds, test_ds, tokenizer = load_datasets(
        sample_size=sample_size, test_size=test_size, max_length=max_length, seed=seed
    )

    config = AutoConfig.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE, config=config)
    # PhraseBank's class order. FinancialSentimentAnalyzer reads id2label, so predictions
    # are decoded with the right class names.
    model.config.id2label = {index: name for index, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: index for index, name in enumerate(LABEL_NAMES)}
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=max(1, len(train_ds) // (batch_size * 4)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished. Evaluating on the held-out test split...")

    test_metrics = trainer.evaluate(test_ds)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        key.replace("eval_", ""): float(value)
        for key, value in test_metrics.items()
        if isinstance(value, (int, float))
    }

    metrics_path = output_dir / "fine_tune_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(
            {
                "test_metrics": summary,
                "splits": {
                    "train": len(train_ds),
                    "val": len(val_ds),
                    "test": len(test_ds),
                },
                "dataset_sample_size": sample_size,
                "test_size": test_size,
                "max_length": max_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": device,
                "seed": seed,
            },
            handle,
            indent=2,
        )

    print("\n================ FINETUNED FINBERT TEST METRICS ================")
    print(f"test samples    : {len(test_ds)}")
    print(f"Accuracy        : {summary.get('accuracy', float('nan')):.4f}")
    print(f"Macro Precision : {summary.get('macro_precision', float('nan')):.4f}")
    print(f"Macro Recall    : {summary.get('macro_recall', float('nan')):.4f}")
    print(f"Macro F1-Score  : {summary.get('macro_f1', float('nan')):.4f}")
    print("===============================================================")

    logger.info("Saved fine-tuned model + tokenizer to %s", output_dir)
    logger.info("Saved fine-tuning metrics to %s", metrics_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight fine-tuning for financial sentiment (FinBERT)"
    )
    parser.add_argument("--sample-size", type=int, default=4800, help="Number of PhraseBank examples to use")
    parser.add_argument("--test-size", type=int, default=800, help="Held-out test sentences (must match eval.py)")
    parser.add_argument("--max-length", type=int, default=64, help="Token max length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_MODEL_DIR, help="Where to save the fine-tuned model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    main(
        sample_size=args.sample_size,
        test_size=args.test_size,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
        seed=args.seed,
    )
