"""
Lightweight FinBERT fine-tuning for sentence-level financial sentiment.

This script fine-tunes ProsusAI/finbert on the Financial PhraseBank dataset and saves
a local model + tokenizer that the existing pipeline can load directly.

It is intentionally kept small and CPU/GPU-friendly:
- uses a small sample of the dataset for quick iteration
- saves artifacts under models/finbert_financial_sentiment_ft
- keeps the same 3-class label space as the off-the-shelf model
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
import numpy as np

from src.logger_config import setup_logging

logger = logging.getLogger(__name__)

MODEL_SOURCE = "ProsusAI/finbert"
LOCAL_MODEL_DIR = Path("models/finbert_financial_sentiment_ft")

# Financial PhraseBank maps 0 -> negative, 1 -> neutral, 2 -> positive
LABEL_NAMES = ["negative", "neutral", "positive"]


class PhraseBankDataset(Dataset):
    """Tiny wrapper turning tokenized inputs into a torch Dataset."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: List[int]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_datasets(
    sample_size: int = 1200,
    test_size: int = 200,
    max_length: int = 128,
) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Load Financial PhraseBank from a local parquet snapshot, tokenize it, and return
    train/val/test datasets.

    If the local snapshot does not exist, this helper raises a clear error asking the user to
    download it once from HuggingFace and place it under data/financial_phrasebank.parquet.
    """
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    parquet_path = Path("data/loughran_mcdonald.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            "Local dataset snapshot not found at data/loughran_mcdonald.parquet.\n\n"
            "Convert the Loughran-McDonald file once with the project venv python:\n\n"
            "    python - <<'PY'\n"
            "    import pandas as pd\n"
            "    df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv')\n"
            "    df.to_parquet('data/loughran_mcdonald.parquet')\n"
            "    print('saved to data/loughran_mcdonald.parquet')\n"
            "    PY\n\n"
            "The current capstone setup uses a one-time local parquet copy instead.\n"
        )

    import pandas as pd
    import numpy as np

    df = pd.read_parquet(parquet_path)
    df = df.head(max(sample_size, test_size + 50))

    logger.info("Loaded Financial PhraseBank snapshot with %d rows", len(df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    ds = PhraseBankDataset(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=labels,
    )

    total = len(ds)
    rng = np.random.default_rng(42)
    perm = rng.permutation(total).tolist()

    test_count = test_size
    train_count = total - test_count
    train_indices = perm[:train_count]
    test_indices = perm[train_count:]

    train_ds = torch.utils.data.Subset(ds, train_indices)
    test_ds = torch.utils.data.Subset(ds, test_indices)

    val_count = max(1, int(len(train_ds) * 0.1))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds,
        [len(train_ds) - val_count, val_count],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(
        "Prepared datasets -> train=%d, val=%d, test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds, tokenizer


def compute_metrics(result) -> Dict[str, float]:
    """
    Minimal metrics computed from Trainer predictions.
    result is a TrainerEvaluationPredictionOutput-like tuple (predictions, label_ids).
    """
    preds, labels = result
    preds = np.argmax(preds, axis=-1)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABEL_NAMES))), average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": float(accuracy), "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}


def main(
    sample_size: int = 1200,
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: Path = LOCAL_MODEL_DIR,
    use_gpu: bool = True,
):
    setup_logging(log_level=logging.INFO)
    logger.info("Starting lightweight FinBERT financial sentiment fine-tuning")

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    logger.info("Using device: %s", device)

    train_ds, val_ds, test_ds, tokenizer = load_datasets(
        sample_size=sample_size, test_size=test_size, max_length=max_length
    )

    config = AutoConfig.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE, config=config)
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=max(1, len(train_ds) // (batch_size * 4)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        dataloader_num_workers=0,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished. Evaluating on held-out test set...")

    test_metrics = trainer.evaluate(test_ds)
    logger.info("Test metrics: %s", test_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved fine-tuned model + tokenizer to %s", output_dir)

    metrics_path = output_dir / "fine_tune_metrics.json"
    import json

    with metrics_path.open("w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "dataset_sample_size": sample_size,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
        }, f, indent=2)
    logger.info("Saved fine-tuning metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight fine-tuning for financial sentiment (FinBERT)"
    )
    parser.add_argument("--sample-size", type=int, default=1200, help="Number of PhraseBank examples to use")
    parser.add_argument("--max-length", type=int, default=128, help="Token max length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_MODEL_DIR, help="Where to save the fine-tuned model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    main(
        sample_size=args.sample_size,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_gpu=not args.cpu,
    )
