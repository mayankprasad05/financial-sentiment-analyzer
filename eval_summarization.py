"""
Summarization evaluation harness.

Compares the section-aware, Loughran-McDonald-guided summarizer against the legacy
"truncate to the model window and summarize" behavior on the same transcripts, and
reports the metrics that actually matter for this project:

* **ROUGE-1/2/L** against reference summaries.
* **LM sentiment retention** - cosine similarity between the transcript's LM category
  share vector and the summary's. This is the metric that answers "does the summary
  still say what the call said?", which ROUGE alone does not.
* **Disclaimer suppression** - how much Litigious/Constraining (boilerplate) language
  survives into the summary, measured as a share delta against the transcript.
* **Compression** - summary tokens over transcript tokens.

Usage:
    python eval_summarization.py
    python eval_summarization.py --dataset data/summarization_benchmark.jsonl
    python eval_summarization.py --model sshleifer/distilbart-cnn-6-6 --max-examples 5
    python eval_summarization.py --ablations

Dataset format (JSONL or JSON list), one object per line:
    {"transcript": "...", "reference": "human written summary"}

Without ``--dataset`` the harness runs a two-example built-in smoke benchmark. Those
numbers are a wiring check, not a benchmark result.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rouge_score import rouge_scorer

from src.loughran_mcdonald import LoughranMcDonaldScorer
from src.logger_config import setup_logging
from src.summarizer import (
    DEFAULT_MODEL_NAME,
    SummarizerConfig,
    TranscriptSummarizer,
    legacy_summarizer_config,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("eval_results/summarization_eval.json")

# Built-in smoke benchmark: the two reference snippets that already existed in eval.py.
# Kept only so the harness has something to run without external data. Replace with a
# real labelled set before reporting numbers.
SMOKE_BENCHMARK: List[Dict[str, str]] = [
    {
        "transcript": (
            "We delivered exceptional revenue growth of 24% year-over-year, hitting $1.2B in Q4. "
            "Gross margins expanded by 150 basis points due to operational efficiency and strong "
            "Enterprise adoption. However, supply chain bottlenecks created mild cost headwinds in "
            "logistics. The inflation impact on freight cost is real and lingering, though demand "
            "remains robust."
        ),
        "reference": (
            "Q4 revenue grew 24% YoY to $1.2B with gross margins expanding 150 bps. Despite strong "
            "demand, lingering supply chain inflation and logistics costs presented mild headwinds."
        ),
    },
    {
        "transcript": (
            "Total net sales increased 12% to $850 million compared to the prior year period. "
            "Net income was $95 million, down from $110 million due to elevated operating expenses "
            "and R&D investments in next-generation platforms. Free cash flow generation was strong "
            "at $140 million."
        ),
        "reference": (
            "Net sales increased 12% to $850M, but net income dropped to $95M driven by higher "
            "operating and R&D expenses. Free cash flow stood solid at $140M."
        ),
    },
]


# --- datasets ---------------------------------------------------------------------
def load_benchmark(dataset_path: Optional[str]) -> List[Dict[str, str]]:
    """Load a labelled benchmark, or fall back to the built-in smoke set."""
    if not dataset_path:
        logger.warning(
            "No --dataset supplied; using the two-example built-in smoke benchmark. "
            "Treat the resulting numbers as a wiring check only."
        )
        return SMOKE_BENCHMARK

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        examples = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload if isinstance(payload, list) else [payload]
    else:
        raise ValueError(f"Unsupported benchmark format: {path.suffix} (use .jsonl or .json)")

    for index, example in enumerate(examples):
        if "transcript" not in example or "reference" not in example:
            raise ValueError(
                f"Benchmark example {index} must have 'transcript' and 'reference' keys"
            )
    logger.info("Loaded %d benchmark examples from %s", len(examples), path)
    return examples


# --- metrics ----------------------------------------------------------------------
def rouge_scores(prediction: str, reference: str, scorer: rouge_scorer.RougeScorer) -> Dict[str, float]:
    scores = scorer.score(target=reference, prediction=prediction)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def lm_share_vector(scorer: LoughranMcDonaldScorer, text: str) -> np.ndarray:
    shares = scorer.score_text(text)
    return np.array([shares.get(category, 0.0) for category in scorer.label_schema], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def lm_retention(
    scorer: LoughranMcDonaldScorer, transcript: str, summary: str
) -> Optional[float]:
    """Cosine similarity between transcript and summary LM category shares.

    Returns ``None`` when either text has no dictionary hits at all: a BART summary made
    of pure figures ("revenue grew 12% to $850 million") carries no LM-tagged words, and
    scoring that as 0.0 would read as "no retention" instead of "not measurable".
    """
    transcript_vector = lm_share_vector(scorer, transcript)
    summary_vector = lm_share_vector(scorer, summary)
    if not transcript_vector.any() or not summary_vector.any():
        return None
    return cosine(transcript_vector, summary_vector)


def hedging_per_1k(scorer: LoughranMcDonaldScorer, text: str) -> float:
    """Litigious + Constraining dictionary hits per 1,000 tokens.

    A density rather than a share, so it stays defined and comparable when a text
    contains no tagged words at all.
    """
    stats = scorer.tag_stats(text)
    tokens = max(1, stats["tokens"])
    hedge_hits = stats["counts"].get("Litigious", 0) + stats["counts"].get("Constraining", 0)
    return 1000.0 * hedge_hits / tokens


def disclaimer_suppression(
    scorer: LoughranMcDonaldScorer, transcript: str, summary: str
) -> float:
    """Drop in boilerplate density, in dictionary hits per 1,000 tokens.

    Positive means the summary carries less Litigious/Constraining language per token
    than the transcript it came from.
    """
    return hedging_per_1k(scorer, transcript) - hedging_per_1k(scorer, summary)


# --- arms -------------------------------------------------------------------------
def build_arms(model_name: str, include_ablations: bool) -> Dict[str, SummarizerConfig]:
    """Configurations compared by the harness."""
    arms: Dict[str, SummarizerConfig] = {
        "legacy_truncation": legacy_summarizer_config(model_name),
        "guided": SummarizerConfig(model_name=model_name),
    }
    if include_ablations:
        arms["strip_only"] = SummarizerConfig(
            model_name=model_name, select_by_salience=False
        )
        arms["salience_only"] = SummarizerConfig(
            model_name=model_name, strip_boilerplate=False
        )
    return arms


# --- evaluation -------------------------------------------------------------------
def evaluate_summarizer(
    summarizer: TranscriptSummarizer,
    examples: Sequence[Dict[str, str]],
    arms: Dict[str, SummarizerConfig],
    lm_scorer: Optional[LoughranMcDonaldScorer] = None,
) -> Dict[str, Any]:
    """Run every arm over every example and return per-example and aggregate results.

    The summarizer instance is reused across arms; only its config changes, so the
    model and tokenizer are loaded once.
    """
    lm_scorer = lm_scorer or LoughranMcDonaldScorer()
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results: Dict[str, Any] = {"arms": {}, "examples": len(examples)}

    for arm_name, config in arms.items():
        logger.info("Evaluating arm: %s", arm_name)
        summarizer.config = config
        arm_records: List[Dict[str, Any]] = []

        for index, example in enumerate(examples, start=1):
            transcript = example["transcript"]
            reference = example["reference"]

            started = time.perf_counter()
            output = summarizer.summarize_with_metadata(transcript)
            latency = time.perf_counter() - started
            summary = output["summary"]

            retention = lm_retention(lm_scorer, transcript, summary)

            # What the arm actually fed the model (planning is cheap; generation is not).
            plan = summarizer.plan(transcript)
            input_text = " ".join(chunk.text for chunk in plan["selected_chunks"])
            input_hedging = hedging_per_1k(lm_scorer, input_text)

            record: Dict[str, Any] = {
                "index": index,
                "summary": summary,
                "latency_seconds": round(latency, 2),
                "summary_tokens": output["summary_tokens"],
                "transcript_tokens": output["transcript_tokens"],
                "compression_ratio": output["compression_ratio"],
                "chunks_used": output["chunks_used"],
                "chunks_total": output["chunks_total"],
                "input_tokens": output["input_tokens"],
                "boilerplate_removed_count": output["boilerplate_removed_count"],
                "lm_retention": None if retention is None else round(retention, 4),
                "lm_retention_defined": retention is not None,
                "disclaimer_suppression": round(
                    disclaimer_suppression(lm_scorer, transcript, summary), 4
                ),
                "hedging_per_1k_input": round(input_hedging, 4),
                "hedging_per_1k_transcript": round(hedging_per_1k(lm_scorer, transcript), 4),
                "hedging_per_1k_summary": round(hedging_per_1k(lm_scorer, summary), 4),
            }
            record.update({key: round(value, 4) for key, value in rouge_scores(summary, reference, rouge).items()})
            arm_records.append(record)
            logger.info(
                "arm=%s example=%d rougeL=%.4f lm_retention=%s latency=%.1fs",
                arm_name,
                index,
                record["rougeL"],
                "n/a" if retention is None else f"{retention:.4f}",
                latency,
            )

        results["arms"][arm_name] = {
            "config": asdict(config),
            "records": arm_records,
            "aggregate": _aggregate(arm_records),
        }

    results["comparison"] = _compare(results["arms"])
    return results


_METRIC_KEYS = (
    "rouge1",
    "rouge2",
    "rougeL",
    "lm_retention",
    "lm_retention_coverage",
    "disclaimer_suppression",
    "hedging_per_1k_input",
    "compression_ratio",
    "input_tokens",
    "chunks_used",
    "latency_seconds",
    "boilerplate_removed_count",
)


def _aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Average every metric over the examples where it is defined."""
    if not records:
        return {}

    aggregate: Dict[str, float] = {}
    for key in _METRIC_KEYS:
        if key == "lm_retention_coverage":
            continue
        values = [record[key] for record in records if record.get(key) is not None]
        if values:
            aggregate[key] = round(float(np.mean(values)), 4)

    aggregate["lm_retention_coverage"] = round(
        sum(1 for record in records if record.get("lm_retention_defined")) / len(records), 4
    )
    return aggregate


def _compare(arms: Dict[str, Any]) -> Dict[str, Any]:
    """Delta of every arm against the legacy truncation baseline."""
    if "legacy_truncation" not in arms:
        return {}
    baseline = arms["legacy_truncation"]["aggregate"]
    comparison: Dict[str, Any] = {}
    for arm_name, payload in arms.items():
        if arm_name == "legacy_truncation":
            continue
        aggregate = payload["aggregate"]
        comparison[arm_name] = {
            key: round(aggregate[key] - baseline[key], 4)
            for key in aggregate
            if key in baseline and key != "lm_retention_coverage"
        }
    return comparison


# --- reporting --------------------------------------------------------------------
def print_report(results: Dict[str, Any], dataset_label: str) -> None:
    print("\n================ SUMMARIZATION EVALUATION ================")
    print(f"dataset : {dataset_label}")
    print(f"examples: {results['examples']}")
    print(f"model   : {results.get('model')}")
    print("----------------------------------------------------------")
    header = f"{'metric':<26}" + "".join(
        f"{arm:>22}" for arm in results["arms"]
    )
    print(header)
    for key in _METRIC_KEYS:
        row = f"{key:<26}"
        for arm in results["arms"]:
            value = results["arms"][arm]["aggregate"].get(key)
            row += f"{(f'{value:.4f}' if value is not None else '-'):>22}"
        print(row)
    print("lm_retention is undefined when the summary has no dictionary hits;")
    print("lm_retention_coverage is the share of examples where it could be computed.")

    if results.get("comparison"):
        print("----------------------------------------------------------")
        print("delta vs legacy_truncation (higher is better except latency and compression):")
        for arm, deltas in results["comparison"].items():
            rendered = ", ".join(f"{key}={value:+.4f}" for key, value in deltas.items())
            print(f"  {arm}: {rendered}")
    print("==========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarization evaluation harness")
    parser.add_argument("--dataset", default=None, help="Path to a .jsonl or .json benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Summarization model")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit examples (0 = all)")
    parser.add_argument("--ablations", action="store_true", help="Add strip-only and salience-only arms")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Where to write the JSON report")
    args = parser.parse_args()

    setup_logging(log_level=logging.INFO)

    examples = load_benchmark(args.dataset)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    dataset_label = args.dataset or "built-in smoke benchmark (2 examples)"

    summarizer = TranscriptSummarizer(model_name=args.model, device=args.device)
    arms = build_arms(args.model, include_ablations=args.ablations)

    results = evaluate_summarizer(summarizer, examples, arms)
    results["model"] = args.model
    results["dataset"] = dataset_label

    print_report(results, dataset_label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
