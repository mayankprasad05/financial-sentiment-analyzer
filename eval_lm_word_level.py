"""
Word-level Loughran-McDonald evaluation entry point.

This script can run two modes:
- a small built-in smoke test
- a fetched-transcript mode that mirrors the existing app flow: fetch a transcript
  with the earningscall API, split it with the existing SpaCy preprocessor, and score
  it with the Loughran-McDonald scorer

Usage:
    python eval_lm_word_level.py
    python eval_lm_word_level.py --ticker AAPL --year 2024 --quarter 3
"""

import argparse
import logging

from src.loughran_mcdonald import LoughranMcDonaldScorer
from src.loughran_mcdonald import run_lm_smoke_test
from src.loughran_mcdonald import transcript_to_lm_records
from src.logger_config import setup_logging
from src.fetcher import fetch_transcript_by_ticker


def main(ticker: str | None = None, year: int = 2024, quarter: int = 3) -> None:
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting word-level Loughran-McDonald evaluation")

    scorer = LoughranMcDonaldScorer()

    if ticker:
        logger.info("Fetching transcript for %s Q%d %d", ticker, quarter, year)
        raw_transcript = fetch_transcript_by_ticker(ticker, year, quarter)
        if raw_transcript is None:
            logger.warning("No transcript returned for %s Q%d %d.", ticker, quarter, year)
            print("\n=== No transcript fetched ===")
            return

        logger.info("Scoring fetched transcript with Loughran-McDonald scorer")
        records = transcript_to_lm_records(scorer, raw_transcript)

        print("\n=== Word-level Loughran-McDonald scores for fetched transcript ===")
        print("ticker:", ticker)
        print("year:", year)
        print("quarter:", quarter)
        print("total_sentences:", records["total_sentences"])
        print("schema:", records["label_schema"])
        print("\nsentence_scores (first 20):")
        for idx, scores in enumerate(records["sentence_scores"][:20], start=1):
            print(f"  sentence {idx}:")
            for category, share in scores.items():
                print(f"    {category}: {share:.3f}")
        if len(records["sentence_scores"]) > 20:
            print(
                "  ... and "
                + str(len(records["sentence_scores"]) - 20)
                + " more sentences"
            )

        print("\naggregated:")
        for category, share in records["aggregated"].items():
            print(f"  {category}: {share:.3f}")

        return

    logger.info("Running built-in smoke test")
    result = run_lm_smoke_test()

    print("\n=== Word-level Loughran-McDonald scores ===")
    print("schema:", result["label_schema"])
    for idx, scores in enumerate(result["sentence_scores"], start=1):
        print(f"sentence {idx}:")
        for category, share in scores.items():
            print(f"  {category}: {share:.3f}")
    print("\naggregated:")
    for category, share in result["aggregated"].items():
        print(f"  {category}: {share:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word-level Loughran-McDonald evaluation"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Fetch and score a transcript for this ticker",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Earnings call year",
    )
    parser.add_argument(
        "--quarter",
        type=int,
        default=3,
        help="Earnings call quarter",
    )
    args = parser.parse_args()

    main(
        ticker=args.ticker,
        year=args.year,
        quarter=args.quarter,
    )
