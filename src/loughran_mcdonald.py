"""
Loughran-McDonald word-level sentiment scoring for financial text.

This module is a standalone LM word-count scoring engine that consumes the project's
local Loughran-McDonald parquet snapshot and assigns category hits and shares to
arbitrary text.

Label schema used here:
- Negative
- Positive
- Uncertainty
- Litigious
- Constraining
- Superfluous
- ModalStrong
- ModalWeak
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logger_config import setup_logging

logger = logging.getLogger(__name__)

LM_PARQUET_PATH: Path = Path("data/loughran_mcdonald.parquet")

CATEGORY_COLUMNS: Dict[str, str] = {
    "Negative": "Negative",
    "Positive": "Positive",
    "Uncertainty": "Uncertainty",
    "Litigious": "Litigious",
    "Constraining": "Constraining",
    "Superfluous": "Superfluous",
    "ModalStrong": "Strong_Modal",
    "ModalWeak": "Weak_Modal",
}

DEFAULT_LABEL_SCHEMA: List[str] = list(CATEGORY_COLUMNS.keys())


class LoughranMcDonaldScorer:
    """Word-level LM sentiment scorer built from the local Master Dictionary snapshot."""

    def __init__(
        self,
        lm_parquet_path: Optional[Path] = None,
        label_schema: Optional[List[str]] = None,
    ) -> None:
        path = lm_parquet_path or LM_PARQUET_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Loughran-McDonald parquet snapshot not found at {path}.\n\n"
                "Convert the CSV once with the project python:\n\n"
                "    python - <<'PY'\n"
                "    import pandas as pd\n"
                "    from pathlib import Path\n"
                "    df = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2025.csv')\n"
                "    df.to_parquet('data/loughran_mcdonald.parquet')\n"
                "    print('saved to data/loughran_mcdonald.parquet')\n"
                "    PY"
            )

        self._label_schema = label_schema or DEFAULT_LABEL_SCHEMA
        logger.info("Loading Loughran-McDonald snapshot from %s", path)
        self._df = pd.read_parquet(path)
        self._word_index = self._build_word_index()
        logger.info(
            "Loughran-McDonald scorer ready: %d dictionary entries indexed",
            len(self._word_index),
        )

    def _build_word_index(self) -> Dict[str, Dict[str, int]]:
        index: Dict[str, Dict[str, int]] = {}
        for row in self._df.itertuples(index=False):
            word = str(row.Word).strip().upper()
            if not word:
                continue
            row_map: Dict[str, int] = {}
            for category, column_name in CATEGORY_COLUMNS.items():
                value = getattr(row, column_name, 0)
                # The Master Dictionary flags words with a year marker (e.g. 2009), not
                # with 1. Coerce to a boolean hit so counts, densities and salience are
                # in real word units instead of thousands.
                try:
                    row_map[category] = 1 if int(value) != 0 else 0
                except (TypeError, ValueError):
                    row_map[category] = 0
            index[word] = row_map
        return index

    @property
    def label_schema(self) -> List[str]:
        return list(self._label_schema)

    def score_text(self, text: str) -> Dict[str, float]:
        """Return per-category hit share for a single text string.

        Hit share is category hits divided by total LM-tagged token hits, so every
        returned category is in [0, 1] and the shares emphasize relative emphasis
        rather than raw counts.
        """
        if not text or not isinstance(text, str):
            return {category: 0.0 for category in self._label_schema}

        tokens = _tokenize(text)
        if not tokens:
            return {category: 0.0 for category in self._label_schema}

        category_hits: Dict[str, int] = {category: 0 for category in self._label_schema}
        total_hits = 0

        for token in tokens:
            entry = self._word_index.get(token)
            if entry is None:
                continue
            for category in self._label_schema:
                hits = entry.get(category, 0)
                if hits:
                    category_hits[category] += hits
                    total_hits += hits

        if total_hits == 0:
            return {category: 0.0 for category in self._label_schema}

        return {
            category: category_hits[category] / total_hits
            for category in self._label_schema
        }

    def score_sentences(self, sentences: List[str]) -> List[Dict[str, float]]:
        """Score a list of sentences and return one dict per sentence."""
        return [self.score_text(sentence) for sentence in sentences]

    def tag_stats(self, text: str) -> Dict[str, Any]:
        """Return raw dictionary hit counts and per-token densities for a text.

        ``score_text`` normalizes by tagged-token hits, which is useful for reporting
        but useless for ranking: a one-word sentence with a single tag scores 1.0.
        Densities (hits per token) are what salience ranking needs.
        """
        empty_counts = {category: 0 for category in self._label_schema}
        if not text or not isinstance(text, str):
            return {
                "tokens": 0,
                "tagged_tokens": 0,
                "counts": empty_counts,
                "densities": {category: 0.0 for category in self._label_schema},
            }

        tokens = _tokenize(text)
        counts = {category: 0 for category in self._label_schema}
        tagged_tokens = 0

        for token in tokens:
            entry = self._word_index.get(token)
            if entry is None:
                continue
            for category in self._label_schema:
                hits = entry.get(category, 0)
                if hits:
                    counts[category] += hits
                    tagged_tokens += hits

        token_count = len(tokens)
        densities = {
            category: (counts[category] / token_count if token_count else 0.0)
            for category in self._label_schema
        }

        return {
            "tokens": token_count,
            "tagged_tokens": tagged_tokens,
            "counts": counts,
            "densities": densities,
        }

    def aggregate(self, sentence_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-sentence hit shares into a corpus-level share vector."""
        if not sentence_scores:
            return {category: 0.0 for category in self._label_schema}

        aggregated: Dict[str, float] = {category: 0.0 for category in self._label_schema}
        for scores in sentence_scores:
            for category in self._label_schema:
                aggregated[category] += scores.get(category, 0.0)
        count = len(sentence_scores)
        return {category: aggregated[category] / count for category in self._label_schema}


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer sufficient for LM dictionary matching.

    The Master Dictionary keys are uppercase plain words, so we uppercase and split
    on non-alphanumeric boundaries. This is intentionally lightweight and avoids
    adding a SpaCy dependency to the LM path.
    """
    if not text:
        return []

    lowered = text.upper()
    tokens = re.findall(r"[A-Z0-9]+", lowered)
    return [token for token in tokens if token]


def transcript_to_lm_records(
    scorer: LoughranMcDonaldScorer,
    raw_transcript: str,
    preprocessor: Optional["TranscriptPreprocessor"] = None,
) -> Dict[str, object]:
    """Score a raw transcript the same way the rest of the project processes transcripts.

    This mirrors the existing pipeline flow: preprocess into sentences, then score each
    sentence with the LM scorer and return both sentence-level and aggregated results.
    """
    if preprocessor is None:
        from src.preprocessing import TranscriptPreprocessor

        preprocessor = TranscriptPreprocessor()

    sentences = preprocessor.extract_sentences(raw_transcript)
    if not sentences:
        return {
            "transcript": raw_transcript,
            "sentences": [],
            "sentence_scores": [],
            "aggregated": {category: 0.0 for category in scorer.label_schema},
            "label_schema": scorer.label_schema,
            "total_sentences": 0,
        }

    sentence_scores = scorer.score_sentences(sentences)
    aggregated = scorer.aggregate(sentence_scores)

    return {
        "transcript": raw_transcript,
        "sentences": sentences,
        "sentence_scores": sentence_scores,
        "aggregated": aggregated,
        "label_schema": scorer.label_schema,
        "total_sentences": len(sentences),
    }


def run_lm_smoke_test() -> Dict[str, object]:
    """Quick end-to-end check that the LM scorer can load and score text."""
    setup_logging(log_level=logging.INFO)
    logger.info("Running Loughran-McDonald word-level smoke test")

    scorer = LoughranMcDonaldScorer()
    sample_sentences = [
        "Litigation risk remains uncertain and constraining.",
        "Strong results and abundant liquidity were reported.",
        "Weakening guidance reflects weak demand and superfluous costs.",
    ]

    sentence_scores = scorer.score_sentences(sample_sentences)
    aggregated = scorer.aggregate(sentence_scores)

    return {
        "sentence_scores": sentence_scores,
        "aggregated": aggregated,
        "label_schema": scorer.label_schema,
    }


if __name__ == "__main__":
    setup_logging(log_level=logging.INFO)
    result = run_lm_smoke_test()
    print("label_schema:", result["label_schema"])
    print("aggregated:")
    for category, share in result["aggregated"].items():
        print(f"  {category}: {share:.3f}")
    print("sentence_scores:")
    for idx, scores in enumerate(result["sentence_scores"], start=1):
        print(f"  sentence {idx}:")
        for category, share in scores.items():
            print(f"    {category}: {share:.3f}")
