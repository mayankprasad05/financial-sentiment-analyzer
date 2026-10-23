import logging
from typing import Dict, Any
from src.preprocessing import TranscriptPreprocessor
from src.modeling import FinancialSentimentAnalyzer
from src.summarizer import TranscriptSummarizer

logger = logging.getLogger(__name__)


class EarningsAnalysisPipeline:
    """End-to-end NLP pipeline evaluating sentiment and abstractive summarization across transcripts."""

    def __init__(self) -> None:
        logger.info("Initializing EarningsAnalysisPipeline...")
        self.preprocessor = TranscriptPreprocessor()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.summarizer = TranscriptSummarizer()

    def run(self, raw_transcript: str) -> Dict[str, Any]:
        """Runs unified sentiment analysis and summarization on the transcript."""
        logger.info("Executing pipeline on input transcript...")

        # Extract sentences directly from the complete transcript
        sentences = self.preprocessor.extract_sentences(raw_transcript)

        if not sentences:
            logger.warning("No valid sentences parsed from input text.")
            return {
                "metrics": {
                    "overall_net_sentiment": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "total_sentences": 0
                },
                "summary": "No content available to summarize.",
                "sentence_details": []
            }

        # Classify sentiment sentence-by-sentence via FinBERT (from src.modeling)
        sentence_details = self.sentiment_analyzer.analyze_sentences(sentences)

        # Compute aggregate sentiment metrics across the transcript
        total_sentences = len(sentence_details)
        pos_count = sum(1 for s in sentence_details if s["label"] == "positive")
        neg_count = sum(1 for s in sentence_details if s["label"] == "negative")
        neu_count = sum(1 for s in sentence_details if s["label"] == "neutral")

        overall_net_sentiment = (
            sum(s["net_sentiment"] for s in sentence_details) / total_sentences
            if total_sentences > 0 else 0.0
        )

        # Generate section-aware, LM-guided abstractive summary
        summary_result = self.summarizer.summarize_with_metadata(raw_transcript)

        return {
            "metrics": {
                "overall_net_sentiment": overall_net_sentiment,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "neutral_count": neu_count,
                "total_sentences": total_sentences
            },
            "summary": summary_result["summary"],
            "summary_meta": summary_result,
            "sentence_details": sentence_details
        }