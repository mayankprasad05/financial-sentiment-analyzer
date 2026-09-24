from typing import Dict, Any, List
import logging
import numpy as np
from src.preprocessing import TranscriptPreprocessor
from src.modeling import FinancialSentimentAnalyzer, ExecutiveSummarizer

logger = logging.getLogger(__name__)

class EarningsAnalysisPipeline:
    """Orchestrates ingestion, tokenization, sentence-level analysis, and summary extraction."""

    def __init__(self) -> None:
        logger.info("Initializing EarningsAnalysisPipeline")
        try:
            self.preprocessor = TranscriptPreprocessor()
            self.sentiment_engine = FinancialSentimentAnalyzer()
            self.summarizer_engine = ExecutiveSummarizer()
            logger.info("Pipeline components initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize pipeline components", exc_info=True)
            raise

    def run(self, raw_transcript: str) -> Dict[str, Any]:
        """Runs the complete NLP pipeline across segmented streams."""
        logger.info(f"Starting pipeline with transcript of length {len(raw_transcript)} characters")
        
        try:
            # Preprocessing
            logger.debug("Step 1: Preprocessing transcript")
            sections = self.preprocessor.segment_transcript(raw_transcript)

            logger.debug("Step 2: Extracting sentences from executive remarks")
            remarks_sents = self.preprocessor.extract_sentences(sections["remarks"])
            
            logger.debug("Step 3: Extracting sentences from Q&A section")
            qa_sents = self.preprocessor.extract_sentences(sections["qa"])

            # Sentiment Analysis
            logger.debug("Step 4: Analyzing sentiment of remarks")
            remarks_sentiment = self.sentiment_engine.analyze_sentences(remarks_sents)
            
            logger.debug("Step 5: Analyzing sentiment of Q&A")
            qa_sentiment = self.sentiment_engine.analyze_sentences(qa_sents)

            # Metrics calculation
            def calc_net_score(sentiments: List[Dict[str, Any]]) -> float:
                if not sentiments:
                    return 0.0
                return float(np.mean([item["net_sentiment"] for item in sentiments]))

            remarks_net = calc_net_score(remarks_sentiment)
            qa_net = calc_net_score(qa_sentiment)
            sentiment_shift = qa_net - remarks_net
            
            logger.info(f"Sentiment metrics - Remarks: {remarks_net:.4f}, Q&A: {qa_net:.4f}, Shift: {sentiment_shift:.4f}")

            # Summarization
            logger.debug("Step 6: Generating executive summary")
            summary = self.summarizer_engine.summarize(sections["remarks"])

            logger.info("Pipeline execution completed successfully")
            
            return {
                "metrics": {
                    "remarks_net_sentiment": remarks_net,
                    "qa_net_sentiment": qa_net,
                    "sentiment_shift": sentiment_shift,
                    "total_sentences": len(remarks_sents) + len(qa_sents),
                },
                "summary": summary,
                "remarks_details": remarks_sentiment,
                "qa_details": qa_sentiment,
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise