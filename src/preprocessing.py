import logging
from typing import List
import spacy

logger = logging.getLogger(__name__)


class TranscriptPreprocessor:
    """Preprocesses earnings call transcripts using SpaCy."""

    def __init__(self) -> None:
        logger.info("Initializing TranscriptPreprocessor")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError as e:
            logger.error("SpaCy model 'en_core_web_sm' not found", exc_info=True)
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Run 'python -m spacy download en_core_web_sm'."
            ) from e

    def extract_sentences(self, text: str) -> List[str]:
        """Splits text into clean individual sentences using SpaCy NLP."""
        if not text or not text.strip():
            logger.warning("Empty text provided for sentence extraction.")
            return []

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

        logger.info(f"Extracted {len(sentences)} sentences from input transcript.")
        return sentences