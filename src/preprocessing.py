import re
import logging
from typing import Dict, List
import spacy

logger = logging.getLogger(__name__)

class TranscriptPreprocessor:
    """Preprocesses and segments earnings call transcripts using SpaCy."""

    def __init__(self) -> None:
        logger.info("Initializing TranscriptPreprocessor")
        try:
            logger.debug("Loading SpaCy model: en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError as e:
            logger.error("SpaCy model 'en_core_web_sm' not found", exc_info=True)
            raise RuntimeError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Run 'python -m spacy download en_core_web_sm'."
            ) from e

    def segment_transcript(self, raw_text: str) -> Dict[str, str]:
        """Segments transcript into Executive Remarks and Q&A sections."""
        logger.debug(f"Segmenting transcript of length {len(raw_text)} characters")
        
        qa_patterns = [
            r"question[- ]and[- ]answer session",
            r"q\s*&\s*a\s*session",
            r"questions and answers",
        ]
        
        split_pattern = re.compile("|".join(qa_patterns), re.IGNORECASE)
        parts = split_pattern.split(raw_text, maxsplit=1)

        remarks = parts[0].strip() if parts else raw_text.strip()
        qa_stream = parts[1].strip() if len(parts) > 1 else ""
        
        logger.info(f"Segmented into remarks ({len(remarks)} chars) and Q&A ({len(qa_stream)} chars)")
        
        return {"remarks": remarks, "qa": qa_stream}

    def extract_sentences(self, text: str) -> List[str]:
        """Splits text into clean individual sentences using SpaCy NLP."""
        logger.debug(f"Extracting sentences from text of length {len(text)} characters")
        
        if not text:
            logger.warning("Empty text provided for sentence extraction")
            return []
            
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        
        logger.info(f"Extracted {len(sentences)} sentences")
        return sentences