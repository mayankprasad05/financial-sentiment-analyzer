import logging
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import numpy as np

# Get logger for this module
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """Handles domain-specific sentiment analysis using FinBERT with PyTorch batching."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing FinancialSentimentAnalyzer on device: {self.device}")
        
        try:
            logger.debug(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.debug(f"Loading model: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.labels = ["positive", "negative", "neutral"]
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def analyze_sentences(self, sentences: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Batches input text, handles 512 token truncation, and computes probabilities."""
        logger.debug(f"Analyzing {len(sentences)} sentences with batch_size={batch_size}")
        
        if not sentences:
            logger.warning("No sentences provided for analysis")
            return []

        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} sentences")
            
            try:
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                for idx, text in enumerate(batch):
                    scores = probs[idx].cpu().numpy()
                    score_dict = {label: float(scores[j]) for j, label in enumerate(self.labels)}
                    net_score = score_dict["positive"] - score_dict["negative"]
                    dominant_label = max(score_dict, key=score_dict.get)

                    results.append({
                        "sentence": text,
                        "label": dominant_label,
                        "net_sentiment": net_score,
                        "scores": score_dict,
                    })
                    
            except Exception as e:
                logger.error(f"Batch inference failed on batch {i}: {e}", exc_info=True)
                raise RuntimeError(f"Sentiment analysis failed: {str(e)}") from e

        logger.info(f"Successfully analyzed {len(results)} sentences")
        return results


class ExecutiveSummarizer:
    """Generates abstractive summaries using BART Large CNN."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = None) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing ExecutiveSummarizer on device: {self.device}")
        
        try:
            logger.debug(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.debug(f"Loading model: {model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            logger.info("BART model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BART model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load BART: {str(e)}") from e

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Abstractive summarization handling long context safely."""
        
        # Input validation
        if not text or not isinstance(text, str):
            logger.error("Invalid input: text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")
        
        word_count = len(text.split())
        if word_count < 30:
            logger.warning(f"Text too short for summarization ({word_count} words)")
            return "Text too short to generate an abstractive summary."
        
        logger.debug(f"Generating summary for text with {word_count} words (max_length={max_length})")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            logger.debug(f"Tokenized input: {inputs['input_ids'].shape}")
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode to text
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            if not summary or summary.isspace():
                logger.warning("Summary generation produced empty result")
                return "Unable to generate summary from provided text."
            
            logger.info(f"Generated summary ({len(summary.split())} words)")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            raise RuntimeError(f"Summarization failed: {str(e)}") from e