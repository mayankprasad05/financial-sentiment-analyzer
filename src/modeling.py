import logging
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Get logger for this module
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """Handles domain-specific sentiment analysis using FinBERT with PyTorch batching.

    model_source can be a HuggingFace model id (for example "ProsusAI/finbert") or a local
    directory containing a fine-tuned model and tokenizer.
    """

    def __init__(self, model_source: str = "ProsusAI/finbert", device: str = None) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing FinancialSentimentAnalyzer from source={model_source} on device={self.device}")
        
        try:
            logger.debug(f"Loading tokenizer from: {model_source}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            logger.debug(f"Loading model from: {model_source}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source).to(self.device)
            self.labels = ["positive", "negative", "neutral"]
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model from {model_source}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model from {model_source}: {str(e)}") from e

    def analyze_sentences(
        self, sentences: List[str], batch_size: int = 16, max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """Batches input text, handles token truncation, and computes probabilities."""
        logger.debug(f"Analyzing {len(sentences)} sentences with batch_size={batch_size}, max_length={max_length}")
        
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
                    max_length=max_length,
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
