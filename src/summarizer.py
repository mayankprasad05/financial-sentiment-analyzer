import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class TranscriptSummarizer:
    """Abstractive text summarization engine using BART/Seq2Seq transformers."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn") -> None:
        logger.info(f"Initializing TranscriptSummarizer with model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("BART summarization model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load summarization model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not load summarization model: {str(e)}") from e

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Generates an abstractive summary of the input transcript text."""
        if not text or not text.strip():
            logger.warning("Empty text provided for summarization.")
            return "No text available to summarize."

        try:
            # Tokenize input text (truncating long transcripts to fit model positional limits)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            input_token_len = inputs["input_ids"].shape[1]
            adjusted_max = min(max_length, max(30, int(input_token_len * 0.5)))
            adjusted_min = min(min_length, max(10, int(adjusted_max * 0.5)))

            # Generate summary sequence via model.generate()
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=adjusted_max,
                    min_length=adjusted_min,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode token IDs back to plain string text
            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary_text.strip()

        except Exception as e:
            logger.error(f"Error during transcript summarization: {e}", exc_info=True)
            return "Summary generation failed due to processing error."