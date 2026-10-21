import logging
from typing import Optional
from earningscall import get_company

logger = logging.getLogger(__name__)


def fetch_transcript_by_ticker(symbol: str, year: int, quarter: int) -> Optional[str]:
    """Fetches the earnings call transcript text for a given ticker, year, and quarter."""
    clean_symbol = symbol.strip().upper()
    logger.info(f"Attempting to fetch transcript for {clean_symbol} (Q{quarter} {year})...")

    try:
        company = get_company(clean_symbol.lower())
        transcript = company.get_transcript(year=year, quarter=quarter)

        if not transcript:
            logger.warning(f"No transcript returned for {clean_symbol} Q{quarter} {year}.")
            return None

        raw_text = getattr(transcript, "text", None)
        if raw_text and raw_text.strip():
            logger.info(f"Successfully retrieved transcript text for {clean_symbol}.")
            return raw_text.strip()

        logger.warning(f"Transcript text was empty for {clean_symbol} Q{quarter} {year}.")
        return None

    except Exception as e:
        logger.error(f"Error fetching transcript for {clean_symbol}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to fetch transcript for {clean_symbol}: {str(e)}") from e