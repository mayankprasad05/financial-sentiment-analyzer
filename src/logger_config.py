# logging_config.py
import logging
from pathlib import Path
from typing import Optional

def setup_logging(log_level=logging.ERROR, console_level: Optional[int] = None):
    """
    Configure logging for the entire application.

    ``log_level`` controls what reaches the log file and the root logger.
    ``console_level`` independently controls what reaches the terminal; when omitted it
    matches ``log_level``, which preserves the original behaviour. Raising only the
    console level keeps the full record in logs/app.log while leaving the terminal quiet.
    """

    if console_level is None:
        console_level = log_level
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # File handler (writes to file)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)  # File still gets ALL messages
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (writes to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)  # ✅ Console respects console_level
    console_formatter = logging.Formatter(
        "%(levelname)s: %(name)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


# Third-party loggers that otherwise interleave HTTP and model-loading chatter into
# CLI output (for example "INFO: httpx2 - HTTP Request: GET https://huggingface.co/...").
_NOISY_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "absl",
    "filelock",
    "datasets",
    "evaluate",
    "matplotlib",
)


def silence_third_party():
    """Quiet third-party libraries so CLI output holds only the caller's own output.

    Drops library log records to ERROR and switches off the HuggingFace
    weight-loading and download progress bars. Safe to call more than once.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        logging.getLogger(__name__).debug(
            "transformers unavailable; progress bars left unchanged"
        )