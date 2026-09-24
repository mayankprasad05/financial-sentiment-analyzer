# logging_config.py
import logging
from pathlib import Path

def setup_logging(log_level=logging.ERROR):
    """
    Configure logging for the entire application.
    Only shows ERROR and CRITICAL messages.
    """
    
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
    console_handler.setLevel(log_level)  # ✅ Console only gets ERRORS
    console_formatter = logging.Formatter(
        "%(levelname)s: %(name)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger