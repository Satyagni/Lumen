import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(logs_dir: Path) -> logging.Logger:
    """Configures structured logging writing to console and a rotating log file."""
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "lumen.log"

    logger = logging.getLogger("Lumen")
    logger.setLevel(logging.DEBUG)

    # Prevent adding duplicate handlers if logger is already initialized
    if logger.handlers:
        return logger

    # Formatter for structured logs
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (5MB per file, keeping up to 3 backups)
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to initialize file logger: {e}", file=sys.stderr)

    logger.info("Logger successfully initialized. Log file: %s", log_file)
    return logger

# Create a placeholder logger that can be retrieved before setup
logger = logging.getLogger("Lumen")
