import logging
import os
import sys
from pathlib import Path
from typing import Optional


def mask_sensitive_value(value: Optional[str], visible_chars: int = 4) -> str:
    """Mask a secret value so it is safe to log or display in status checks."""
    if not value:
        return "<not set>"
    if len(value) <= visible_chars * 2:
        return "***"
    return f"{value[:visible_chars]}...{value[-visible_chars:]}"


def setup_logger(
    name: str = "nivaran",
    log_dir: str = "logs",
    log_file: str = "nivaran.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a structured logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Could not set up file logging: %s", str(e))

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger or root application logger."""
    base_logger = logging.getLogger("nivaran")
    if not base_logger.handlers:
        setup_logger()
    if name:
        return logging.getLogger(f"nivaran.{name}")
    return base_logger
