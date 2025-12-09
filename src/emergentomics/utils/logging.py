"""
Logging configuration for Prophetic Emergentomics.

Uses structlog for structured logging with rich console output.
"""

import logging
import sys
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from emergentomics.core.config import get_settings


def setup_logging(
    level: Optional[str] = None,
    json_output: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Uses settings if None.
        json_output: If True, output JSON instead of human-readable logs.
    """
    settings = get_settings()
    level = level or settings.log_level

    # Configure standard logging
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Rich handler for pretty console output
    console = Console(stderr=True)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug,
    )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[rich_handler],
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger for a module."""
    return structlog.get_logger(name)
