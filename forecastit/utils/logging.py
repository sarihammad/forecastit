"""Logging configuration with structlog."""

import logging
import sys
from typing import Any

import structlog
from rich.logging import RichHandler

from forecastit.config.settings import Settings


def setup_logging(settings: Settings) -> None:
    """Set up structured logging with structlog.

    Args:
        settings: Application settings containing logging configuration.
    """
    # Configure structlog
    if settings.is_development:
        # Development: human-readable logs with colors
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Set up rich handler for development
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        # Production: JSON logs
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Set up basic logging for production
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper()),
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given module.

    Args:
        name: Module name for the logger.

    Returns:
        Configured structured logger.
    """
    return structlog.get_logger(name)


def log_function_call(func_name: str, **kwargs: Any) -> dict[str, Any]:
    """Log a function call with parameters.

    Args:
        func_name: Name of the function being called.
        **kwargs: Function parameters to log.

    Returns:
        Dictionary of logged parameters.
    """
    logger = get_logger("function_call")
    logger.info("Function called", function=func_name, **kwargs)
    return {"function": func_name, **kwargs}
