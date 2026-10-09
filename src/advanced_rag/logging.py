"""Structured logging: JSON to stdout with a Cloud Logging-compatible `severity` field."""

import logging
import sys
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_configured = False


_SEVERITY = {"warn": "WARNING", "exception": "ERROR", "msg": "INFO"}


def _add_severity(_logger, method_name: str, event_dict: dict) -> dict:
    # Cloud Logging parses `severity` from JSON payloads written to stdout.
    event_dict["severity"] = _SEVERITY.get(method_name, method_name.upper())
    return event_dict


def _add_request_id(_logger, _method_name: str, event_dict: dict) -> dict:
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    global _configured
    if _configured:
        return
    renderer = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.processors.add_log_level,
            _add_severity,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    # Quiet noisy third-party loggers that use stdlib logging.
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(message)s")
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        from advanced_rag.config import get_settings

        s = get_settings()
        configure_logging(s.log_level, s.log_json)
    return structlog.get_logger(name)
