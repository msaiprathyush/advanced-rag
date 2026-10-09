"""Shared retry policy for every external call (arXiv, Groq).

Free-tier rate limits make retries a correctness concern: a 429 must back off (honouring
`Retry-After` when the server sends it) instead of failing the request or hammering the API.
"""

from collections.abc import Callable
from typing import Any

import groq
import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from advanced_rag.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _status_of(exc: BaseException) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code
    return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | groq.APIConnectionError | groq.APITimeoutError):
        return True
    status = _status_of(exc)
    return status is not None and status in RETRYABLE_STATUS


def retry_after_seconds(exc: BaseException | None) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:  # HTTP-date form; fall back to exponential backoff
        return None


class wait_retry_after(wait_base):
    """Use the server's Retry-After when present, else the fallback strategy."""

    def __init__(self, fallback: wait_base, min_wait: float, max_wait: float) -> None:
        self.fallback = fallback
        self.min_wait = min_wait
        self.max_wait = max_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        ra = retry_after_seconds(exc)
        if ra is not None:
            # Floor at min_wait: a server sending `Retry-After: 0` mid rate-limit must not
            # cause a hot retry loop.
            return max(self.min_wait, min(ra, self.max_wait))
        return self.fallback(retry_state)


def with_retry(
    service: str, *, attempts: int = 5, min_wait: float = 1.0, max_wait: float = 30.0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _before_sleep(rs: RetryCallState) -> None:
        exc = rs.outcome.exception() if rs.outcome else None
        log.warning(
            "external_call_retry",
            service=service,
            attempt=rs.attempt_number,
            sleep_s=round(rs.next_action.sleep, 2) if rs.next_action else None,
            status=_status_of(exc) if exc else None,
            error=type(exc).__name__ if exc else None,
        )

    return retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_retry_after(
            wait_random_exponential(min=min_wait, max=max_wait),
            min_wait=min_wait,
            max_wait=max_wait * 2,
        ),
        stop=stop_after_attempt(attempts),
        before_sleep=_before_sleep,
        reraise=True,
    )
