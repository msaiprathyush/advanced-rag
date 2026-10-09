"""Groq LLM access behind a small interface so graph nodes can be tested with a fake."""

import threading
import time
from functools import lru_cache
from typing import Literal, Protocol, TypeVar

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from advanced_rag.clients.retry import with_retry
from advanced_rag.config import get_settings
from advanced_rag.logging import get_logger

log = get_logger(__name__)

Role = Literal["generator", "judge"]
T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, role: Role, messages: list[BaseMessage]) -> str: ...

    def structured(self, role: Role, messages: list[BaseMessage], schema: type[T]) -> T: ...


@lru_cache
def get_chat_model(role: Role) -> ChatGroq:
    s = get_settings()
    return ChatGroq(
        model=s.generator_model if role == "generator" else s.judge_model,
        api_key=s.groq_api_key,
        temperature=0,
        max_retries=0,  # retries handled by our own policy (honours Retry-After)
        timeout=s.llm_timeout_s,
    )


class GroqLLM:
    """Real client: bounded concurrency + retry/backoff + usage logging on every call."""

    def __init__(self, max_concurrency: int | None = None) -> None:
        self._sem = threading.BoundedSemaphore(
            max_concurrency or get_settings().llm_max_concurrency
        )

    @with_retry("groq", attempts=6, max_wait=30)
    def _invoke(self, role: Role, runnable, messages: list[BaseMessage], kind: str):
        with self._sem:
            t0 = time.perf_counter()
            result = runnable.invoke(messages)
            usage = getattr(result, "usage_metadata", None) or {}
            log.info(
                "external_call",
                service="groq",
                role=role,
                kind=kind,
                latency_ms=round((time.perf_counter() - t0) * 1000),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
            return result

    def complete(self, role: Role, messages: list[BaseMessage]) -> str:
        msg = self._invoke(role, get_chat_model(role), messages, "complete")
        return str(msg.content).strip()

    def structured(self, role: Role, messages: list[BaseMessage], schema: type[T]) -> T:
        # json_schema (constrained decoding) was reliable on gpt-oss; function-calling intermittently
        # produced malformed tool-call JSON (Groq 400 tool_use_failed).
        runnable = get_chat_model(role).with_structured_output(schema, method="json_schema")
        return self._invoke(role, runnable, messages, f"structured:{schema.__name__}")


@lru_cache
def get_llm() -> GroqLLM:
    return GroqLLM()
