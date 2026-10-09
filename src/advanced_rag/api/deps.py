import hmac

from fastapi import Header, HTTPException

from advanced_rag.config import get_settings


def require_ingest_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().ingest_api_key.get_secret_value()
    # Fail closed: an unset server key must never let requests through.
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
