"""Consistent, credential-safe errors for the synchronous and async transports."""

import json
import re
from typing import Any, Dict

import aiohttp
import requests
from langchain_core.tools import ToolException

TRANSPORT_ERRORS = (requests.RequestException, aiohttp.ClientError, TimeoutError)


def transport_error(error: BaseException) -> ToolException:
    # Transport messages may contain request URLs, headers or proxy credentials.
    return ToolException(f"Tavily request failed ({type(error).__name__}).")


def _error_detail(body: Any) -> Any:
    if body is None or body == [] or body == {}:
        return None
    if isinstance(body, str):
        return body if body.strip() else None
    if isinstance(body, dict):
        keys = [key for key in ("detail", "error", "message") if key in body]
        for key in keys:
            detail = _error_detail(body[key])
            if detail is not None:
                return detail
        if keys:
            return None
    return body


def response_error(status: int, body: Any, api_key: str) -> ToolException:
    """Keep the HTTP status and server detail without exposing credentials."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            pass
    body = _error_detail(body)
    if body is None:
        body = "Request failed"
    message = body if isinstance(body, str) else json.dumps(body)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+[^\s\"',;<>]+", "Bearer [REDACTED]", message)
    message = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|tavily_api_key|access_token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}<>]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(r"\btvly-[A-Za-z0-9_-]+", "[REDACTED]", message)
    return ToolException(
        f"Tavily API error {status}: {message[:2000] or 'Request failed'}"
    )


def sync_response_error(response: requests.Response, api_key: str) -> ToolException:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response_error(response.status_code, body, api_key)


def validate_response(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise ToolException(
            "Tavily returned an invalid response: expected a JSON object."
        )
    return body


def sync_response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise ToolException("Tavily returned an invalid JSON response.") from None
    return validate_response(body)


def response_json(text: str) -> Dict[str, Any]:
    try:
        body = json.loads(text)
    except ValueError:
        raise ToolException("Tavily returned an invalid JSON response.") from None
    return validate_response(body)
