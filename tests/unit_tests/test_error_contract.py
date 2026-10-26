"""Public invocation error semantics with real Requests responses, no network."""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch
from langchain_tavily.tavily_research import TavilyGetResearch, TavilyResearch

KEY = "tvly-test-credential-never-send"
CASES = [
    (
        TavilySearch,
        {"query": "public example"},
        {
            "results": [
                {"url": "https://example.com", "content": "Example", "score": 0.9}
            ]
        },
    ),
    (
        TavilyExtract,
        {"urls": ["https://example.com"]},
        {
            "results": [{"url": "https://example.com", "raw_content": "Example"}],
            "failed_results": [],
        },
    ),
    (
        TavilyMap,
        {"url": "https://example.com"},
        {"results": ["https://example.com/about"]},
    ),
    (
        TavilyCrawl,
        {"url": "https://example.com"},
        {"results": [{"url": "https://example.com", "raw_content": "Example"}]},
    ),
    (
        TavilyResearch,
        {"input": "public example"},
        {"request_id": "test-id", "status": "pending"},
    ),
    (
        TavilyGetResearch,
        {"request_id": "test-id"},
        {
            "request_id": "test-id",
            "status": "completed",
            "content": "Report",
            "sources": [{"url": "https://example.com"}],
        },
    ),
]


@contextmanager
def transport(status=200, body=None, raw=None, failure=None):
    response = requests.Response()
    response.status_code = status
    response._content = (raw if raw is not None else json.dumps(body)).encode()
    response.headers["Content-Type"] = "application/json"
    async_response = MagicMock(status=status, reason="fixture reason")
    async_response.text = AsyncMock(return_value=response.text)
    async_response.__aenter__ = AsyncMock(return_value=async_response)
    async_response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = async_response
    session.get.return_value = async_response
    with patch(
        "langchain_tavily._utilities.requests.post",
        return_value=response,
        side_effect=failure,
    ) as post, patch(
        "langchain_tavily._utilities.requests.get",
        return_value=response,
        side_effect=failure,
    ) as get, patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        factory.return_value.__aenter__.return_value = session
        if failure:
            session.post.side_effect = failure
            session.get.side_effect = failure
        yield post, get, session


async def invoke(tool, args, asynchronous):
    return await tool.ainvoke(args) if asynchronous else tool.invoke(args)


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403, 429, 432, 500])
@pytest.mark.parametrize(
    "body",
    [
        {"detail": {"error": "credit or request problem"}},
        {"detail": "credit or request problem"},
        "credit or request problem",
    ],
)
async def test_http_failure_is_an_exception_with_status_and_detail(
    cls, args, success, asynchronous, status, body
):
    tool = cls(tavily_api_key=KEY)
    assert tool.handle_tool_error is False
    with transport(status, body), pytest.raises(ToolException) as caught:
        await invoke(tool, args, asynchronous)
    assert str(status) in str(caught.value)
    assert "credit or request problem" in str(caught.value)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "raw", ["upstream unavailable", "<html>upstream unavailable</html>", '{"detail":']
)
async def test_non_json_error_preserves_status_and_text(asynchronous, raw):
    with transport(500, raw=raw), pytest.raises(ToolException) as caught:
        await invoke(TavilySearch(tavily_api_key=KEY), {"query": "test"}, asynchronous)
    assert "500" in str(caught.value)
    assert raw in str(caught.value)


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_handled_tool_call_has_error_status_and_redacted_detail(
    cls, args, success, asynchronous
):
    call = {
        "name": cls.model_fields["name"].default,
        "args": args,
        "id": "call-error",
        "type": "tool_call",
    }
    body = {
        "detail": {"error": f"Invalid key {KEY}; Authorization: Bearer another-secret"}
    }
    with transport(401, body):
        result = await invoke(
            cls(tavily_api_key=KEY, handle_tool_error=True), call, asynchronous
        )
    assert isinstance(result, ToolMessage)
    assert result.status == "error" and result.tool_call_id == "call-error"
    assert "401" in result.content and "Invalid key" in result.content
    assert KEY not in result.content and "another-secret" not in result.content


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_success_tool_call_preserves_real_shaped_payload(
    cls, args, success, asynchronous
):
    body = {**success, "usage": {"credits": 0}, "response_time": 0.2}
    call = {
        "name": cls.model_fields["name"].default,
        "args": args,
        "id": "call-ok",
        "type": "tool_call",
    }
    with transport(200, body):
        result = await invoke(cls(tavily_api_key=KEY), call, asynchronous)
    assert isinstance(result, ToolMessage) and result.status == "success"
    assert json.loads(result.content) == body


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_kind", ["timeout", "connection", "unexpected"])
async def test_network_failures_raise_and_programming_errors_stay_unhandled(
    cls, args, success, asynchronous, failure_kind
):
    if failure_kind == "unexpected":
        failure = RuntimeError("synthetic implementation bug")
        expected = RuntimeError
    elif failure_kind == "timeout":
        failure = TimeoutError(KEY) if asynchronous else requests.Timeout(KEY)
        expected = ToolException
    else:
        failure = (
            aiohttp.ClientConnectionError(KEY)
            if asynchronous
            else requests.ConnectionError(KEY)
        )
        expected = ToolException
    with transport(failure=failure), pytest.raises(expected) as caught:
        await invoke(cls(tavily_api_key=KEY), args, asynchronous)
    assert KEY not in str(caught.value)
    if failure_kind == "unexpected":
        assert caught.value is failure


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raw", ["not json", "[]", "null"])
async def test_malformed_success_is_not_success(cls, args, success, asynchronous, raw):
    with transport(200, raw=raw), pytest.raises(ToolException, match="response"):
        await invoke(cls(tavily_api_key=KEY), args, asynchronous)


@pytest.mark.parametrize(
    "cls,args,locked,value",
    [
        (TavilySearch, {"query": "test"}, "max_results", 5),
        (TavilyExtract, {"urls": ["https://example.com"]}, "format", "text"),
        (TavilyMap, {"url": "https://example.com"}, "include_usage", True),
        (TavilyCrawl, {"url": "https://example.com"}, "format", "text"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("same", [False, True])
async def test_constructor_only_invoke_parameter_raises_before_http(
    cls, args, locked, value, asynchronous, same
):
    tool = cls(tavily_api_key=KEY, **({locked: value} if same else {}))
    with transport(200, {"results": ["placeholder"]}) as (
        post,
        get,
        session,
    ), pytest.raises(ValueError, match="only be set during instantiation"):
        await invoke(tool, {**args, locked: value}, asynchronous)
    post.assert_not_called()
    get.assert_not_called()
    session.post.assert_not_called()


@pytest.mark.parametrize("cls,args,success", CASES)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_schema_errors_reject_before_http(cls, args, success, asynchronous):
    args = {key: {"not": "a valid value"} for key in args}
    with transport() as (post, get, session), pytest.raises(ValidationError):
        await invoke(cls(tavily_api_key=KEY), args, asynchronous)
    post.assert_not_called()
    get.assert_not_called()
    session.post.assert_not_called()


@pytest.mark.parametrize(
    "handler", ["Please retry later", lambda error: "Please retry later"]
)
def test_standard_custom_tool_error_handlers(handler):
    call = {
        "name": "tavily_search",
        "args": {"query": "test"},
        "id": "custom",
        "type": "tool_call",
    }
    with transport(429, {"detail": "limit"}):
        result = TavilySearch(tavily_api_key=KEY, handle_tool_error=handler).invoke(
            call
        )
    assert result.status == "error" and result.content == "Please retry later"


@pytest.mark.parametrize(
    "body",
    [
        {"detail": {"api_key": "other-api-key", "message_extra": KEY}},
        {"detail": f"api_key=other-api-key Authorization: Bearer other-token {KEY}"},
    ],
)
def test_http_error_redacts_echoed_credentials(body):
    import traceback

    with transport(401, body), pytest.raises(ToolException) as caught:
        TavilySearch(tavily_api_key=KEY).invoke({"query": "test"})
    text = "".join(traceback.format_exception(caught.value))
    assert KEY not in text and "other-api-key" not in text and "other-token" not in text


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        {"detail": None, "error": {"message": "actionable server detail"}},
        {"detail": {}, "message": "actionable server detail"},
        {"detail": "", "error": [], "message": "actionable server detail"},
        {"detail": {"error": None}, "message": "actionable server detail"},
        {"detail": "  ", "error": {"message": "actionable server detail"}},
        {"detail": {"error": {}, "message": "actionable server detail"}},
    ],
)
async def test_http_detail_falls_back_past_empty_priority_fields(asynchronous, body):
    with transport(401, body), pytest.raises(
        ToolException, match="401.*actionable server detail"
    ):
        await invoke(TavilySearch(tavily_api_key=KEY), {"query": "test"}, asynchronous)
