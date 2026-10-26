"""Exercise current public options through real tool and HTTP request mapping."""

import json
from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch
from langchain_tavily.tavily_research import TavilyGetResearch, TavilyResearch

RESPONSE = {
    "results": [{"url": "https://example.com", "content": "synthetic content"}],
    "failed_results": [],
    "request_id": "test-request",
    "usage": {"credits": 0},
    "sources": [{"url": "https://example.com"}],
}
CASES = [
    (
        TavilySearch,
        {"query": "synthetic query", "include_domains": ["example.com"]},
        {
            "language": "en",
            "include_published_date": True,
            "filter_by_published_date": False,
            "filter_by_language": False,
            "safe_search": False,
            "include_domains_mode": "prefer",
            "chunks_per_source": 2,
        },
    ),
    (
        TavilyExtract,
        {"urls": ["https://example.com"], "query": "synthetic"},
        {
            "timeout": 30.0,
            "chunks_per_source": 2,
        },
    ),
    (
        TavilyCrawl,
        {"url": "https://example.com", "instructions": "synthetic"},
        {
            "timeout": 120.0,
            "chunks_per_source": 2,
        },
    ),
    (TavilyMap, {"url": "https://example.com"}, {"timeout": 120.0}),
    (
        TavilyResearch,
        {"input": "synthetic research"},
        {
            "include_domains": ["example.com"],
            "exclude_domains": [],
            "output_length": "short",
            "files": [],
        },
    ),
]


def response_mock() -> MagicMock:
    response = MagicMock(status=200, status_code=200)
    response.json.return_value = RESPONSE
    response.text = AsyncMock(return_value=json.dumps(RESPONSE))
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


@pytest.mark.parametrize("tool_type,required,options", CASES)
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_current_options_and_response(
    tool_type: Type[BaseTool],
    required: dict,
    options: dict,
    configured: bool,
    asynchronous: bool,
) -> None:
    tool = tool_type(
        tavily_api_key="test-placeholder", **(options if configured else {})
    )
    arguments = {**required, **({} if configured else options)}
    # Constructor settings retain precedence, including false and empty lists.
    if configured:
        arguments.update({key: True for key in options if key.endswith("search")})
    with patch("langchain_tavily._utilities.requests.post") as post, patch(
        "langchain_tavily._utilities.aiohttp.ClientSession"
    ) as factory:
        post.return_value = response_mock()
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        session.post.return_value = response_mock()
        result = (
            await tool.ainvoke(arguments) if asynchronous else tool.invoke(arguments)
        )
        assert result == RESPONSE
        request = session.post.call_args if asynchronous else post.call_args
        for key, value in options.items():
            assert request.kwargs["json"][key] == value
        assert request.kwargs["headers"]["X-Client-Name"] == "langchain-tavily"
        assert request.kwargs["headers"]["X-Client-Source"] == "api"
    schema = tool.get_input_schema().model_json_schema()["properties"]
    assert options.keys() <= schema.keys()


@pytest.mark.parametrize(
    "tool_type,arguments",
    [
        (TavilySearch, {"query": "synthetic"}),
        (TavilyExtract, {"urls": ["https://example.com"]}),
        (TavilyCrawl, {"url": "https://example.com"}),
        (TavilyMap, {"url": "https://example.com"}),
        (TavilyResearch, {"input": "synthetic"}),
        (TavilyGetResearch, {"request_id": "test-request"}),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_configured_headers_and_base_url(
    tool_type: Type[BaseTool],
    arguments: dict,
    asynchronous: bool,
) -> None:
    tool = tool_type(
        tavily_api_key="test-placeholder",
        client_name="custom-integration",
        project_id="test-project",
        session_id="test-session",
        human_id="test-human",
        api_base_url="https://synthetic.invalid",
    )
    method = "get" if tool_type is TavilyGetResearch else "post"
    with patch(f"langchain_tavily._utilities.requests.{method}") as request, patch(
        "langchain_tavily._utilities.aiohttp.ClientSession"
    ) as factory:
        request.return_value = response_mock()
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        getattr(session, method).return_value = response_mock()
        result = (
            await tool.ainvoke(arguments) if asynchronous else tool.invoke(arguments)
        )
        assert result == RESPONSE
        call = getattr(session, method).call_args if asynchronous else request.call_args
        assert call.args[0].startswith("https://synthetic.invalid/")
        headers = call.kwargs["headers"]
        assert headers["X-Client-Name"] == "custom-integration"
        assert headers["X-Client-Source"] == "api"
        assert headers["X-Project-ID"] == "test-project"
        assert headers["X-Session-Id"] == "test-session"
        assert headers["X-Human-Id"] == "test-human"
