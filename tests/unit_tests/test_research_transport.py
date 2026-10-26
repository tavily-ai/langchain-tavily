"""Research transport regressions using synthetic responses and no network."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from langchain_tavily.tavily_research import TavilyGetResearch, TavilyResearch

RESULT = {"request_id": "test-request", "status": "pending"}
OPTIONS = {
    "input": "A synthetic research task",
    "model": "mini",
    "include_domains": ["example.com"],
    "exclude_domains": ["excluded.example.com"],
    "output_length": "short",
    "files": [{"name": "context.txt", "data": "dGVzdA==", "type": "base64"}],
}


def response_mock(status: int, body: dict) -> MagicMock:
    response = MagicMock(status=status, status_code=status, reason="test response")
    response.json.return_value = body
    response.text = AsyncMock(return_value=json.dumps(body))
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


@pytest.mark.parametrize("status", [200, 201])
def test_research_create_success(status: int) -> None:
    tool = TavilyResearch(tavily_api_key="test-placeholder")
    with patch("langchain_tavily._utilities.requests.post") as post:
        post.return_value = response_mock(status, RESULT)
        assert tool.invoke(OPTIONS) == RESULT
        assert post.call_args.kwargs["json"] == {
            **OPTIONS,
            "stream": False,
            "citation_format": "numbered",
        }


@pytest.mark.parametrize("status", [200, 201])
async def test_research_create_async_success(status: int) -> None:
    tool = TavilyResearch(tavily_api_key="test-placeholder")
    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        session.post.return_value = response_mock(status, RESULT)
        assert await tool.ainvoke(OPTIONS) == RESULT
        assert session.post.call_args.kwargs["json"] == {
            **OPTIONS,
            "stream": False,
            "citation_format": "numbered",
        }


@pytest.mark.parametrize("status", [400, 429, 500])
def test_research_create_rejects_failure(status: int) -> None:
    with patch("langchain_tavily._utilities.requests.post") as post:
        post.return_value = response_mock(status, {"detail": {"error": "test error"}})
        with pytest.raises(ToolException, match=str(status)):
            TavilyResearch(tavily_api_key="test-placeholder").invoke(OPTIONS)


@pytest.mark.parametrize("status", [400, 429, 500])
async def test_research_create_async_rejects_failure(status: int) -> None:
    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        session.post.return_value = response_mock(status, {})
        with pytest.raises(ToolException, match=str(status)):
            await TavilyResearch(tavily_api_key="test-placeholder").ainvoke(OPTIONS)


@pytest.mark.parametrize("status", [200, 202])
def test_get_research_preserves_response(status: int) -> None:
    result = {**RESULT, "sources": [{"url": "https://example.com", "title": "Example"}]}
    with patch("langchain_tavily._utilities.requests.get") as get:
        get.return_value = response_mock(status, result)
        tool = TavilyGetResearch(tavily_api_key="test-placeholder")
        assert tool.invoke({"request_id": "test-request"}) == result


@pytest.mark.parametrize("legacy", [False, True])
def test_documented_and_legacy_research_constructor_options(legacy: bool) -> None:
    schema = {"properties": {"answer": {"type": "string"}}}
    settings = (
        {"stream_results": False, "research_output_schema": schema}
        if legacy
        else {"stream": False, "output_schema": schema}
    )
    with patch("langchain_tavily._utilities.requests.post") as post:
        post.return_value = response_mock(201, RESULT)
        tool = TavilyResearch(tavily_api_key="test-placeholder", **settings)
        assert tool.invoke({"input": "synthetic"}) == RESULT
        assert post.call_args.kwargs["json"]["output_schema"] == schema
        assert post.call_args.kwargs["json"]["stream"] is False


def test_research_stream_keeps_chunks_and_closes_response() -> None:
    chunks = [b"event: mess", b"age\ndata: test\n\n"]
    with patch("langchain_tavily._utilities.requests.post") as post:
        response = response_mock(200, {})
        response.iter_content.return_value = iter(chunks)
        post.return_value = response
        tool = TavilyResearch(tavily_api_key="test-placeholder", stream=True)
        assert list(tool.invoke({"input": "synthetic"})) == chunks
        response.close.assert_called_once()
        assert post.call_args.kwargs["headers"]["X-Client-Name"] == "langchain-tavily"
        assert post.call_args.kwargs["headers"]["X-Client-Source"] == "api"


async def test_research_async_stream_keeps_chunks_and_closes_session() -> None:
    chunks = [b"event: mess", b"age\ndata: test\n\n"]

    async def iter_chunks():
        for chunk in chunks:
            yield chunk

    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        response = response_mock(200, {})
        response.content.iter_any.return_value = iter_chunks()
        session.post.return_value = response
        tool = TavilyResearch(tavily_api_key="test-placeholder", stream=True)
        stream = await tool.ainvoke({"input": "synthetic"})
        assert [chunk async for chunk in stream] == chunks
        response.__aexit__.assert_awaited_once()
        factory.return_value.__aexit__.assert_awaited_once()
        assert (
            session.post.call_args.kwargs["headers"]["X-Client-Name"]
            == "langchain-tavily"
        )


@pytest.mark.parametrize("status", [200, 202])
async def test_get_research_async_preserves_response(status: int) -> None:
    result = {**RESULT, "sources": [{"url": "https://example.com", "title": "Example"}]}
    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        session.get.return_value = response_mock(status, result)
        tool = TavilyGetResearch(tavily_api_key="test-placeholder")
        assert await tool.ainvoke({"request_id": "test-request"}) == result


@pytest.mark.parametrize("status", [400, 401, 403, 429, 432, 500])
def test_stream_http_failure_closes_response_and_raises(status: int) -> None:
    with patch("langchain_tavily._utilities.requests.post") as post:
        response = response_mock(status, {"detail": "stream rejected"})
        post.return_value = response
        tool = TavilyResearch(tavily_api_key="test-placeholder", stream=True)
        with pytest.raises(ToolException, match=f"{status}.*stream rejected"):
            tool.invoke({"input": "synthetic"})
        response.close.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 432, 500])
async def test_async_stream_http_failure_raises_during_iteration(status: int) -> None:
    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        response = response_mock(status, {"detail": "stream rejected"})
        session.post.return_value = response
        tool = TavilyResearch(
            tavily_api_key="test-placeholder", stream=True, handle_tool_error=True
        )
        stream = await tool.ainvoke({"input": "synthetic"})
        with pytest.raises(ToolException, match=f"{status}.*stream rejected"):
            _ = [chunk async for chunk in stream]
        response.__aexit__.assert_awaited_once()
        factory.return_value.__aexit__.assert_awaited_once()


def test_stream_transport_failure_is_safe_and_closes_response() -> None:
    import requests

    def chunks():
        yield b"partial"
        raise requests.ConnectionError("Bearer synthetic-secret")

    with patch("langchain_tavily._utilities.requests.post") as post:
        response = response_mock(200, {})
        response.iter_content.return_value = chunks()
        post.return_value = response
        stream = TavilyResearch(tavily_api_key="test-placeholder", stream=True).invoke(
            {"input": "synthetic"}
        )
        assert next(stream) == b"partial"
        with pytest.raises(ToolException, match="ConnectionError") as caught:
            next(stream)
        assert "synthetic-secret" not in str(caught.value)
        response.close.assert_called_once()


async def test_async_stream_transport_failure_is_safe_and_closes_session() -> None:
    import aiohttp

    async def chunks():
        yield b"partial"
        raise aiohttp.ClientConnectionError("Bearer synthetic-secret")

    with patch("langchain_tavily._utilities.aiohttp.ClientSession") as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        response = response_mock(200, {})
        response.content.iter_any.return_value = chunks()
        session.post.return_value = response
        stream = await TavilyResearch(
            tavily_api_key="test-placeholder", stream=True
        ).ainvoke({"input": "synthetic"})
        assert await anext(stream) == b"partial"
        with pytest.raises(ToolException, match="ClientConnectionError") as caught:
            await anext(stream)
        assert "synthetic-secret" not in str(caught.value)
        response.__aexit__.assert_awaited_once()
        factory.return_value.__aexit__.assert_awaited_once()
