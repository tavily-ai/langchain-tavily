"""Public capability schema and wire regressions; no provider calls."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from langchain_tavily import TavilyExtract
from langchain_tavily.tavily_research import TavilyGetResearch, TavilyResearch


def response(status, body):
    result = MagicMock(status=status, status_code=status)
    result.json.return_value = body
    result.text = AsyncMock(return_value=json.dumps(body))
    result.__aenter__ = AsyncMock(return_value=result)
    result.__aexit__ = AsyncMock(return_value=False)
    return result


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("usage", [None, False, True])
async def test_get_research_usage_is_available_in_schema_and_query(async_mode, usage):
    tool = TavilyGetResearch(tavily_api_key="test-placeholder")
    assert "include_usage" in tool.args_schema.model_json_schema()["properties"]
    body = {
        "request_id": "id/with?query",
        "status": "completed",
        "content": {"answer": "ok"},
        "sources": [],
        "usage": {"credits": 0},
    }
    args = {"request_id": "id/with?query"}
    if usage is not None:
        args["include_usage"] = usage
    with patch("langchain_tavily._utilities.requests.get") as get, patch(
        "langchain_tavily._utilities.aiohttp.ClientSession"
    ) as factory:
        session = MagicMock()
        factory.return_value.__aenter__.return_value = session
        get.return_value = response(200, body)
        session.get.return_value = response(200, body)
        actual = await tool.ainvoke(args) if async_mode else tool.invoke(args)
        assert actual == body
        call = session.get.call_args if async_mode else get.call_args
        assert call.args[0].endswith("/research/id%2Fwith%3Fquery")
        assert call.kwargs.get("params", {}) == (
            {} if usage is None else {"include_usage": str(usage).lower()}
        )


def test_extract_model_schema_enforces_one_to_twenty_urls_before_http():
    tool = TavilyExtract(tavily_api_key="test-placeholder")
    schema = tool.args_schema.model_json_schema()["properties"]["urls"]
    assert schema["minItems"] == 1
    assert schema["maxItems"] == 20
    with patch("langchain_tavily._utilities.requests.post") as post:
        for urls in [[], ["https://example.com"] * 21]:
            with pytest.raises(ValidationError):
                tool.invoke({"urls": urls})
        post.assert_not_called()


def attachment(text, name="context.md"):
    return {
        "name": name,
        "data": base64.b64encode(text.encode()).decode(),
        "type": "base64",
    }


@pytest.mark.parametrize("constructor", [False, True])
def test_research_attachment_schema_and_limits_preserve_api_objects(constructor):
    file = attachment("context")
    tool = TavilyResearch(
        tavily_api_key="test-placeholder", **({"files": [file]} if constructor else {})
    )
    schema = tool.args_schema.model_json_schema()
    assert {"name", "data", "type"} <= set(
        schema["$defs"]["ResearchFile"]["properties"]
    )
    body = {"request_id": "id", "status": "pending"}
    with patch("langchain_tavily._utilities.requests.post") as post:
        post.return_value = response(201, body)
        assert (
            tool.invoke({"input": "task", **({} if constructor else {"files": [file]})})
            == body
        )
        assert post.call_args.kwargs["json"]["files"] == [file]


@pytest.mark.parametrize(
    "files",
    [
        [attachment("x")] * 6,
        [attachment("word " * 80001)],
        [attachment("word " * 40001), attachment("word " * 40000)],
        [{"name": "context.md", "data": "invalid***"}],
        [attachment("x", "context.pdf")],
    ],
)
@pytest.mark.parametrize("constructor", [False, True])
def test_research_attachment_validation_fails_before_http(files, constructor):
    with patch("langchain_tavily._utilities.requests.post") as post:
        with pytest.raises(ValidationError):
            tool = TavilyResearch(
                tavily_api_key="test-placeholder",
                **({"files": files} if constructor else {}),
            )
            tool.invoke({"input": "task", **({} if constructor else {"files": files})})
        post.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
async def test_constructor_usage_false_overrides_true_invocation(async_mode):
    tool = TavilyGetResearch(tavily_api_key="test-placeholder", include_usage=False)
    with patch.object(type(tool.api_wrapper), "get_research") as sync_get, patch.object(
        type(tool.api_wrapper), "get_research_async", new_callable=AsyncMock
    ) as async_get:
        sync_get.return_value = async_get.return_value = {"status": "pending"}
        args = {"request_id": "id", "include_usage": True}
        if async_mode:
            await tool.ainvoke(args)
        else:
            tool.invoke(args)
        call = async_get.call_args if async_mode else sync_get.call_args
        assert call.kwargs["include_usage"] is False


@pytest.mark.parametrize(
    "texts", [[""], ["word " * 80000], ["word " * 40000, "word " * 40000]]
)
def test_file_boundary_inputs_remain_supported(texts):
    files = [attachment(text) for text in texts]
    tool = TavilyResearch(tavily_api_key="test-placeholder", files=files)
    assert tool.files == files
