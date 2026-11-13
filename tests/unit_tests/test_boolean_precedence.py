from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch


@pytest.mark.parametrize(
    ("tool_class", "wrapper_attribute", "parameter", "arguments"),
    [
        (TavilySearch, "api_wrapper", "include_images", {"query": "test"}),
        (
            TavilyExtract,
            "apiwrapper",
            "include_images",
            {"urls": ["https://example.com"]},
        ),
        (TavilyCrawl, "api_wrapper", "include_images", {"url": "https://example.com"}),
        (TavilyCrawl, "api_wrapper", "allow_external", {"url": "https://example.com"}),
        (TavilyMap, "api_wrapper", "allow_external", {"url": "https://example.com"}),
    ],
)
@pytest.mark.parametrize(
    ("configured", "invoked", "expected"),
    [
        (False, True, False),
        (False, None, False),
        (True, False, True),
        (None, True, True),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_configured_booleans_take_precedence(
    tool_class: type[BaseTool],
    wrapper_attribute: str,
    parameter: str,
    arguments: dict,
    configured: Optional[bool],
    invoked: Optional[bool],
    expected: bool,
    asynchronous: bool,
) -> None:
    tool = tool_class(tavily_api_key="test-key", **{parameter: configured})
    wrapper = getattr(tool, wrapper_attribute)
    method = "raw_results_async" if asynchronous else "raw_results"
    response = {"results": [{"url": "https://example.com", "content": "test"}]}
    with patch.object(
        type(wrapper),
        method,
        return_value=response,
        new_callable=AsyncMock if asynchronous else MagicMock,
    ) as raw:
        arguments = {**arguments, parameter: invoked}
        if asynchronous:
            result = await tool.ainvoke(arguments)
            raw.assert_awaited_once()
        else:
            result = tool.invoke(arguments)
            raw.assert_called_once()
        assert result == response
        assert raw.call_args.kwargs[parameter] is expected
