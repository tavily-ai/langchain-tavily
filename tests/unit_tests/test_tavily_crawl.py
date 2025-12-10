from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_tavily.tavily_crawl import TavilyCrawl


class TestTavilyCrawlToolUnit(ToolsUnitTests):  # Fixed class name to match its purpose
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_tavily._utilities.TavilyCrawlAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[TavilyCrawl]:
        return TavilyCrawl

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyCrawl tool
        return {
            "tavily_api_key": "fake_key_for_testing",
            "max_depth": 1,
            "max_breadth": 20,
            "limit": 50,
            "instructions": "best time to visit japan",
            "select_paths": None,
            "select_domains": None,
            "exclude_paths": None,
            "exclude_domains": None,
            "allow_external": False,
            "include_images": False,
            "categories": None,
            "extract_depth": "basic",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"url": "https://en.wikipedia.org/wiki/Japan"}

    def test_include_usage_controls_crawl_response(self) -> None:
        tool = TavilyCrawl()
        tool.api_wrapper = MagicMock()
        tool.api_wrapper.raw_results.return_value = {
            "results": [{"url": "https://example.com"}],
            "usage": 5,
        }
        result = tool.invoke({"url": "https://example.com"})
        assert "usage" not in result

        tool_with_usage = TavilyCrawl(include_usage=True)
        tool_with_usage.api_wrapper = MagicMock()
        tool_with_usage.api_wrapper.raw_results.return_value = {
            "results": [{"url": "https://example.com"}],
            "usage": 5,
        }
        result_with_usage = tool_with_usage.invoke({"url": "https://example.com"})
        assert result_with_usage["usage"] == 5

        _, kwargs = tool_with_usage.api_wrapper.raw_results.call_args
        assert kwargs["include_usage"] is True
