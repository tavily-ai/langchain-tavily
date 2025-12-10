from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_tavily.tavily_search import TavilySearch


class TestTavilySearchToolUnit(ToolsUnitTests):  # Fixed class name to match its purpose
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_tavily._utilities.TavilySearchAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[TavilySearch]:
        return TavilySearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilySearch tool
        return {
            "tavily_api_key": "fake_key_for_testing",
            "max_results": 5,
            "search_depth": "basic",
            "include_answer": False,
            "include_images": False,
            "topic": "general",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"query": "best time to visit japan"}

    def test_include_usage_controls_response_payload(self) -> None:
        tool = TavilySearch()
        tool.api_wrapper = MagicMock()
        tool.api_wrapper.raw_results.return_value = {
            "results": [{"title": "example"}],
            "usage": 3,
        }

        result = tool.invoke({"query": "best cafes"})
        assert "usage" not in result

        tool_with_usage = TavilySearch(include_usage=True)
        tool_with_usage.api_wrapper = MagicMock()
        tool_with_usage.api_wrapper.raw_results.return_value = {
            "results": [{"title": "example"}],
            "usage": 4,
        }
        result_with_usage = tool_with_usage.invoke({"query": "best cafes"})
        assert result_with_usage["usage"] == 4

        _, kwargs = tool_with_usage.api_wrapper.raw_results.call_args
        assert kwargs["include_usage"] is True
