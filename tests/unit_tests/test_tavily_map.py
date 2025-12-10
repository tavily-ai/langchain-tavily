from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_tavily.tavily_map import TavilyMap


class TestTavilyMapToolUnit(ToolsUnitTests):  # Fixed class name to match its purpose
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_tavily._utilities.TavilyMapAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[TavilyMap]:
        return TavilyMap

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyMap tool
        return {
            "tavily_api_key": "fake_key_for_testing",
            "url": "https://en.wikipedia.org/wiki/Japan",
            "max_depth": 1,
            "max_breadth": 20,
            "limit": 50,
            "instructions": "best time to visit japan",
            "select_paths": None,
            "select_domains": None,
            "exclude_paths": None,
            "exclude_domains": None,
            "allow_external": False,
            "categories": None,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"url": "https://en.wikipedia.org/wiki/Japan"}

    def test_include_usage_forwarded(self) -> None:
        mock_response = {"results": ["https://example.com"]}
        with patch(
            "langchain_tavily.tavily_map.TavilyMapAPIWrapper.raw_results",
            return_value=mock_response,
        ) as mock_raw:
            tool = TavilyMap(tavily_api_key="fake_key_for_testing")
            tool._run(url="https://en.wikipedia.org/wiki/Japan", include_usage=True)

        assert mock_raw.called
        assert mock_raw.call_args.kwargs["include_usage"] is True
