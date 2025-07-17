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


def test_tavily_search_date_parameters():
    """Test that start_date and end_date parameters are properly handled."""
    with patch("langchain_tavily._utilities.TavilySearchAPIWrapper.validate_environment"):
        # Create TavilySearch instance
        tool = TavilySearch(tavily_api_key="fake_key_for_testing")
        
        # Mock the api_wrapper.raw_results method
        with patch.object(tool.api_wrapper, 'raw_results') as mock_raw_results:
            mock_raw_results.return_value = {
                "results": [{"title": "Test", "url": "http://test.com", "content": "Test content"}],
                "query": "test query"
            }
            
            # Test with date parameters
            result = tool._run(
                query="test query",
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
            
            # Verify the method was called with date parameters
            mock_raw_results.assert_called_once()
            call_args = mock_raw_results.call_args
            assert call_args.kwargs["start_date"] == "2024-01-01"
            assert call_args.kwargs["end_date"] == "2024-12-31"
