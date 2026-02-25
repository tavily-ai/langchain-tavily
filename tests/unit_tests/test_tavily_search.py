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


class TestTavilySearchExactMatch:
    """Tests for exact_match parameter."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        patcher = patch(
            "langchain_tavily._utilities.TavilySearchAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()
        request.addfinalizer(patcher.stop)
        return mock_validate

    def test_exact_match_passed_to_api(self) -> None:
        """Test that exact_match is correctly passed to the API wrapper."""
        with patch(
            "langchain_tavily._utilities.TavilySearchAPIWrapper.raw_results"
        ) as mock_raw_results:
            mock_raw_results.return_value = {
                "query": "test",
                "results": [{"title": "Test", "url": "https://test.com"}],
            }
            tool = TavilySearch(tavily_api_key="fake_key")
            tool.invoke({"query": '"John Smith" CEO', "exact_match": True})

            mock_raw_results.assert_called_once()
            call_kwargs = mock_raw_results.call_args[1]
            assert call_kwargs["exact_match"] is True

    def test_exact_match_not_sent_when_none(self) -> None:
        """Test that exact_match is not sent when not specified."""
        with patch(
            "langchain_tavily._utilities.TavilySearchAPIWrapper.raw_results"
        ) as mock_raw_results:
            mock_raw_results.return_value = {
                "query": "test",
                "results": [{"title": "Test", "url": "https://test.com"}],
            }
            tool = TavilySearch(tavily_api_key="fake_key")
            tool.invoke({"query": "general search"})

            mock_raw_results.assert_called_once()
            call_kwargs = mock_raw_results.call_args[1]
            assert call_kwargs["exact_match"] is None

    def test_exact_match_instantiation(self) -> None:
        """Test that exact_match can be set during instantiation."""
        with patch(
            "langchain_tavily._utilities.TavilySearchAPIWrapper.raw_results"
        ) as mock_raw_results:
            mock_raw_results.return_value = {
                "query": "test",
                "results": [{"title": "Test", "url": "https://test.com"}],
            }
            tool = TavilySearch(tavily_api_key="fake_key", exact_match=True)
            tool.invoke({"query": "test query"})

            mock_raw_results.assert_called_once()
            call_kwargs = mock_raw_results.call_args[1]
            assert call_kwargs["exact_match"] is True

