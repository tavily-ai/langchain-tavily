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


@pytest.fixture(autouse=True)
def _patch_api_key_validation(request):
    """Patch API key validation for all tests in this module."""
    patcher = patch(
        "langchain_tavily._utilities.TavilySearchAPIWrapper.validate_environment"
    )
    patcher.start()
    request.addfinalizer(patcher.stop)


class TestExactMatch:
    """Tests for exact_match parameter support."""

    def test_exact_match_accepted_at_instantiation(self):
        """exact_match can be set when creating the tool."""
        tool = TavilySearch(
            tavily_api_key="fake_key",
            exact_match=True,
        )
        assert tool.exact_match is True

    def test_exact_match_default_is_none(self):
        """exact_match defaults to None when not set."""
        tool = TavilySearch(tavily_api_key="fake_key")
        assert tool.exact_match is None

    def test_exact_match_not_in_input_schema(self):
        """exact_match should NOT be in the agent-facing input schema."""
        schema = TavilySearch.model_fields["args_schema"].default
        field_names = list(schema.model_fields.keys())
        assert "exact_match" not in field_names

    def test_exact_match_in_forbidden_params(self):
        """exact_match is rejected if passed at invocation via kwargs."""
        tool = TavilySearch(tavily_api_key="fake_key")
        result = tool._run(query="test", exact_match=True)
        assert "error" in result
        assert isinstance(result["error"], ValueError)
        assert "exact_match" in str(result["error"])

    @patch("langchain_tavily._utilities.requests.post")
    def test_exact_match_passed_to_api(self, mock_post):
        """exact_match=True is included in the API request payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": "test",
            "results": [{"title": "t", "url": "u", "content": "c", "score": 1}],
        }
        mock_post.return_value = mock_response

        tool = TavilySearch(tavily_api_key="fake_key", exact_match=True)
        tool._run(query='"Sam Altman" CEO')

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["exact_match"] is True

    @patch("langchain_tavily._utilities.requests.post")
    def test_exact_match_none_excluded_from_payload(self, mock_post):
        """When exact_match is None (default), it's not sent to the API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": "test",
            "results": [{"title": "t", "url": "u", "content": "c", "score": 1}],
        }
        mock_post.return_value = mock_response

        tool = TavilySearch(tavily_api_key="fake_key")
        tool._run(query="test query")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "exact_match" not in payload

    @patch("langchain_tavily._utilities.requests.post")
    def test_exact_match_false_included_in_payload(self, mock_post):
        """When exact_match=False, it's explicitly sent to the API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": "test",
            "results": [{"title": "t", "url": "u", "content": "c", "score": 1}],
        }
        mock_post.return_value = mock_response

        tool = TavilySearch(tavily_api_key="fake_key", exact_match=False)
        tool._run(query="test query")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["exact_match"] is False

