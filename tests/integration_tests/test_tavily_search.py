from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_tavily.tavily_search import TavilySearch


class TestTavilySearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[TavilySearch]:
        return TavilySearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilySearch tool
        return {
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

    def test_search_with_date_range(self) -> None:
        """Test TavilySearch with date range parameters."""
        tool = TavilySearch()
        result = tool.invoke(
            {
                "query": "AI developments in 2024",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        )
        assert isinstance(result, dict)
        assert "query" in result
