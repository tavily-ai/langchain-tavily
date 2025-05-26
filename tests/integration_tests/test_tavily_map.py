from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_tavily.tavily_map import TavilyMap


class TestTavilyMapToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[TavilyMap]:
        return TavilyMap

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyMap tool
        return {
            "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
            "max_depth": 1,
            "max_breadth": 20,
            "limit": 50,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"url": "https://en.wikipedia.org/wiki/Lionel_Messi"}
