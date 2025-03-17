from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_tavily.tavily_extract import TavilyExtract


class TestTavilyExtractToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[TavilyExtract]:
        return TavilyExtract

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyExtract tool
        return {"extract_depth": "basic", "include_images": False}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"urls": ["https://en.wikipedia.org/wiki/Lionel_Messi"]}
