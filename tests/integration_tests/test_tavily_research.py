from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_tavily.tavily_research import TavilyResearch, TavilyGetResearch


class TestTavilyResearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[TavilyResearch]:
        return TavilyResearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyResearch tool
        return {
            "model": "tvly-mini",
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"input": "Research the latest developments in AI"}


class TestTavilyGetResearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[TavilyGetResearch]:
        return TavilyGetResearch

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyGetResearch tool
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"request_id": "test-request-id"}
