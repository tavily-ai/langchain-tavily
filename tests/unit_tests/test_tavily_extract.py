from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_tavily.tavily_extract import TavilyExtract


class TestTavilyExtractToolUnit(ToolsUnitTests):
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request: pytest.FixtureRequest) -> MagicMock:
        # Patch the validation_environment class method
        patcher = patch(
            "langchain_tavily._utilities.TavilyExtractAPIWrapper.validate_environment"
        )
        mock_validate = patcher.start()

        # Use pytest's cleanup mechanism
        request.addfinalizer(patcher.stop)
        return mock_validate

    @property
    def tool_constructor(self) -> Type[TavilyExtract]:
        return TavilyExtract

    @property
    def tool_constructor_params(self) -> dict:
        # Parameters for initializing the TavilyExtract tool
        return {
            "tavily_api_key": "fake_key_for_testing",
            "extract_depth": "basic",
            "include_images": False,
        }

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"urls": ["https://en.wikipedia.org/wiki/Lionel_Messi"]}
