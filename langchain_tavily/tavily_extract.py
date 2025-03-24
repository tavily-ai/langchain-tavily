"""Tool for the Tavily Extract API."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilyExtractAPIWrapper


class TavilyExtractInput(BaseModel):
    """
    Input for [TavilyExtract]
    Extract web page content from one or more specified URLs using Tavily Extract.
    """

    urls: List[str] = Field(description="list of urls to extract")
    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls the thoroughness of web content extraction.
        
        Use "basic" for faster extraction of main text content.
        
        Use "advanced" (default) to retrieve comprehensive content including 
        tables and embedded elements. Always use "advanced" for LinkedIn 
        and YouTube URLs for optimal results.
        
        Better for complex websites but may increase response time.
        """,  # noqa: E501
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="""Determines whether to extract and include images from the source URLs.
   
        Set to True when visualizations are needed for better context or understanding.
    
        Default is False (extracts text content only).
        """,  # noqa: E501
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    if params.get("extract_depth") and params["extract_depth"] == "basic":
        suggestions.append(
            "Try a more detailed extraction using 'advanced' extract_depth"
        )

    return suggestions


class TavilyExtract(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Extract API with dynamically settable parameters."""

    name: str = "tavily_extract"
    description: str = (
        "Extracts comprehensive content from web pages based on provided URLs. "
        "This tool retrieves raw text of a web page, with an option to include images. "
        "It supports two extraction depths: 'basic' for standard text extraction and "
        "'advanced' for a more comprehensive extraction with higher success rate. "
        "Ideal for use cases such as content curation, data ingestion for NLP models, "
        "and automated information retrieval, this endpoint seamlessly integrates into "
        "your content processing pipeline. Input should be a list of one or more URLs."
    )

    args_schema: Type[BaseModel] = TavilyExtractInput
    handle_tool_error: bool = True

    # Default parameters
    extract_depth: Optional[Literal["basic", "advanced"]] = "basic"
    """The depth of the extraction process. 
    'advanced' extraction retrieves more data than 'basic',
    with higher success but may increase latency.
    
    Default is 'basic'
    """
    include_images: Optional[bool] = False
    """Include a list of images extracted from the URLs in the response.
    
    Default is False
    """

    apiwrapper: TavilyExtractAPIWrapper = Field(default_factory=TavilyExtractAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create apiwrapper with tavily_api_key if provided
        if "tavily_api_key" in kwargs:
            kwargs["apiwrapper"] = TavilyExtractAPIWrapper(
                tavily_api_key=kwargs["tavily_api_key"]
            )
        super().__init__(**kwargs)

    def _run(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        try:
            # Execute search with parameters directly
            raw_results = self.apiwrapper.raw_results(
                urls=urls,
                extract_depth=extract_depth if extract_depth else self.extract_depth,
                include_images=include_images
                if include_images
                else self.include_images,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth
                    if extract_depth
                    else self.extract_depth,
                    "include_images": include_images
                    if include_images
                    else self.include_images,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}

    async def _arun(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.apiwrapper.raw_results_async(
                urls=urls,
                extract_depth=extract_depth if extract_depth else self.extract_depth,
                include_images=include_images
                if include_images
                else self.include_images,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "urls": urls,
                    "extract_depth": extract_depth
                    if extract_depth
                    else self.extract_depth,
                    "include_images": include_images
                    if include_images
                    else self.include_images,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}
