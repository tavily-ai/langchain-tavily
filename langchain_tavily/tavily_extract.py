"""Tool for the Tavily Extract API."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_tavily._errors import TRANSPORT_ERRORS, transport_error
from langchain_tavily._utilities import TavilyExtractAPIWrapper


class TavilyExtractInput(BaseModel):
    """
    Input for [TavilyExtract]
    Extract web page content from one or more specified URLs using Tavily Extract.
    """

    model_config = ConfigDict(extra="allow")

    chunks_per_source: Optional[int] = Field(
        default=None,
        description="Content chunks per source (1-5); requires query/instructions."
    )

    timeout: Optional[float] = Field(
        default=None,
        description='API extraction timeout in seconds (1-60).',
    )

    urls: List[str] = Field(min_length=1, max_length=20, description="One to twenty URLs to extract")
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
    query: Optional[str] = Field(
        default=None,
        description="""A query string to filter and prioritize content extraction.
        
        When provided, the extraction will focus on content most relevant to the query.
        """,  # noqa: E501
    )


def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
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

    # Default parameters
    extract_depth: Optional[Literal["basic", "advanced"]] = None
    """The depth of the extraction process. 
    'advanced' extraction retrieves more data than 'basic',
    with higher success but may increase latency.
    
    Default is 'basic'
    """
    include_images: Optional[bool] = None
    """Include a list of images extracted from the URLs in the response.
    
    Default is False
    """
    format: Optional[str] = None
    """
    The format of the extracted web page content. markdown returns content in markdown 
    format. text returns plain text and may increase latency.
    
    Default is 'markdown'
    """
    include_favicon: Optional[bool] = None
    """Whether to include the favicon URL for each result.
    
    Default is False.
    """
    include_usage: Optional[bool] = None
    """Whether to include credit usage information in the response.
    
    Default is False.
    """
    query: Optional[str] = None
    """A query string to filter and prioritize content extraction.
    
    When provided, the extraction will focus on content most relevant to the query.
    """
    chunks_per_source: Optional[int] = None
    """Number of content chunks to return per source URL.
    
    Use this to limit the amount of content returned from each URL.
    """
    timeout: Optional[float] = None

    apiwrapper: TavilyExtractAPIWrapper = Field(default_factory=TavilyExtractAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        wrapper_kwargs = {
            key: kwargs[key]
            for key in (
                "tavily_api_key", "api_base_url", "client_name",
                "project_id", "session_id", "human_id",
            )
            if key in kwargs
        }
        if wrapper_kwargs:
            kwargs["apiwrapper"] = TavilyExtractAPIWrapper(**wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        query: Optional[str] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Use the tool."""

        try:
            forbidden_params = [
                "include_usage", "include_favicon", "format"
            ]
            for param in forbidden_params:
                if param in kwargs:
                    raise ValueError(
                        f"The parameter '{param}' can only be set during instantiation, not during invocation. Please set it when creating the TavilyExtract instance."
                    )
            
            # Execute search with parameters directly
            for key in (
                'timeout',
            ):
                value = getattr(self, key)
                if value is not None:
                    kwargs[key] = value

            raw_results = self.apiwrapper.raw_results(
                urls=urls,
                extract_depth=self.extract_depth
                if self.extract_depth
                else extract_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                include_favicon=self.include_favicon,
                format=self.format,
                include_usage=self.include_usage,
                query=self.query if self.query else query,
                chunks_per_source=(
                    self.chunks_per_source
                    if self.chunks_per_source is not None
                    else chunks_per_source
                ),
                **kwargs,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
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
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None

    async def _arun(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        query: Optional[str] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""

        try:
            forbidden_params = [
                "include_usage", "include_favicon", "format"
            ]
            for param in forbidden_params:
                if param in kwargs:
                    raise ValueError(
                        f"The parameter '{param}' can only be set during instantiation, not during invocation. Please set it when creating the TavilyExtract instance."
                    )
            
            for key in (
                'timeout',
            ):
                value = getattr(self, key)
                if value is not None:
                    kwargs[key] = value

            raw_results = await self.apiwrapper.raw_results_async(
                urls=urls,
                extract_depth=self.extract_depth
                if self.extract_depth
                else extract_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                include_favicon=self.include_favicon,
                format=self.format,
                include_usage=self.include_usage,
                query=self.query if self.query else query,
                chunks_per_source=(
                    self.chunks_per_source
                    if self.chunks_per_source is not None
                    else chunks_per_source
                ),
                **kwargs,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None
