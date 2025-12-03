"""Tool for the Tavily Research API."""

from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_tavily._utilities import TavilyResearchAPIWrapper


class MCPObject(BaseModel):
    """MCP object structure for research tasks."""

    name: str = Field(description="MCP object name")
    url: str = Field(description="MCP object URL")
    transport: Optional[Literal["streamable_http", "sse"]] = Field(
        default="streamable_http",
        description="Transport type for the MCP connection",
    )
    tools_to_include: Optional[List[str]] = Field(
        default=None,
        description="List of tool names to include from the MCP",
    )
    headers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom headers for the MCP connection",
    )


class TavilyResearchInput(BaseModel):
    """
    Input for [TavilyResearch]
    Create comprehensive research reports on any topic using Tavily Research.
    """

    model_config = ConfigDict(extra="allow")

    input: str = Field(
        description="The research task description. This is the main query that describes what you want to research."
    )
    model: Optional[Literal["mini", "pro", "auto"]] = Field(
        default=None,
        description="""Controls the depth and thoroughness of the research.
        
        Use "mini" for quick, surface-level research on common topics.
        
        Use "pro" for comprehensive, in-depth research requiring extensive analysis 
        and multiple sources. Best for complex topics or when detailed information is needed.
        
        Use "auto" to let Tavily automatically determine the appropriate depth 
        based on the task description.
        
        Default is None (uses API default).
        """,  # noqa: E501
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""JSON Schema dict for structured output format.
        
        Use this parameter when you need the research results in a specific structured format.
        The schema should follow JSON Schema specification. The top level of the schema must include "title" and "description".
        
        Example:

         "{
            "title": "ResearchReport",
            "description": "A structured research report",
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "sections": {"type": "array", "items": {...}}
            }
         }
                    
        Default is None (returns unstructured text content).
        """,  # noqa: E501
    )
    stream: Optional[bool] = Field(
        default=False,
        description="""Whether to stream the research task results.
        
        Set to True when you want to receive results incrementally as they become available.
        
        Default is False.
        """,  # noqa: E501
    )
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = Field(
        default="numbered",
        description="""Citation format for sources in the research report.
        
        Options:
        - "numbered": Numbered citations (e.g., [1], [2])
        - "mla": Modern Language Association format
        - "apa": American Psychological Association format
        - "chicago": Chicago Manual of Style format
        
        Default is "numbered".
        """,  # noqa: E501
    )
    mcps: Optional[List[MCPObject]] = Field(
        default=None,
        description="""List of MCP (Model Context Protocol) objects to use for the research task.
        
        MCP objects allow the research task to use external tools and data sources.
        Each MCP object should have at least 'name' and 'url' fields.
        
        Default is None (no MCP objects used).
        """,  # noqa: E501
    )


class TavilyGetResearchInput(BaseModel):
    """Input for getting research results by request_id."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(
        description="The research request ID returned from creating a research task."
    )


class TavilyResearch(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Research API with dynamically settable parameters."""

    name: str = "tavily_research"
    description: str = (
        "Creates comprehensive research reports on any topic with automatic source gathering, "
        "analysis, and structured output. This tool initiates a research task that gathers information "
        "from multiple sources, analyzes the content, and produces a detailed research report. "
        "The research task is asynchronous - you'll receive a request_id that you can use to "
        "retrieve the results once the research is complete. Input should be a research task description."
    )

    args_schema: Type[BaseModel] = TavilyResearchInput
    handle_tool_error: bool = True

    # Default parameters
    research_model: Optional[Literal["mini", "pro", "auto"]] = Field(default="auto", alias="model")
    """The depth of the research. It can be 'mini', 'pro', or 'auto'
    
    Default is 'auto' (uses API default)
    """
    output_schema: Optional[Dict[str, Any]] = None
    """JSON Schema dict for structured output format.
    
    Default is None
    """
    stream: Optional[bool] = None
    """Whether to stream the research task.
    
    Default is False
    """
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = None
    """Citation format for sources in the research report.
    
    Default is "numbered"
    """
    mcps: Optional[List[MCPObject]] = None
    """List of MCP objects to use for the research task.
    
    Default is None
    """
    api_wrapper: TavilyResearchAPIWrapper = Field(
        default_factory=TavilyResearchAPIWrapper
    )  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = TavilyResearchAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        input: str,
        research_model: Optional[Literal["mini", "pro", "auto"]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = None,
        mcps: Optional[List[MCPObject]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Generator[bytes, None, None]]:
        """Execute a research task using the Tavily Research API.

        Returns:
            When stream=False or None:
                Dict[str, Any]: Research task response containing:
                    - request_id: ID to use for retrieving research results
                    - created_at: Timestamp when the task was created
                    - status: Current status of the research task (e.g., "pending", "in_progress")
                    - input: The research task input
                    - model: The research model used
            When stream=True:
                Generator[bytes, None, None]: A generator that yields response chunks as bytes
        """
        try:
            # Convert MCPObject models to dicts for API
            mcps_dict = None
            if mcps:
                mcps_dict = [
                    {
                        "name": mcp.name,
                        "url": mcp.url,
                        "transport": mcp.transport,
                        "tools_to_include": mcp.tools_to_include,
                        "headers": mcp.headers,
                    }
                    for mcp in mcps
                ]
            
            is_streaming = stream if stream is not None else (self.stream if self.stream is not None else False)
            
            raw_results = self.api_wrapper.raw_results(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
                mcps=mcps_dict,
                **kwargs,
            )

            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        input: str,
        research_model: Optional[Literal["mini", "pro", "auto"]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = None,
        mcps: Optional[List[MCPObject]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncGenerator[bytes, None]]:
        """Use the tool asynchronously.
        
        Returns:
            When stream=False or None:
                Dict[str, Any]: Research task response containing:
                    - request_id: ID to use for retrieving research results
                    - created_at: Timestamp when the task was created
                    - status: Current status of the research task (e.g., "pending", "in_progress")
                    - input: The research task input
                    - model: The research model used
            When stream=True:
                AsyncGenerator[bytes, None]: An async generator that yields response chunks as bytes
        """
        try:
            # Convert MCPObject models to dicts for API
            mcps_dict = None
            if mcps:
                mcps_dict = [
                    {
                        "name": mcp.name,
                        "url": mcp.url,
                        "transport": mcp.transport,
                        "tools_to_include": mcp.tools_to_include,
                        "headers": mcp.headers,
                    }
                    for mcp in mcps
                ]
            
            is_streaming = stream if stream is not None else (self.stream if self.stream is not None else False)
            
            raw_results = await self.api_wrapper.raw_results_async(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
                mcps=mcps_dict,
                **kwargs,
            )
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            is_streaming = stream if stream is not None else (self.stream if self.stream is not None else False)
            if is_streaming:
                raise
            return {"error": str(e)}


class TavilyGetResearch(BaseTool):  # type: ignore[override, override]
    """Tool that retrieves research results by request_id."""

    name: str = "tavily_get_research"
    description: str = (
        "Retrieves the results of a research task by its request_id. "
        "Use this tool after creating a research task to get the completed research report, "
        "including the content, sources, and status. Input should be a request_id from a "
        "previously created research task."
    )

    args_schema: Type[BaseModel] = TavilyGetResearchInput
    handle_tool_error: bool = True

    api_wrapper: TavilyResearchAPIWrapper = Field(
        default_factory=TavilyResearchAPIWrapper
    )  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = TavilyResearchAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        request_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get research results by request_id.

        Returns:
            Dict[str, Any]: Research results containing:
                - request_id: The research request ID
                - created_at: Timestamp when the task was created
                - completed_at: Timestamp when the task was completed (if completed)
                - status: Current status (e.g., "pending", "in_progress", "completed", "failed")
                - content: The research report content (if completed)
                - sources: List of sources used in the research (if completed)
        """
        try:
            raw_results = self.api_wrapper.get_research(request_id=request_id)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        request_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.get_research_async(request_id=request_id)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": str(e)}

