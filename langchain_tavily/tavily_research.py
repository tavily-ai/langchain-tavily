"""Tool for the Tavily Research API."""

from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_tavily._utilities import TavilyResearchAPIWrapper

class TavilyResearchInput(BaseModel):
    """
    Input for [TavilyResearch]
    Create comprehensive research reports on any topic using Tavily Research.
    """

    model_config = ConfigDict(extra="allow")

    input: str = Field(
        description="The research task or question to investigate."
    )
    model: Optional[Literal["mini", "pro", "auto"]] = Field(
        default=None,
        description="""
        The model used by the research agent. Can be "mini", "pro", or "auto".
        "mini" is optimized for targeted, efficient research and works best for narrow or well-scoped questions. 
        "pro" provides comprehensive, multi-angle research and is suited for complex topics that span multiple subtopics or domains.
        "auto" lets Tavily automatically determine the appropriate model based on the task complexity.
        Default is "auto".
        """,  # noqa: E501
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""A JSON Schema object that defines the structure of the research output. 
        
        When provided, the research response will be structured to match this schema, ensuring a predictable and validated output shape. 
        Must include a 'properties' field, and may optionally include 'required' field.
        
        Example:

        {
            "properties": {
                "company": {
                "type": "string",
                "description": "The name of the company"
                },
                "key_metrics": {
                "type": "array",
                "description": "List of key performance metrics",
                "items": { "type": "string" }
                },
                "financial_details": {
                "type": "object",
                "description": "Detailed financial breakdown",
                "properties": {
                    "operating_income": {
                    "type": "number",
                    "description": "Operating income for the period"
                    }
                }
            }
        },
        "required": ["company"]
        }
        """,  # noqa: E501
    )
    stream: Optional[bool] = Field(
        default=False,
        description="""Whether to stream the research results as they are generated. 
        When 'true', returns a Server-Sent Events (SSE) stream.
        
        Default is False.
        """,  # noqa: E501
    )
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = Field(
        default="numbered",
        description="""The format for citations in the research report.
        Default is "numbered".
        """,  # noqa: E501
    )



class TavilyGetResearchInput(BaseModel):
    """Input for getting research results by request_id."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(
        description="The unique identifier of the research task."
    )


class TavilyResearch(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Research API with dynamically settable parameters."""

    name: str = "tavily_research"
    description: str = (
        "Creates comprehensive research reports on any topic using Tavily Research. "
        "Useful for when you need to answer complex questions or gather in-depth information about a subject. "
        "The research task has two modes: streamed and non-streamed. If streamed, you will receive the results as they are generated. "
        "If not streamed, you can check the request status and obtain the result using the `TavilyGetResearch` tool. "
        "Input should be a research task or question to investigate."
    )

    args_schema: Type[BaseModel] = TavilyResearchInput
    handle_tool_error: bool = True

    # Default parameters
    research_model: Optional[Literal["mini", "pro", "auto"]] = Field(default="auto", alias="model")
    """The model used by the research agent. 
    
    Default is 'auto'
    """
    output_schema: Optional[Dict[str, Any]] = None
    """A JSON Schema object that defines the structure of the research output.
    
    Default is None
    """
    stream: Optional[bool] = None
    """Whether to stream the research results as they are generated. When 'true', returns a Server-Sent Events (SSE) stream
    
    Default is False
    """
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = None
    """The format for citations in the research report.
    
    Default is "numbered"
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Generator[bytes, None, None]]:
        """Execute a research task using the Tavily Research API.

        Returns:
            When stream=False or None:
                Dict[str, Any]: Research task response containing:
                    - request_id: The unique identifier of the research task
                    - created_at: Timestamp when the task was created
                    - status: Current status of the research task (e.g., "pending", "in_progress")
                    - input: The research task or question to investigate.
                    - model: The model used by the research agent
            When stream=True:
                Generator[bytes, None, None]: A generator that yields response chunks as bytes
        """
        try:
            is_streaming = stream if stream is not None else (self.stream if self.stream is not None else False)
            
            raw_results = self.api_wrapper.raw_results(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
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
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncGenerator[bytes, None]]:
        """Use the tool asynchronously.
        
        Returns:
            When stream=False or None:
                Dict[str, Any]: Research task response containing:
                    - request_id: The unique identifier of the research task
                    - created_at: Timestamp when the task was created
                    - status: Current status of the research task (e.g., "pending", "in_progress")
                    - input: The research task or question to investigate.
                    - model: The model used by the research agent.
            When stream=True:
                AsyncGenerator[bytes, None]: An async generator that yields response chunks as bytes
        """
        try:
            is_streaming = stream if stream is not None else (self.stream if self.stream is not None else False)
            
            raw_results = await self.api_wrapper.raw_results_async(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
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
                - request_id: The unique identifier of the research task
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

