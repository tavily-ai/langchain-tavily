"""Tool for the Tavily Research API."""

import base64
import binascii
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from langchain_tavily._errors import TRANSPORT_ERRORS, transport_error
from langchain_tavily._utilities import TavilyResearchAPIWrapper


class ResearchFile(TypedDict):
    """Text attachment supplied by the caller; no local files are read."""

    name: Annotated[str, Field(description="Filename ending in .txt, .md, or .json.")]
    data: Annotated[
        str,
        Field(description=(
            "Base64-encoded UTF-8 text, at most 80,000 words per file "
            "and across all files."
        )),
    ]
    type: NotRequired[Literal["base64"]]


def _validate_files(files: List[ResearchFile]) -> List[ResearchFile]:
    total = 0
    for file in files:
        if not file["name"].lower().endswith((".txt", ".md", ".json")):
            raise ValueError("Research files must use .txt, .md, or .json filenames")
        compact = "".join(file["data"].split())
        try:
            decoded = base64.b64decode(compact, validate=True)
            if base64.b64encode(decoded).decode() != compact:
                raise ValueError("non-canonical base64")
            words = len(decoded.decode("utf-8").split())
        except (binascii.Error, UnicodeError, ValueError) as exc:
            raise ValueError(
                "Research file data must be valid base64-encoded UTF-8 text"
            ) from exc
        if words > 80000:
            raise ValueError("Research files must not exceed 80,000 words per file")
        total += words
    if total > 80000:
        raise ValueError("Research files must not exceed 80,000 words combined")
    return files


ResearchFiles = Annotated[
    List[ResearchFile],
    Field(
        max_length=5,
        description=(
            "Up to five text files, at most 80,000 words per file and combined."
        ),
    ),
    AfterValidator(_validate_files),
]


class TavilyResearchInput(BaseModel):
    """
    Input for [TavilyResearch]
    Create comprehensive research reports on any topic using Tavily Research.
    """

    model_config = ConfigDict(extra="allow")

    include_domains: Optional[List[str]] = Field(
        default=None,
        description='Prefer these domains (up to 20); this is not a hard filter.',
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description='Exclude these domains (up to 20).',
    )
    output_length: Optional[Literal["short", "standard", "long"]] = Field(
        default=None,
        description='Length of the generated report.',
    )
    files: Optional[ResearchFiles] = Field(
        default=None,
        description=(
            "Up to five base64 text attachments; "
            "at most 80,000 words per file and combined."
        ),
    )

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
        default=None,
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

    include_usage: Optional[bool] = Field(
        default=None, description="Include credit usage in the Research task result."
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

    model_config = ConfigDict(populate_by_name=True)

    args_schema: Type[BaseModel] = TavilyResearchInput

    # Default parameters
    research_model: Optional[Literal["mini", "pro", "auto"]] = Field(default="auto", alias="model")
    """The model used by the research agent. 
    
    Default is 'auto'
    """
    research_output_schema: Optional[Dict[str, Any]] = Field(
        default=None, alias="output_schema"
    )
    """A JSON Schema object that defines the structure of the research output.
    
    Default is None
    """
    stream_results: Optional[bool] = Field(default=None, alias="stream")
    """Whether to stream the research results as they are generated. When 'true', returns a Server-Sent Events (SSE) stream
    
    Default is False
    """
    citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]] = None
    """The format for citations in the research report.
    
    Default is "numbered"
    """
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    output_length: Optional[Literal["short", "standard", "long"]] = None
    files: Optional[ResearchFiles] = None

    api_wrapper: TavilyResearchAPIWrapper = Field(
        default_factory=TavilyResearchAPIWrapper
    )  # type: ignore[arg-type]

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
            is_streaming = stream if stream is not None else (self.stream_results if self.stream_results is not None else False)
            
            for key in (
                'include_domains',
                'exclude_domains',
                'output_length',
                'files',
            ):
                value = getattr(self, key)
                if value is not None:
                    kwargs[key] = value

            raw_results = self.api_wrapper.raw_results(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema if output_schema is not None else self.research_output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
                **kwargs,
            )

            return raw_results
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None

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
            is_streaming = stream if stream is not None else (self.stream_results if self.stream_results is not None else False)
            
            for key in (
                'include_domains',
                'exclude_domains',
                'output_length',
                'files',
            ):
                value = getattr(self, key)
                if value is not None:
                    kwargs[key] = value

            raw_results = await self.api_wrapper.raw_results_async(
                input=input,
                research_model=research_model if research_model is not None else self.research_model,
                output_schema=output_schema if output_schema is not None else self.research_output_schema,
                stream=is_streaming,
                citation_format=self.citation_format
                if self.citation_format
                else citation_format,
                **kwargs,
            )
            return raw_results
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None


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

    include_usage: Optional[bool] = None
    """Optional application setting; overrides the invocation value when configured."""

    api_wrapper: TavilyResearchAPIWrapper = Field(
        default_factory=TavilyResearchAPIWrapper
    )  # type: ignore[arg-type]

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
            kwargs["api_wrapper"] = TavilyResearchAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        request_id: str,
        include_usage: Optional[bool] = None,
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
            raw_results = self.api_wrapper.get_research(
                request_id=request_id,
                include_usage=(
                    self.include_usage
                    if self.include_usage is not None
                    else include_usage
                ),
            )
            return raw_results
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None

    async def _arun(
        self,
        request_id: str,
        include_usage: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.get_research_async(
                request_id=request_id,
                include_usage=(
                    self.include_usage
                    if self.include_usage is not None
                    else include_usage
                ),
            )
            return raw_results
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from None

