"""Tool for the Tavily Crawl API."""

from typing import Any, Dict, List, Literal, Optional, Type, overload

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilyCrawlAPIWrapper
from langchain_tavily.models.crawl import TavilyCrawlInput, TavilyCrawlAgentInput


def _generate_suggestions(params: Dict[str, Any]) -> List[str]:
    """Generate helpful suggestions based on the failed crawl parameters."""
    suggestions = []

    instructions = params.get("instructions")
    select_paths = params.get("select_paths")
    select_domains = params.get("select_domains")
    exclude_paths = params.get("exclude_paths")
    exclude_domains = params.get("exclude_domains")

    if instructions:
        suggestions.append("Try more consice instructions")
    if select_paths:
        suggestions.append("Remove select_paths argument")
    if select_domains:
        suggestions.append("Remove select_domains argument")
    if exclude_paths:
        suggestions.append("Remove exclude_paths argument")
    if exclude_domains:
        suggestions.append("Remove exclude_domains argument")

    return suggestions


class TavilyCrawl(BaseTool): 
    """Wrapper around the Tavily Crawl API.""" 

    name: str = "tavily_crawl"

    description: str = """An intelligent web crawler that initiates a structured web crawl starting from a specified base URL. 
    You can provide natural language instructions to guide the crawler.
    """ 

    args_schema: Type[BaseModel] = TavilyCrawlInput

    handle_tool_error: bool = True

    agent_mode: Optional[bool] = True
    """Enable an agent-optimized invocation schema.

    When `True`, the tool exposes a simplified invocation schema, making it easier for agents to call.
    You can still manually specify parameters at instantiation time.

    Default is `True`.
    """

    max_depth: Optional[int] = None
    """The maximum depth of the crawl. Determines how many hops the crawler can make from the root URL.
    
    Must be greater than `0`.

    Default is `3`.
    """ 
    max_breadth: Optional[int] = None
    """The maximum number of links to follow per page.

    Must be greater than `0`.

    Default is `20`.
    """
    limit: Optional[int] = None
    """The maximum number of links the crawler will return.

    Must be greater than `0`.

    Default is `50`.
    """

    instructions: Optional[str] = None
    """Natural language instructions for the crawler.

    ex. `"Python SDK"`
    """

    select_paths: Optional[List[str]] = None
    """RegEx patterns to select only URLs with specific path patterns.

    ex. `["/api/v1.*"]`
    """

    select_domains: Optional[List[str]] = None
    """RegEx patterns to select only URLs from specific domains or subdomains.

    ex. `["^docs\\.example\\.com$"]`
    """

    exclude_paths: Optional[List[str]] = None
    """
    RegEx patterns to exclude URLs with specific path patterns.

    ex.  `[/private/.*, /admin/.*]`
    """

    exclude_domains: Optional[List[str]] = None
    """
    RegEx patterns to exclude specific domains or subdomains from crawling.

    ex. `[^private\\.example\\.com$]`
    """

    allow_external: Optional[bool] = None
    """Whether to allow following links that go to external domains.

    Default is `False`.
    """

    include_images: Optional[bool] = None
    """Whether to include images in the crawl results.

    Default is `False`.
    """

    extract_depth: Optional[Literal["basic", "advanced"]] = None
    """Advanced extraction retrieves more data, including tables and embedded content, 
    with higher success but may increase latency.

    Default is `"basic"`.
    """

    format: Optional[str] = None
    """
    The format of the extracted web page content. markdown returns content in markdown 
    format. text returns plain text and may increase latency.

    Default is `"markdown"`.
    """

    include_favicon: Optional[bool] = None
    """Whether to include the favicon URL for each result.
    
    Default is `False`.
    """

    chunks_per_source: Optional[int] = None
    """Number of content chunks to extract from each page.
    
    Must be between 1 and 10.
    
    Default is `None` for regular mode and `3` for agent mode.
    """

    api_wrapper: TavilyCrawlAPIWrapper = Field(default_factory=TavilyCrawlAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, agent_mode: bool = True, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = TavilyCrawlAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

        # Explicitly set the agent_mode instance attribute
        self.agent_mode = agent_mode

        # Set chunks_per_source default based on agent_mode (if not explicitly provided)
        if self.chunks_per_source is None:
            if self.agent_mode:
                self.chunks_per_source = 3  # Default for agent mode
            # else: leave as None for regular mode

        # If agent mode is enabled, switch to a simplified args schema and
        # update description to reflect agent-focused usage.
        if self.agent_mode:
            # Switch the invocation schema for this instance
            self.args_schema = TavilyCrawlAgentInput  # type: ignore[assignment]

    def _map_crawl_depth(self, crawl_depth: Literal["fast", "basic", "deep"]) -> Dict[str, int]:
        """Map crawl_depth values to specific max_depth, max_breadth, and limit parameters."""
        mapping = {
            "fast": {"max_depth": 1, "max_breadth": 20, "limit": 20},
            "basic": {"max_depth": 3, "max_breadth": 50, "limit": 100},
            "deep": {"max_depth": 5, "max_breadth": 50, "limit": 200},
        }
        return mapping[crawl_depth]

    def _run_agent(
        self,
        url: str,
        instructions: Optional[str] = None,
        crawl_depth: Optional[Literal["fast", "basic", "deep"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a crawl using the agent-optimized schema."""
        # Map crawl_depth to specific parameters if provided
        if crawl_depth:
            depth_params = self._map_crawl_depth(crawl_depth)
            max_depth = depth_params["max_depth"]
            max_breadth = depth_params["max_breadth"]
            limit = depth_params["limit"]
        else:
            # Use default values from the mapping for "basic" mode
            depth_params = self._map_crawl_depth("basic")
            max_depth = depth_params["max_depth"]
            max_breadth = depth_params["max_breadth"]
            limit = depth_params["limit"]

        # Use instance attributes if set, otherwise use mapped values
        final_max_depth = self.max_depth if self.max_depth is not None else max_depth
        final_max_breadth = self.max_breadth if self.max_breadth is not None else max_breadth
        final_limit = self.limit if self.limit is not None else limit
        final_instructions = self.instructions if self.instructions is not None else instructions

        try:
            raw_results = self.api_wrapper.raw_results(
                url=url,
                max_depth=final_max_depth,
                max_breadth=final_max_breadth,
                limit=final_limit,
                instructions=final_instructions,
                select_paths=self.select_paths,
                select_domains=self.select_domains,
                exclude_paths=self.exclude_paths,
                exclude_domains=self.exclude_domains,
                allow_external=self.allow_external,
                include_images=self.include_images,
                extract_depth=self.extract_depth,
                include_favicon=self.include_favicon,
                format=self.format,
                chunks_per_source=self.chunks_per_source,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": final_instructions,
                    "select_paths": self.select_paths,
                    "select_domains": self.select_domains,
                    "exclude_paths": self.exclude_paths,
                    "exclude_domains": self.exclude_domains,
                    "format": self.format,
                }
                suggestions = _generate_suggestions(search_params)

                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches." 
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": e}

    @overload
    def _run(
        self,
        url: str,
        instructions: Optional[str] = None,
        crawl_depth: Optional[Literal["fast", "basic", "deep"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]: ...

    @overload
    def _run(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        include_images: Optional[bool] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: Optional[bool] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]: ...

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """Route to appropriate implementation based on agent_mode."""
        if getattr(self, "agent_mode", False):
            # Extract only agent-compatible parameters
            agent_kwargs = {}
            if "url" in kwargs:
                agent_kwargs["url"] = kwargs["url"]
            if "instructions" in kwargs:
                agent_kwargs["instructions"] = kwargs["instructions"]
            if "crawl_depth" in kwargs:
                agent_kwargs["crawl_depth"] = kwargs["crawl_depth"]
            if "run_manager" in kwargs:
                agent_kwargs["run_manager"] = kwargs["run_manager"]
            return self._run_agent(**agent_kwargs)
        else:
            return self._run_full(**kwargs)

    def _run_full(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        include_images: Optional[bool] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: Optional[bool] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a crawl using the Tavily Crawl API.

        Returns:
            - results (List[Dict]): A list of extracted content from the crawled URLs
                - url (str): The URL that was crawled
                    Example: "https://tavily.com/#features"
                - raw_content (str): The full content extracted from the page
                - images (List[str]): A list of image URLs extracted from the page

            - response_time (float): Time in seconds it took to complete the request

        """
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=(
                    self.select_domains if self.select_domains else select_domains
                ),
                exclude_paths=(
                    self.exclude_paths if self.exclude_paths else exclude_paths
                ),
                exclude_domains=(
                    self.exclude_domains if self.exclude_domains else exclude_domains
                ),
                allow_external=(
                    self.allow_external if self.allow_external else allow_external
                ),
                include_images=(
                    self.include_images if self.include_images else include_images
                ),
                extract_depth=(
                    self.extract_depth if self.extract_depth else extract_depth
                ),
                include_favicon=(
                    self.include_favicon if self.include_favicon else include_favicon
                ),
                format=self.format,
                chunks_per_source=(
                    self.chunks_per_source if self.chunks_per_source else chunks_per_source
                ),
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "format": self.format,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches." 
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}

    async def _arun_agent(
        self,
        url: str,
        instructions: Optional[str] = None,
        crawl_depth: Optional[Literal["fast", "basic", "deep"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a crawl asynchronously using the agent-optimized schema."""
        # Map crawl_depth to specific parameters if provided
        if crawl_depth:
            depth_params = self._map_crawl_depth(crawl_depth)
            max_depth = depth_params["max_depth"]
            max_breadth = depth_params["max_breadth"]
            limit = depth_params["limit"]
        else:
            # Use default values from the mapping for "basic" mode
            depth_params = self._map_crawl_depth("basic")
            max_depth = depth_params["max_depth"]
            max_breadth = depth_params["max_breadth"]
            limit = depth_params["limit"]

        # Use instance attributes if set, otherwise use mapped values
        final_max_depth = self.max_depth if self.max_depth is not None else max_depth
        final_max_breadth = self.max_breadth if self.max_breadth is not None else max_breadth
        final_limit = self.limit if self.limit is not None else limit
        final_instructions = self.instructions if self.instructions is not None else instructions

        try:
            raw_results = await self.api_wrapper.raw_results_async(
                url=url,
                max_depth=final_max_depth,
                max_breadth=final_max_breadth,
                limit=final_limit,
                instructions=final_instructions,
                select_paths=self.select_paths,
                select_domains=self.select_domains,
                exclude_paths=self.exclude_paths,
                exclude_domains=self.exclude_domains,
                allow_external=self.allow_external,
                include_images=self.include_images,
                extract_depth=self.extract_depth,
                include_favicon=self.include_favicon,
                format=self.format,
                chunks_per_source=self.chunks_per_source,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": final_instructions,
                    "select_paths": self.select_paths,
                    "select_domains": self.select_domains,
                    "exclude_paths": self.exclude_paths,
                    "exclude_domains": self.exclude_domains,
                }
                suggestions = _generate_suggestions(search_params)

                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches." 
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": e}

    async def _arun_full(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        include_images: Optional[bool] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: Optional[bool] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=(
                    self.select_domains if self.select_domains else select_domains
                ),
                exclude_paths=(
                    self.exclude_paths if self.exclude_paths else exclude_paths
                ),
                exclude_domains=(
                    self.exclude_domains if self.exclude_domains else exclude_domains
                ),
                allow_external=(
                    self.allow_external if self.allow_external else allow_external
                ),
                include_images=(
                    self.include_images if self.include_images else include_images
                ),
                extract_depth=(
                    self.extract_depth if self.extract_depth else extract_depth
                ),
                include_favicon=(
                    self.include_favicon if self.include_favicon else include_favicon
                ),
                format=self.format,
                chunks_per_source=(
                    self.chunks_per_source if self.chunks_per_source else chunks_per_source
                ),
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches." 
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}

    @overload
    async def _arun(
        self,
        url: str,
        instructions: Optional[str] = None,
        crawl_depth: Optional[Literal["fast", "basic", "deep"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]: ...

    @overload
    async def _arun(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        include_images: Optional[bool] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_favicon: Optional[bool] = None,
        chunks_per_source: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]: ...

    async def _arun(self, **kwargs: Any) -> Dict[str, Any]:
        """Route to appropriate async implementation based on agent_mode."""
        if getattr(self, "agent_mode", False):
            # Extract only agent-compatible parameters
            agent_kwargs = {}
            if "url" in kwargs:
                agent_kwargs["url"] = kwargs["url"]
            if "instructions" in kwargs:
                agent_kwargs["instructions"] = kwargs["instructions"]
            if "crawl_depth" in kwargs:
                agent_kwargs["crawl_depth"] = kwargs["crawl_depth"]
            if "run_manager" in kwargs:
                agent_kwargs["run_manager"] = kwargs["run_manager"]
            return await self._arun_agent(**agent_kwargs)
        else:
            return await self._arun_full(**kwargs)
