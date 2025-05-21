"""Tool for the Tavily Map API."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilyMapAPIWrapper


class TavilyMapInput(BaseModel):
    """Input for [TavilyMap]"""

    url: str = Field(description=("The root URL to begin the mapping."))
    max_depth: Optional[int] = Field(
        default=1,
        description="""Max depth of the mapping. Defines how far from the base URL the crawler can explore.

        Increase this parameter when:
        1. To crawl large websites and get a comprehensive overview of its structure.
        2. To crawl a website that has a lot of links to other pages.

        Set this parameter to 1 when:
        1. To stay local to the base_url
        2. To crawl a single page


        max_depth must be greater than 0
        """,  # noqa: E501
    )
    max_breadth: Optional[int] = Field(
        default=20,
        description="""Max number of links to follow per level of the tree (i.e., per page).

        tavily-map uses a BFS Depth: refering to the number of link hops from the root URL. 
        A page directly linked from the root is at BFS depth 1, regardless of its URL structure.

        Increase this parameter when:
        1. You want many links from each page to be crawled.

        max_breadth must be greater than 0
        """,  # noqa: E501
    )
    limit: Optional[int] = Field(
        default=50,
        description="""Total number of links the crawler will process before stopping.
        
        limit must be greater than 0
        """,  # noqa: E501
    )
    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions for the crawler.

        The instructions parameter allows the crawler to intelligently navigate through a website using natural language.
        Take the users request to set the instructions parameter to guide the crawler in the direction of the users request.
        
        ex. "I want to find all the Javascript SDK documentation from Tavily" ---> instructions = "Javascript SDK documentation"
        """,  # noqa: E501
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs with specific path patterns.

        Use when the user explcitily asks from a specific path from a website.
        
        ex. "Only crawl the /api/v1 path" ---> ["/api/v1.*"] 
        ex. "Only crawl the /documentation path" ---> ["/documentation/.*"]
        """,  # noqa: E501
    )
    select_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs from specific domains or subdomains.
   
        Use when the user explcitily asks from a specific subdomains from a website.

        ex. "Crawl only the docs.tavily.com subdomain" ---> ["^docs\.tavily\.com$"]
        """,  # noqa: E501
    )
    allow_external: Optional[bool] = Field(
        default=False,
        description="""Allow the crawler to follow external links.

        Use when the user explcitily asks to allow or deny external links.
        """,  # noqa: E501
    )
    categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = Field(
        default=None,
        description="""Direct the crawler to crawl specific categories of a website.

        Set this field to the category that best matches the user's request. Use the following guide to choose the appropriate category:

            Careers: Crawl pages related to job listings, open positions, and company career information.
            Blog: Crawl blog posts, news articles, and editorial content.
            Documentation: Crawl technical documentation, user guides, API references, and manuals.
            About: Crawl 'About Us' pages, company background, mission statements, and team information.
            Pricing: Crawl pages that detail product or service pricing, plans, and cost comparisons.
            Community: Crawl forums, discussion boards, user groups, and community-driven content.
            Developers: Crawl developer portals, SDKs, API documentation, and resources for software developers.
            Contact: Crawl contact information pages, support forms, and customer service details.
            Media: Crawl press releases, media kits, newsrooms, and multimedia content.


        ex. "Crawl apple.com for carrer opportunities" ---> categories="Careers"
        ex. "Crawl tavily.com for API documentation" ---> categories="Documentation"
    """
    )
    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Advanced extraction retrieves more data, including tables and embedded content
        with higher success but may increase latency.
        """
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed crawl parameters.
    """
    suggestions = []

    instructions = params.get("instructions")
    select_paths = params.get("select_paths")
    select_domains = params.get("select_domains")
    categories = params.get("categories")

    if instructions:
        suggestions.append("Try a more consice instructions")
    if select_paths:
        suggestions.append("Remove select_paths argument")
    if select_domains:
        suggestions.append("Remove select_domains argument")
    if categories:
        suggestions.append("Remove categories argument")

    return suggestions


class TavilyMap(BaseTool):  # type: ignore[override]
    """Tool that sends requests to the Tavily Map API with dynamically settable parameters."""

    name: str = "tavily_map"
    description: str = (
        """"A powerful web mapping tool that creates a structured map of website URLs, allowing 
        you to discover and analyze site structure, content organization, and navigation paths. 
        Perfect for site audits, content discovery, and understanding website architecture.
        """
    )

    args_schema: Type[BaseModel] = TavilyMapInput
    handle_tool_error: bool = True

    max_depth: Optional[int] = 1
    """Max depth of the crawl. Defines how far from the base URL the crawler can explore.

    max_depth must be greater than 0
    """
    max_breadth: Optional[int] = 20
    """The maximum number of links to follow per level of the tree (i.e., per page).

    max_breadth must be greater than 0
    """
    limit: Optional[int] = 50
    """Total number of links the crawler will process before stopping.

    limit must be greater than 0
    """
    instructions: Optional[str] = None
    """Natural language instructions for the crawler.

    ex. "Python SDK"
    """
    select_paths: Optional[List[str]] = None
    """Regex patterns to select only URLs with specific path patterns.

    ex. ["/api/v1.*"]
    """
    select_domains: Optional[List[str]] = None
    """Regex patterns to select only URLs from specific domains or subdomains.

    ex. ["^docs\.example\.com$"]
    """
    allow_external: Optional[bool] = False
    """Whether to allow following links that go to external domains.
    """
    categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = None
    """Filter URLs using predefined categories like 'Documentation', 'Blog', 'API', etc.
    """
    extract_depth: Optional[Literal["basic", "advanced"]] = "basic"
    """Advanced extraction retrieves more data, including tables and embedded content, 
    with higher success but may increase latency.
    """

    api_wrapper: TavilyMapAPIWrapper = Field(default_factory=TavilyMapAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key if provided
        if "tavily_api_key" in kwargs:
            kwargs["api_wrapper"] = TavilyMapAPIWrapper(
                tavily_api_key=kwargs["tavily_api_key"]
            )

        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a mapping using the Tavily Map API.

        Returns:
            - base_url (str): The base URL that was mapped
                Example: "https://tavily.com/"
            
            - results (List[str]): A list of mapped URLs
                - url (str): The URL that was mapped
                    Example: "https://tavily.com/#features"
            
            - response_time (float): Time in seconds it took to complete the request

        """
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                url=url,
                max_depth=max_depth if max_depth else self.max_depth,   
                max_breadth=max_breadth if max_breadth else self.max_breadth,
                limit=limit if limit else self.limit,
                instructions=instructions if instructions else self.instructions,
                select_paths=select_paths if select_paths else self.select_paths,
                select_domains=select_domains if select_domains else self.select_domains,
                allow_external=allow_external if allow_external else self.allow_external,
                categories=categories if categories else self.categories,
                extract_depth=extract_depth if extract_depth else self.extract_depth,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."  # noqa: E501
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
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                url=url,
                max_depth=max_depth if max_depth else self.max_depth,   
                max_breadth=max_breadth if max_breadth else self.max_breadth,
                limit=limit if limit else self.limit,
                instructions=instructions if instructions else self.instructions,
                select_paths=select_paths if select_paths else self.select_paths,
                select_domains=select_domains if select_domains else self.select_domains,
                allow_external=allow_external if allow_external else self.allow_external,
                categories=categories if categories else self.categories,
                extract_depth=extract_depth if extract_depth else self.extract_depth,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}
