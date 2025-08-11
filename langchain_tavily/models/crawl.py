from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class TavilyCrawlInput(BaseModel):
    """Input for TavilyCrawl"""

    url: str = Field(description=("The root URL to begin the crawl."))

    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions for the crawler.

        The instructions parameter allows the crawler to intelligently navigate through a website using natural language.
        Take the users request to set the instructions parameter to guide the crawler in the direction of the users request.
        
        ex. "I want to find all the Javascript SDK documentation from Tavily" ---> instructions = "Javascript SDK documentation"
        """,
    )

    max_depth: Optional[int] = Field(
        default=3,
        ge=1,
        le=5,
        description="""Max depth of the crawl. Determines how many hops the crawler can make from the root URL.

        Increase this parameter to:
        1. Crawl large websites and get a comprehensive overview of its pages.
        2. Crawl a website that has a lot of links to other pages.
        """,
    )

    max_breadth: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
        description="""Max number of links to follow per per page.

        Increase this parameter when:
        1. You want many links from each page to be crawled.
        """,
    )

    limit: Optional[int] = Field(
        default=50,
        ge=1,
        le=500,
        description="""Maximum number of links the crawler will return.""",
    )

    select_paths: Optional[List[str]] = Field(
        default=None,
        description="""RegEx patterns to select only URLs with specific path patterns.

        Use only when the user explicitly asks for a specific path from a website.
        
        ex. "Only crawl the /api/v1 path" ---> ["/api/v1.*"] 
        ex. "Only crawl the /documentation path" ---> ["/documentation/.*"]
        """,
    )

    select_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs from specific domains or subdomains.
   
        Use only when the user explicitly asks for a specific domain or subdomain from a website.

        ex. "Crawl only the docs.tavily.com subdomain" ---> ["^docs\\.tavily\\.com$"]
        """,
    )

    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from the crawl with specific path patterns.

        Use only when the user explicitly asks to exclude a specific path from a website.

        ex. "Crawl example.com but exclude the /api/v1 path form the crawl" ---> ["/api/v1.*"] 
        ex. "Crawl example.com but exclude the /documentation path from the crawl" ---> ["/documentation/.*"]
        """,
    )

    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from specific domains or subdomains.

        Use only when the user explicitly asks to exclude a specific domain or subdomain from a website.

        ex. "Crawl tavily.com but exclude the docs.tavily.com subdomain from the crawl" ---> ["^docs\\.tavily\\.com$"]
        """,
    )

    allow_external: Optional[bool] = Field(
        default=True,
        description="""Allow the crawler to follow external links.

        Use when the user explicitly asks to allow or deny external links.
        """,
    )

    include_images: Optional[bool] = Field(
        default=False,
        description="""Whether to include images in the crawl results.
        """,
    )

    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Advanced extraction retrieves more data, including tables and embedded content
        with higher success but may increase latency.
        """,
    )

    include_favicon: Optional[bool] = Field(
        default=False,
        description="Whether to include the favicon URL for each result.",
    )

    chunks_per_source: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of content chunks to extract from each page.",
    )


class TavilyCrawlAgentInput(BaseModel):
    """Simplified input for agent-optimized TavilyCrawl invocation."""

    url: str = Field(description=("The root URL to begin the crawl."))

    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions to guide the crawl.
        
        Specify what the crawler should focus on.

        ex. "JavaScript SDK developer documentation" 
        """,
    )

    crawl_depth: Optional[Literal["fast", "basic", "deep"]] = Field(
        default="basic",
        description="""The scope of the crawl.
        
        `fast` - Crawls a minimal of pages at a low depth and optimizes for speed.
        `basic` - Balances speed and completeness. Recommended for most general use cases.
        `deep` - Crawls a large number of pages at a high depth. Optimizes for completeness and accuracy. Recommended for deep research. 

        Default is `basic`.
        """,
    )
