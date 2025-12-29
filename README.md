# 🦜️🔗 LangChain Tavily

[![PyPI version](https://badge.fury.io/py/langchain-tavily.svg)](https://badge.fury.io/py/langchain-tavily)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/langchain-tavily)](https://pepy.tech/project/langchain-tavily)

This package contains the LangChain integration with [Tavily](https://tavily.com/)

```bash
pip install -U langchain-tavily
```

Don't miss out on these exciting new features! Check out the [full documentation](https://docs.tavily.com/) to learn more.

---

## Installation

```bash
pip install -U langchain-tavily
```

### Credentials

We also need to set our Tavily API key. You can get an API key by visiting [this site](https://app.tavily.com/sign-in) and creating an account.

```bash
import getpass
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
```

## Tavily Search

Here we show how to instantiate an instance of the Tavily search tool. The tool accepts various parameters to customize the search. After instantiation we invoke the tool with a simple query. This tool allows you to complete search queries using Tavily's Search API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `max_results` (optional, int): Maximum number of search results to return. Default is 5.
- `topic` (optional, str): Category of the search. Can be "general", "news", or "finance". Default is "general".
- `include_answer` (optional, bool | str): Include an answer to original query in results. Default is False. String options include "basic" (quick answer) or "advanced" (detailed answer). If True, defaults to "basic".
- `include_raw_content` (optional, bool | str): Include the cleaned and parsed HTML content of each search result. "markdown" returns search result content in markdown format. "text" returns the plain text from the results and may increase latency. If True, defaults to "markdown"
- `include_images` (optional, bool): Include a list of query related images in the response. Default is False.
- `include_image_descriptions` (optional, bool): Include descriptive text for each image. Default is False.
- `include_favicon` (optional, bool): Whether to include the favicon URL for each result. Default is False.
- `include_usage` (optional, bool): Whether to include credit usage information in the response.
- `search_depth` (optional, str): Depth of the search. Options: "basic", "advanced", "fast", or "ultra-fast". Default is "basic".
- `time_range` (optional, str): The time range back from the current date to filter results - "day", "week", "month", or "year". Default is None.
- `include_domains` (optional, List[str]): List of domains to specifically include. Default is None.
- `exclude_domains` (optional, List[str]): List of domains to specifically exclude. Default is None.
- `country` (optional, str): Boost search results from a specific country. This will prioritize content from the selected country in the search results. Available only if topic is general.

For a comprehensive overview of the available parameters, refer to the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search)

```python
from langchain_tavily import TavilySearch

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # include_favicon=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None,
    # country=None
)
```

### Invoke directly with args

The Tavily search tool accepts the following arguments during invocation:

- `query` (required): A natural language search query
- The following arguments can also be set during invocation : `include_images`, `search_depth`, `time_range`, `include_domains`, `exclude_domains`, `topic`, `start_date` and `end_date`.
- For reliability and performance reasons, certain parameters that affect response size cannot be modified during invocation: `include_answer` and `include_raw_content`. These limitations prevent unexpected context window issues and ensure consistent results.

NOTE: If you set an argument during instantiation this value will persist and overwrite the value passed during invocation.

```python
# Basic query
tool.invoke({"query": "What happened at the last wimbledon"})
```

output:

```bash
{
 'query': 'What happened at the last wimbledon',
 'follow_up_questions': None,
 'answer': None,
 'images': [],
 'results':
 [{'url': 'https://en.wikipedia.org/wiki/Wimbledon_Championships',
   'title': 'Wimbledon Championships - Wikipedia',
   'content': 'Due to the COVID-19 pandemic, Wimbledon 2020 was cancelled ...',
   'score': 0.62365627198,
   'raw_content': None},
    ......................................................................
    {'url': 'https://www.cbsnews.com/news/wimbledon-men-final-carlos-alcaraz-novak-djokovic/',
    'title': "Carlos Alcaraz beats Novak Djokovic at Wimbledon men's final to ...",
    'content': 'In attendance on Sunday was Catherine, the Princess of Wales ...',
    'score': 0.5154731446,
    'raw_content': None}],
  'response_time': 2.3
}
```

### Agent Tool Calling

We can use our tools directly with an agent executor by binding the tool to the agent. This gives the agent the ability to dynamically set the available arguments to the Tavily search tool.

In the below example when we ask the agent to find "What is the most popular sport in the world? include only wikipedia sources" the agent will dynamically set the argments and invoke Tavily search tool : Invoking `tavily_search` with `{'query': 'most popular sport in the world', 'include_domains': ['wikipedia.org'], 'search_depth': 'basic'}`

```python
# !pip install -qU langchain langchain-openai langchain-tavily
from typing import Any, Dict, Optional
import datetime

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.schema import HumanMessage, SystemMessage

# Initialize LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

# Set up Prompt with 'agent_scratchpad'
today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful reaserch assistant, you will be given a query and you will need to
    search the web for the most relevant information. The date today is {today}."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Required for tool calls
])

# Create an agent that can use tools
agent = create_openai_tools_agent(
    llm=llm,
    tools=[tavily_search_tool],
    prompt=prompt
)

# Create an Agent Executor to handle tool execution
agent_executor = AgentExecutor(agent=agent, tools=[tavily_search_tool], verbose=True)

user_input =  "What is the most popular sport in the world? include only wikipedia sources"

# Construct input properly as a dictionary
response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
```

## Tavily Extract

Here we show how to instantiate an instance of the Tavily extract tool. After instantiation we invoke the tool with a list of URLs. This tool allows you to extract content from URLs using Tavily's Extract API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `extract_depth` (optional, str): The depth of the extraction, either "basic" or "advanced". Default is "basic ".
- `include_images` (optional, bool): Whether to include images in the extraction. Default is False.
- `include_favicon` (optional, bool): Whether to include the favicon URL for each result. Default is False.
- `include_usage` (optional, bool): Whether to include credit usage information in the response. NOTE:The value may be 0 if the total successful URL extractions has not yet reached 5 calls. See our [Credits & Pricing documentation](https://docs.tavily.com/documentation/api-credits) for details.
- `format` (optional, str): The format of the extracted web page content. "markdown" returns content in markdown format. "text" returns plain text and may increase latency.

For a comprehensive overview of the available parameters, refer to the [Tavily Extract API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/extract)

```python
from langchain_tavily import TavilyExtract

tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False,
    include_favicon=False,
    format="markdown"
)
```

### Invoke directly with args

The Tavily extract tool accepts the following arguments during invocation:

- `urls` (required): A list of URLs to extract content from.
- The parameters `extract_depth` and `include_images` can also be set during invocation

NOTE: If you set an argument during instantiation this value will persist and overwrite the value passed during invocation.

```python
# Extract content from a URL
result = tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Lionel_Messi"]
})
```

output:

```bash
{
    'results': [{
        'url': 'https://en.wikipedia.org/wiki/Lionel_Messi',
        'raw_content': 'Lionel Messi\nLionel Andrés "Leo" Messi...',
        'images': []
    }],
    'failed_results': [],
    'response_time': 0.79
}
```

## Tavily Crawl

Here we show how to instantiate an instance of the Tavily crawl tool. After instantiation we invoke the tool with a URL. This tool allows you to crawl websites using Tavily's Crawl API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `max_depth` (optional, int): Max depth of the crawl from base URL. Default is 1.
- `max_breadth` (optional, int): Max number of links to follow per page. Default is 20.
- `limit` (optional, int): Total number of links to process before stopping. Default is 50.
- `instructions` (optional, str): Natural language instructions to guide the crawler. Default is None.
- `select_paths` (optional, List[str]): Regex patterns to select specific URL paths. Default is None.
- `select_domains` (optional, List[str]): Regex patterns to select specific domains. Default is None.
- `exclude_paths` (optional, List[str]): Regex patterns to exclude URLs with specific path patterns
- `exclude_domains` (optional, List[str]): Regex patterns to exclude specific domains or subdomains from crawling
- `allow_external` (optional, bool): Allow following external domain links. Default is False.
- `include_images` (optional, bool): Whether to include images in the crawl results.
- `categories` (optional, str): Filter URLs by predefined categories. Can be "Careers", "Blogs", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", or "Media". Default is None.
- `extract_depth` (optional, str): Depth of content extraction, either "basic" or "advanced". Default is "basic".
- `include_favicon` (optional, bool): Whether to include the favicon URL for each result. Default is False.
- `include_usage` (optional, bool): Whether to include credit usage information in the response. NOTE:The value may be 0 if the total use of /extract and /map calls has not yet reached minimum needed. See our [Credits & Pricing documentation](https://docs.tavily.com/documentation/api-credits) for details.
- `format` (optional, str): The format of the extracted web page content. "markdown" returns content in markdown format. "text" returns plain text and may increase latency.

For a comprehensive overview of the available parameters, refer to the [Tavily Crawl API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/crawl)

```python
from langchain_tavily import TavilyCrawl

tool = TavilyCrawl(
    max_depth=1,
    max_breadth=20,
    limit=50,
    # instructions=None,
    # select_paths=None,
    # select_domains=None,
    # exclude_paths=None,
    # exclude_domains=None,
    # allow_external=False,
    # include_images=False,
    # categories=None,
    # extract_depth=None,
    # include_favicon=False,
    # format=None
)
```

### Invoke directly with args

The Tavily crawl tool accepts the following arguments during invocation:

- `url` (required): The root URL to begin the crawl.
- All other parameters can also be set during invocation: `max_depth`, `max_breadth`, `limit`, `instructions`, `select_paths`, `select_domains`, `exclude_paths`, `exclude_domains`, `allow_external`, `include_images`, `categories`, and `extract_depth`.

NOTE: If you set an argument during instantiation this value will persist and overwrite the value passed during invocation.

```python
# Basic crawl of a website
result = tool.invoke({
    "url": "https://docs.tavily.com",
    "instructions": "Find SDK documentation",
    "categories": ["Documentation"]
})
```

output:

```bash
{
    'base_url': 'https://docs.tavily.com',
    'results': [{
        'url': 'https://docs.tavily.com/sdk/python',
        'raw_content': 'Python SDK Documentation...',
        'images': []
    },
    {
        'url': 'https://docs.tavily.com/sdk/javascript',
        'raw_content': 'JavaScript SDK Documentation...',
        'images': []
    }],
    'response_time': 10.28
}
```

## Tavily Map

Here we show how to instantiate an instance of the Tavily Map tool. After instantiation we invoke the tool with a URL. This tool allows you to create a structured map of website URLs using Tavily's Map API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `max_depth` (optional, int): Max depth of the mapping from base URL. Default is 1.
- `max_breadth` (optional, int): Max number of links to follow per page. Default is 20.
- `limit` (optional, int): Total number of links to process before stopping. Default is 50.
- `instructions` (optional, str): Natural language instructions to guide the mapping.
- `select_paths` (optional, List[str]): Regex patterns to select specific URL paths.
- `select_domains` (optional, List[str]): Regex patterns to select specific domains.
- `exclude_paths` (optional, List[str]): Regex patterns to exclude URLs with specific path patterns
- `exclude_domains` (optional, List[str]): Regex patterns to exclude specific domains or subdomains from mapping
- `allow_external` (optional, bool): Allow following external domain links. Default is False.
- `categories` (optional, str): Filter URLs by predefined categories ("Careers", "Blogs", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media").
- `include_usage` (optional, bool): Whether to include credit usage information in the response.NOTE:The value may be 0 if the total successful pages mapped has not yet reached 10 calls. See our [Credits & Pricing documentation](https://docs.tavily.com/documentation/api-credits) for details.

For a comprehensive overview of the available parameters, refer to the [Tavily Map API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/map)

```python
from langchain_tavily import TavilyMap

tool = TavilyMap(
    max_depth=2,
    max_breadth=20,
    limit=50,
    # instructions=None,
    # select_paths=None,
    # select_domains=None,
    # exclude_paths=None,
    # exclude_domains=None,
    # allow_external=False,
    # categories=None,
)
```

### Invoke directly with args

The Tavily map tool accepts the following arguments during invocation:

- `url` (required): The root URL to begin the mapping.
- All other parameters can also be set during invocation: `max_depth`, `max_breadth`, `limit`, `instructions`, `select_paths`, `select_domains`, `exclude_paths`, `exclude_domains`, `allow_external`,and `categories`.

NOTE: If you set an argument during instantiation this value will persist and overwrite the value passed during invocation.

```python
# Basic mapping of a website
result = tool.invoke({
    "url": "https://docs.tavily.com",
    "instructions": "Find SDK documentation",
    "categories": ["Documentation"]
})
```

output:

```bash
{
    'base_url': 'https://docs.tavily.com',
    'results': ['https://docs.tavily.com/sdk', 'https://docs.tavily.com/sdk/python/reference', 'https://docs.tavily.com/sdk/javascript/reference', 'https://docs.tavily.com/sdk/python/quick-start', 'https://docs.tavily.com/sdk/javascript/quick-start']
    'response_time': 10.28
}
```

## Tavily Research

Here we show how to instantiate an instance of the Tavily Research tool. After instantiation we invoke the tool with a research task description. This tool allows you to create comprehensive research reports on any topic using Tavily's Research API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `model` (optional, str): The model used by the research agent. Can be "mini" (quick, surface-level), "pro" (comprehensive, in-depth), or "auto" (automatically determined). Default is "auto".
- `output_schema` (optional, dict): JSON Schema dict for structured output format. Default is None (returns unstructured text content).
- `stream` (optional, bool): Whether to stream the research task results. Default is False.
- `citation_format` (optional, str): Citation format for sources in the research report. Can be "numbered", "mla", "apa", or "chicago". Default is "numbered".

For a comprehensive overview of the available parameters, refer to the [Tavily Research API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/research)

```python
from langchain_tavily import TavilyResearch

tool = TavilyResearch(
    model="mini",
    # output_schema=None,
    # stream=False,
    # citation_format="numbered",
)
```

### Invoke directly with args

The Tavily research tool accepts the following arguments during invocation:

- `input` (required): TThe research task or question to investigate.
- The following arguments can also be set during invocation: `model`, `output_schema`, `stream`, and `citation_format`.

NOTE: If you set an argument during instantiation this value will persist and overwrite the value passed during invocation.

```python
# Creating a research task
response = tool.invoke({
    "input": "Research the latest developments in AI",
    "model": "pro",
    "citation_format": "apa"
})
```

output:

```bash
{
    'request_id': 'abc123-def456-ghi789',
    'created_at': '2024-01-15T10:30:00Z',
    'status': 'pending',
    'input': 'Research the latest developments in AI',
    'model': 'pro'
}
```

### Retrieving research results

After creating a research task, you can retrieve the results using the `TavilyGetResearch` tool:

```python
from langchain_tavily import TavilyGetResearch

# Initialize the get research tool
get_research_tool = TavilyGetResearch()

# Retrieve results using the request_id from the research task
result = get_research_tool.invoke({
    "request_id": "abc123-def456-ghi789"
})
```

output:

```bash
{
    'request_id': 'abc123-def456-ghi789',
    'created_at': '2024-01-15T10:30:00Z',
    'completed_at': '2024-01-15T10:35:00Z',
    'status': 'completed',
    'content': 'Comprehensive research report on AI developments...',
    'sources': [
        {
            'title': 'AI Research Paper',
            'url': 'https://example.com/ai-paper',
        },
        ...
    ]
}
```

### Streaming research results

You can also stream research results as they become available:

```python
# Creating a streaming research task
stream = tool.invoke({
    "input": "Research the latest developments in AI",
    "model": "pro",
    "stream": True
})

# Processing the stream as it arrives
for chunk in stream:
    print(chunk.decode('utf-8'))
```
