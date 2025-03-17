# 🦜️🔗 LangChain Tavily

This package contains the LangChain integration with Tavily

[langchain-tavily](https://pypi.org/project/langchain-{tavily}/)

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


## Tavily Search Results

Here we show how to instantiate an instance of the Tavily search tool. The tool accepts various parameters to customize the search. After instantiation we invoke the tool with a simple query. This tool allows you to complete search queries using Tavily's Search API endpoint.

### [Invoke directly with args](/docs/concepts/tools)

The `TavilySearchResults` tool accepts the following arguments during invocation:
- `query` (required): A natural language search query
- `search_depth` (optional): The depth of the search ("basic" or "advanced")
- `time_range` (optional): Time range filter ("day", "week", "month", "year")
- `include_domains` (optional): List of domains to limit search results to
- `exclude_domains` (optional): List of domains to exclude from search results
- `include_images` (optional): Whether to include query-related images in response

NOTE: The optional arguments are available for ReAct agents to dynamically set, if you set a argument during instantiation and then Invoke the tool with a different value, the tool will use the value you passed during invokation. To read more about ReAct agents, check out the [ReAct agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/) documentation.

For a comprehensive overview of the available parameters, refer to the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search)

```python
from langchain_tavily import TavilySearchResults

tool = TavilySearchResults(
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=False, 
    # include_images=True,
    # include_image_descriptions=True
    # time_range="day",
    # include_domains=[...],
    # exclude_domains=[...],
)

# Basic query
tool.invoke({"query": "What happened at the last wimbledon"})
```
output:
```bash
[{'title': "Andy Murray pulls out of the men's singles draw at his last Wimbledon",
  'url': 'https://www.nbcnews.com/news/sports/andy-murray-wimbledon-tennis-singles-draw-rcna159912',
  'content': "NBC News Now LONDON — Andy Murray, one of the last decade's...",
  'score': 0.6755297},
 {'title': "He beat Roger Federer in his last ever match at Wimbledon and now he's ...",
  'url': 'https://www.thetennisgazette.com/features/i-beat-roger-federer-in-his-last-ever-match-at-wimbledon-and-now-ive-won-eight-atp-titles/',
  'content': "He beat Roger Federer in his last ever match at Wimbledon...",
  'score': 0.64128816},
    ......................................................................
 {'title': "Murray's Wimbledon farewell: The man 'who left no stone unturned'",
  'url': 'https://www.atptour.com/en/news/murray-wimbledon-2024-reflections',
  'content': "It was a moment nobody in the tennis world will soon ...",
  'score': 0.59061456}]
```

## Tavily Extract

Here we show how to instantiate an instance of the Tavily extract tool. After instantiation we invoke the tool with a single URL. Note, the tools supports invokation with a list of multiple URLs. This tool allows you to extract content from URLs using Tavily's Extract API endpoint.

### [Invoke directly with args](/docs/concepts/tools)

The `TavilyExtract` tool accepts the following arguments:

- `urls` (required): A list of URLs to extract content from.
- `extract_depth` (optional): The depth of the extraction.
- `include_images` (optional): Whether to include images in the extraction.

NOTE: The optional arguments are available for ReAct agents to dynamically set, if you set a argument during instantiation and then Invoke the tool with a different value, the tool will use the value you passed during invokation. To read more about ReAct agents, check out the [ReAct agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/) documentation.

For a comprehensive overview of the available parameters, refer to the [Tavily Extract API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/extract)

```python
from langchain_tavily import TavilyExtract

tool = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)

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

