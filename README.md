# 🦜️🔗 LangChain Tavily

This package contains the LangChain integration with Tavily

[langchain-tavily](https://pypi.org/project/langchain-tavily/)

[Tavily website](https://tavily.com/)

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
- `topic` (optional, str): Category of the search. Can be "general" or "news". Default is "general".
- `include_answer` (optional, bool): Include an answer to original query in results. Default is False.
- `include_raw_content` (optional, bool): Include cleaned and parsed HTML of each search result. Default is False.
- `include_images` (optional, bool): Include a list of query related images in the response. Default is False.
- `include_image_descriptions` (optional, bool): Include descriptive text for each image. Default is False.
- `search_depth` (optional, str): Depth of the search, either "basic" or "advanced". Default is "advanced".
- `time_range` (optional, str): The time range back from the current date to filter results - "day", "week", "month", or "year". Default is None.
- `include_domains` (optional, List[str]): List of domains to specifically include. Default is None.
- `exclude_domains` (optional, List[str]): List of domains to specifically exclude. Default is None.

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
    # search_depth="advanced",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)
```

### Invoke directly with args

The Tavily search tool accepts the following arguments during invocation:
- `query` (required): A natural language search query
- The following arguments can also be set during invokation : `include_images`, `search_depth` , `time_range`, `include_domains`, `exclude_domains`, `include_images`
- For reliability and performance reasons, certain parameters that affect response size cannot be modified during invocation: `include_answer` and `include_raw_content`. These limitations prevent unexpected context window issues and ensure consistent results.


NOTE: The optional arguments are available for agents to dynamically set, if you set a argument during instantiation and then invoke the tool with a different value, the tool will use the value you passed during invokation.

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
### Chaining

We can use our tool in a chain by first binding it to a [tool-calling model](/docs/how_to/tool_calling/) and then calling it. This gives the agent the ability to dynamically set the available arguments to the Tavily search tool.

In the below example when we ask the agent to find "Latest news on the stock market from the New York Times" the agent will dynamically set the argments for the Tavily search tool to {'arguments': '{"query":"latest stock market news","include_domains":["nytimes.com"],"time_range":"day"}'}


```python
# !pip install -qU langchain langchain-openai langchain-tavily
import datetime

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# specifying tool_choice will force the model to call this tool.
llm_with_tools = llm.bind_tools([tool])

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("who won the super bowl this year")
```


## Tavily Extract

Here we show how to instantiate an instance of the Tavily extract tool. After instantiation we invoke the tool with a list of URLs. This tool allows you to extract content from URLs using Tavily's Extract API endpoint.

### Instantiation

The tool accepts various parameters during instantiation:

- `extract_depth` (optional, str): The depth of the extraction, either "basic" or "advanced". Default is "advanced".
- `include_images` (optional, bool): Whether to include images in the extraction. Default is False.

For a comprehensive overview of the available parameters, refer to the [Tavily Extract API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/extract)

```python
from langchain_tavily import TavilyExtract

tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False,
)
```

### Invoke directly with args

The Tavily extract tool accepts the following arguments during invocation:
- `urls` (required): A list of URLs to extract content from. 
- Both `extract_depth` and `include_images` can also be set during invokation

NOTE: The optional arguments are available for agents to dynamically set, if you set a argument during instantiation and then invoke the tool with a different value, the tool will use the value you passed during invokation. 

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

## Tavily Research Agent

This example demonstrates how to build a powerful web research agent using Tavily's search and extract capabilities integrated with LangChain's OpenAIFunctionsAgent.

### Features

- Internet Search: Query the web for up-to-date information using Tavily's search API
- Content Extraction: Extract and analyze specific content from web pages
- Seamless Integration: Works with OpenAI's function calling capability for reliable tool use
- Source Citation: Automatically includes sources with links in responses

```python
# !pip install -qU langchain langchain-openai langchain-tavily
"""Test Tavily search and extract with OpenAIFunctionsAgent."""
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
import os

@tool
def tavily_search(query: str) -> str:
    """Search the internet for information. 
    Input should be a search query string."""
    search = TavilySearch(max_results=3)
    return search.invoke(query)

@tool
def tavily_extract(urls: list) -> str:
    """Extract content from specified URLs.
    Input should be a list of URLs to extract content from."""
    extract = TavilyExtract(extract_depth="basic")
    return extract.invoke({"urls": urls})

def test_tavily_agent(query="What are the latest developments in quantum computing?"):
    """Test Tavily search and extract with OpenAIFunctionsAgent."""
    # Initialize tools
    tools = [tavily_search, tavily_extract]
    
    # Create the system message
    system_message = SystemMessage(
        content="""You are a web researcher who answers user questions by looking up information 
        on the internet and retrieving contents of helpful documents.
        
        When using the extract tool, provide a list of URLs directly.
        
        Always cite your sources with links."""
    )
    
    try:
        # Create OpenAI Functions agent
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
        agent_prompt = OpenAIFunctionsAgent.create_prompt(system_message)
        agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=agent_prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=10
        )
        
        # Run query
        result = agent_executor.invoke({"input": query})
        print("\nFinal Answer:", result["output"])
        return result["output"]
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_tavily_agent()
```
