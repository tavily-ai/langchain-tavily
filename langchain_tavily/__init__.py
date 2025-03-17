from importlib import metadata

from langchain_tavily.tavily_extract import TavilyExtract
from langchain_tavily.tavily_search import TavilySearchResults

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "TavilySearchResults",
    "TavilyExtract",
    "__version__",
]
