"""Unit tests for api_base_url parameter."""

from unittest.mock import Mock, patch

from langchain_tavily import TavilySearch, TavilyExtract, TavilyCrawl, TavilyMap
from langchain_tavily._utilities import (
    TavilySearchAPIWrapper,
    TavilyExtractAPIWrapper,
    TavilyCrawlAPIWrapper,
    TavilyMapAPIWrapper,
)


class TestApiBaseUrl:
    """Test the api_base_url parameter functionality."""

    def test_tavily_search_api_base_url(self):
        """Test that TavilySearch accepts and uses api_base_url parameter."""
        custom_base_url = "https://custom-api.example.com"
        
        search = TavilySearch(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        assert search.api_base_url == custom_base_url
        assert search.api_wrapper.api_base_url == custom_base_url

    def test_tavily_extract_api_base_url(self):
        """Test that TavilyExtract accepts and uses api_base_url parameter."""
        custom_base_url = "https://custom-api.example.com"
        
        extract = TavilyExtract(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        assert extract.api_base_url == custom_base_url
        assert extract.apiwrapper.api_base_url == custom_base_url

    def test_tavily_crawl_api_base_url(self):
        """Test that TavilyCrawl accepts and uses api_base_url parameter."""
        custom_base_url = "https://custom-api.example.com"
        
        crawl = TavilyCrawl(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        assert crawl.api_base_url == custom_base_url
        assert crawl.api_wrapper.api_base_url == custom_base_url

    def test_tavily_map_api_base_url(self):
        """Test that TavilyMap accepts and uses api_base_url parameter."""
        custom_base_url = "https://custom-api.example.com"
        
        map_tool = TavilyMap(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        assert map_tool.api_base_url == custom_base_url
        assert map_tool.api_wrapper.api_base_url == custom_base_url

    def test_api_base_url_defaults_to_none(self):
        """Test that api_base_url defaults to None when not provided."""
        search = TavilySearch(tavily_api_key="test_key")
        
        assert search.api_base_url is None
        assert search.api_wrapper.api_base_url is None

    def test_api_base_url_only_parameter(self):
        """Test that api_base_url can be set without tavily_api_key."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "env_key"}):
            custom_base_url = "https://custom-api.example.com"
            
            search = TavilySearch(api_base_url=custom_base_url)
            
            assert search.api_base_url == custom_base_url
            assert search.api_wrapper.api_base_url == custom_base_url

    @patch("requests.post")
    def test_search_wrapper_uses_custom_base_url(self, mock_post):
        """Test that TavilySearchAPIWrapper uses custom base URL in requests."""
        custom_base_url = "https://custom-api.example.com"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        wrapper = TavilySearchAPIWrapper(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        wrapper.raw_results(
            query="test query",
            max_results=5,
            search_depth="basic",
            include_domains=None,
            exclude_domains=None,
            include_answer=None,
            include_raw_content=None,
            include_images=None,
            include_image_descriptions=None,
            include_favicon=None,
            topic=None,
            time_range=None,
            country=None,
            auto_parameters=None,
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{custom_base_url}/search"

    @patch("requests.post")
    def test_extract_wrapper_uses_custom_base_url(self, mock_post):
        """Test that TavilyExtractAPIWrapper uses custom base URL in requests."""
        custom_base_url = "https://custom-api.example.com"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        wrapper = TavilyExtractAPIWrapper(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        wrapper.raw_results(
            urls=["https://example.com"],
            extract_depth="basic",
            include_images=None,
            include_favicon=None,
            format=None,
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{custom_base_url}/extract"

    @patch("requests.post")
    def test_crawl_wrapper_uses_custom_base_url(self, mock_post):
        """Test that TavilyCrawlAPIWrapper uses custom base URL in requests."""
        custom_base_url = "https://custom-api.example.com"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        wrapper = TavilyCrawlAPIWrapper(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        wrapper.raw_results(
            url="https://example.com",
            max_depth=1,
            max_breadth=20,
            limit=50,
            instructions=None,
            select_paths=None,
            select_domains=None,
            exclude_paths=None,
            exclude_domains=None,
            allow_external=None,
            include_images=None,
            categories=None,
            extract_depth=None,
            include_favicon=None,
            format=None,
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{custom_base_url}/crawl"

    @patch("requests.post")
    def test_map_wrapper_uses_custom_base_url(self, mock_post):
        """Test that TavilyMapAPIWrapper uses custom base URL in requests."""
        custom_base_url = "https://custom-api.example.com"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        wrapper = TavilyMapAPIWrapper(
            tavily_api_key="test_key",
            api_base_url=custom_base_url
        )
        
        wrapper.raw_results(
            url="https://example.com",
            max_depth=1,
            max_breadth=20,
            limit=50,
            instructions=None,
            select_paths=None,
            select_domains=None,
            exclude_paths=None,
            exclude_domains=None,
            allow_external=None,
            categories=None,
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{custom_base_url}/map"

    @patch("requests.post")
    def test_wrapper_uses_default_base_url_when_none(self, mock_post):
        """Test that wrappers use default base URL when api_base_url is None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_post.return_value = mock_response
        
        wrapper = TavilySearchAPIWrapper(
            tavily_api_key="test_key",
            api_base_url=None
        )
        
        wrapper.raw_results(
            query="test query",
            max_results=5,
            search_depth="basic",
            include_domains=None,
            exclude_domains=None,
            include_answer=None,
            include_raw_content=None,
            include_images=None,
            include_image_descriptions=None,
            include_favicon=None,
            topic=None,
            time_range=None,
            country=None,
            auto_parameters=None,
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.tavily.com/search"