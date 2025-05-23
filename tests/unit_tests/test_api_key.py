from langchain_tavily._utilities import TavilySearchAPIWrapper


def test_api_wrapper_api_key_not_visible() -> None:
    """Test that an exception is raised if the API key is not present."""
    wrapper = TavilySearchAPIWrapper(tavily_api_key="abcd123")  # type: ignore[arg-type]
    assert "abcd123" not in repr(wrapper)
