"""Util that calls Tavily Search + Extract API.

In order to set this up, follow instructions at:
https://docs.tavily.com/docs/tavily-api/introduction
"""

from typing import Any, AsyncGenerator, Dict, Generator, List, Literal, Optional, Union
from urllib.parse import quote

import aiohttp
import requests
from aiohttp import ClientTimeout
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from langchain_tavily._errors import (
    TRANSPORT_ERRORS,
    response_error,
    response_json,
    sync_response_error,
    sync_response_json,
    transport_error,
)

TAVILY_API_URL: str = "https://api.tavily.com"


def _request_headers(
    api_key: SecretStr,
    client_name: str,
    project_id: Optional[str],
    session_id: Optional[str],
    human_id: Optional[str],
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key.get_secret_value()}",
        "Content-Type": "application/json",
        "X-Client-Source": "api",
        "X-Client-Name": client_name,
    }
    for name, value in (
        ("X-Project-ID", project_id),
        ("X-Session-Id", session_id),
        ("X-Human-Id", human_id),
    ):
        if value is not None:
            headers[name] = value
    return headers


class TavilySearchAPIWrapper(BaseModel):
    """Wrapper for Tavily Search API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None
    client_name: str = "langchain-tavily"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    human_id: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        query: str,
        max_results: Optional[int],
        search_depth: Optional[Literal["basic", "advanced", "fast", "ultra-fast"]],
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_images: Optional[bool],
        include_image_descriptions: Optional[bool],
        include_favicon: Optional[bool],
        topic: Optional[Literal["general", "news", "finance"]],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        country: Optional[str],
        auto_parameters: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
        include_usage: Optional[bool],
        exact_match: Optional[bool],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_favicon": include_favicon,
            "topic": topic,
            "time_range": time_range,
            "country": country,
            "auto_parameters": auto_parameters,
            "start_date": start_date,
            "end_date": end_date,
            "include_usage": include_usage,
            "exact_match": exact_match,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )
        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/search",
            json=params,
            headers=headers,
        )
        if response.status_code != 200:
            raise sync_response_error(
                response, self.tavily_api_key.get_secret_value()
            )
        return sync_response_json(response)

    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int],
        search_depth: Optional[Literal["basic", "advanced", "fast", "ultra-fast"]],
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_images: Optional[bool],
        include_image_descriptions: Optional[bool],
        include_favicon: Optional[bool],
        topic: Optional[Literal["general", "news", "finance"]],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        country: Optional[str],
        auto_parameters: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
        include_usage: Optional[bool],
        exact_match: Optional[bool],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get results from the Tavily Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "include_favicon": include_favicon,
                "topic": topic,
                "time_range": time_range,
                "country": country,
                "auto_parameters": auto_parameters,
                "start_date": start_date,
                "end_date": end_date,
                "include_usage": include_usage,
                "exact_match": exact_match,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = _request_headers(
                self.tavily_api_key, self.client_name, self.project_id,
                self.session_id, self.human_id,
            )
            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/search", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise response_error(
                            res.status, await res.text(),
                            self.tavily_api_key.get_secret_value(),
                        )

        results_json_str = await fetch()

        return response_json(results_json_str)


class TavilyExtractAPIWrapper(BaseModel):
    """Wrapper for Tavily Extract API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None
    client_name: str = "langchain-tavily"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    human_id: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        format: Optional[str],
        include_usage: Optional[bool],
        query: Optional[str],
        chunks_per_source: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {
            "urls": urls,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "extract_depth": extract_depth,
            "format": format,
            "include_usage": include_usage,
            "query": query,
            "chunks_per_source": chunks_per_source,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/extract",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            raise sync_response_error(
                response, self.tavily_api_key.get_secret_value()
            )
        return sync_response_json(response)

    async def raw_results_async(
        self,
        urls: List[str],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        extract_depth: Optional[Literal["basic", "advanced"]],
        format: Optional[str],
        include_usage: Optional[bool],
        query: Optional[str],
        chunks_per_source: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get results from the Tavily Extract API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "urls": urls,
                "include_images": include_images,
                "include_favicon": include_favicon,
                "extract_depth": extract_depth,
                "format": format,
                "include_usage": include_usage,
                "query": query,
                "chunks_per_source": chunks_per_source,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = _request_headers(
                self.tavily_api_key, self.client_name, self.project_id,
                self.session_id, self.human_id,
            )

            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/extract", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise response_error(
                            res.status, await res.text(),
                            self.tavily_api_key.get_secret_value(),
                        )

        results_json_str = await fetch()

        return response_json(results_json_str)


class TavilyCrawlAPIWrapper(BaseModel):
    """Wrapper for Tavily Crawl API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None
    client_name: str = "langchain-tavily"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    human_id: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        include_images: Optional[bool],
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
        include_usage: Optional[bool],
        chunks_per_source: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "include_images": include_images,
            "categories": categories,
            "extract_depth": extract_depth,
            "include_favicon": include_favicon,
            "format": format,
            "include_usage": include_usage,
            "chunks_per_source": chunks_per_source,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/crawl",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            raise sync_response_error(
                response, self.tavily_api_key.get_secret_value()
            )
        return sync_response_json(response)

    async def raw_results_async(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        include_images: Optional[bool],
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
        include_usage: Optional[bool],
        chunks_per_source: Optional[int],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get results from the Tavily Crawl API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "include_images": include_images,
                "categories": categories,
                "extract_depth": extract_depth,
                "include_favicon": include_favicon,
                "format": format,
                "include_usage": include_usage,
                "chunks_per_source": chunks_per_source,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = _request_headers(
                self.tavily_api_key, self.client_name, self.project_id,
                self.session_id, self.human_id,
            )

            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/crawl", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise response_error(
                            res.status, await res.text(),
                            self.tavily_api_key.get_secret_value(),
                        )

        results_json_str = await fetch()

        return response_json(results_json_str)


class TavilyMapAPIWrapper(BaseModel):
    """Wrapper for Tavily Map API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None
    client_name: str = "langchain-tavily"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    human_id: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ],
        include_usage: Optional[bool],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "categories": categories,
            "include_usage": include_usage,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/map",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            raise sync_response_error(
                response, self.tavily_api_key.get_secret_value()
            )
        return sync_response_json(response)

    async def raw_results_async(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ],
        include_usage: Optional[bool],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get results from the Tavily Map API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "categories": categories,
                "include_usage": include_usage,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = _request_headers(
                self.tavily_api_key, self.client_name, self.project_id,
                self.session_id, self.human_id,
            )
            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/map", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise response_error(
                            res.status, await res.text(),
                            self.tavily_api_key.get_secret_value(),
                        )

        results_json_str = await fetch()

        return response_json(results_json_str)


class TavilyResearchAPIWrapper(BaseModel):
    """Wrapper for Tavily Research API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None
    client_name: str = "langchain-tavily"
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    human_id: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        input: str,
        research_model: Optional[Literal["mini", "pro", "auto"]],
        output_schema: Optional[Dict[str, Any]],
        stream: Optional[bool],
        citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Generator[bytes, None, None]]:
        params = {
            "input": input,
            "model": research_model,
            "output_schema": output_schema,
            "stream": stream,
            "citation_format": citation_format,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )
        base_url = self.api_base_url or TAVILY_API_URL
        
        if stream:
            response = requests.post(
                # type: ignore
                f"{base_url}/research",
                json=params,
                headers=headers,
                stream=True,
            )
            if response.status_code != 200:
                try:
                    raise sync_response_error(
                        response, self.tavily_api_key.get_secret_value()
                    )
                finally:
                    response.close()
            
            def stream_generator() -> Generator[bytes, None, None]:
                try:
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                except TRANSPORT_ERRORS as e:
                    raise transport_error(e) from None
                finally:
                    response.close()
            
            return stream_generator()
        else:
            response = requests.post(
                # type: ignore
                f"{base_url}/research",
                json=params,
                headers=headers,
            )
            if response.status_code not in (200, 201):
                raise sync_response_error(
                    response, self.tavily_api_key.get_secret_value()
                )
            return sync_response_json(response)

    async def raw_results_async(
        self,
        input: str,
        research_model: Optional[Literal["mini", "pro", "auto"]],
        output_schema: Optional[Dict[str, Any]],
        stream: Optional[bool],
        citation_format: Optional[Literal["numbered", "mla", "apa", "chicago"]],
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncGenerator[bytes, None]]:
        """Get results from the Tavily Research API asynchronously."""

        params = {
            "input": input,
            "model": research_model,
            "output_schema": output_schema,
            "stream": stream,
            "citation_format": citation_format,
            **kwargs,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )

        base_url = self.api_base_url or TAVILY_API_URL
        
        if stream is True:
            timeout = ClientTimeout(total=None)
            
            async def stream_generator() -> AsyncGenerator[bytes, None]:
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{base_url}/research", json=params, headers=headers
                        ) as res:
                            if res.status != 200:
                                error_text = await res.text()
                                raise response_error(
                                    res.status, error_text,
                                    self.tavily_api_key.get_secret_value(),
                                )
                            
                            async for chunk in res.content.iter_any():
                                if chunk:
                                    yield chunk
                except TRANSPORT_ERRORS as e:
                    raise transport_error(e) from None
            
            return stream_generator()
        
        # Non-streaming response
        async def fetch() -> str:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/research", json=params, headers=headers
                ) as res:
                    if res.status in (200, 201):
                        data = await res.text()
                        return data
                    else:
                        raise response_error(
                            res.status, await res.text(),
                            self.tavily_api_key.get_secret_value(),
                        )

        results_json_str = await fetch()

        return response_json(results_json_str)

    def get_research(
        self,
        request_id: str,
        include_usage: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get research results by request_id."""
        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )
        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.get(
            # type: ignore
            f"{base_url}/research/{quote(request_id, safe='')}",
            headers=headers,
            params=(
                {} if include_usage is None
                else {"include_usage": str(include_usage).lower()}
            ),
        )
        if response.status_code not in [200, 202]:
            raise sync_response_error(
                response, self.tavily_api_key.get_secret_value()
            )
        return sync_response_json(response)

    async def get_research_async(
        self,
        request_id: str,
        include_usage: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get research results by request_id asynchronously."""
        headers = _request_headers(
            self.tavily_api_key, self.client_name, self.project_id,
            self.session_id, self.human_id,
        )
        base_url = self.api_base_url or TAVILY_API_URL
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/research/{quote(request_id, safe='')}", headers=headers,
                params=(
                    {} if include_usage is None
                    else {"include_usage": str(include_usage).lower()}
                ),
            ) as res:
                if res.status in [200, 202]:
                    data = await res.text()
                    return response_json(data)
                else:
                    raise response_error(
                        res.status, await res.text(),
                        self.tavily_api_key.get_secret_value(),
                    )
