"""Chain that calls SerpAPI.

Heavily borrowed from https://github.com/ofirpress/self-ask
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

import aiohttp

from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


ALLOWED_TRAVEL_MODE = {
    "Best": 6,
    "Driving": 0,
    "Two-wheeler": 9,
    "Transit": 3,
    "Walking": 2,
    "Cycling": 1,
    "Flight": 4
}


def validate_travel_mode(v: int) -> int:
    allowed_values = ""
    for k, v in ALLOWED_TRAVEL_MODE.items():
        allowed_values += str(v) + " for " + k + ", "

    assert v in ALLOWED_TRAVEL_MODE.values(), "Allowed values are: " + allowed_values
    return v


class SerpAPIGoogleMapsInput(BaseModel):
    start_addr: str = Field(description="Start address")
    end_addr: str = Field(description="End address")
    travel_mode: Optional[int] = Field(default=6, description="Parameter defines the travel mode. Allowed values are: "
                                                              "6 for Best, 0 for Driving, 9 for Two-wheeler, "
                                                              "3 for Transit, 2 for Walking, 1 for Cycling, "
                                                              "4 for Flight.")


class SerpAPIGoogleMapsWrapper(BaseModel):
    """Wrapper around SerpAPI.

    To use, you should have the ``google-maps`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            import SerpAPIGoogleMapsWrapper
            serpapimaps = SerpAPIGoogleMapsWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google_maps_directions",
            "gl": "fr",
            "hl": "fr",
        }
    )
    serpapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serpapi_api_key = get_from_dict_or_env(
            values, "serpapi_api_key", "SERPAPI_API_KEY"
        )
        values["serpapi_api_key"] = serpapi_api_key
        try:
            import serpapi
            client = serpapi.Client(api_key=serpapi_api_key)

            values["search_engine"] = client
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please install it with `pip install serpapi`."
            )
        return values

    async def arun(self, start_addr: str, end_addr: str, travel_mode: Optional[int], **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result async."""
        if travel_mode is not None:
            validate_travel_mode(travel_mode)
        return self._process_response(await self.aresults(start_addr, end_addr, travel_mode))

    def run(self, start_addr: str, end_addr: str, travel_mode: Optional[int], **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result."""
        if travel_mode is not None:
            validate_travel_mode(travel_mode)
        return self._process_response(self.results(start_addr, end_addr, travel_mode))

    def results(self, start_addr: str, end_addr: str, travel_mode: Optional[int]) -> dict:
        """Run query through SerpAPI and return the raw result."""
        if travel_mode is not None:
            validate_travel_mode(travel_mode)
        params = self.get_params(start_addr, end_addr, travel_mode)
        with HiddenPrints():
            search = self.search_engine.search(params)
            res = search.as_dict()
        return res

    async def aresults(self, start_addr: str, end_addr: str, travel_mode: Optional[int]) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""
        if travel_mode is not None:
            validate_travel_mode(travel_mode)

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(start_addr, end_addr, travel_mode)
            params["source"] = "python"
            if self.serpapi_api_key:
                params["serp_api_key"] = self.serpapi_api_key
            params["output"] = "json"
            url = "https://serpapi.com/search"
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return res

    def get_params(self, start_addr: str, end_addr: str, travel_mode: Optional[int]) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        _params = {
            "api_key": self.serpapi_api_key,
            "start_addr": start_addr,
            "end_addr": end_addr,
            "travel_mode": travel_mode
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")

        if "directions" in res.keys():
            return res["directions"]
