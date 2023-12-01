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


ALLOWED_FILTERS = {
    "Today's Events": "date:today",
    "Tomorrow's Events": "date:tomorrow",
    "This Week's Events": "date:week",
    "Next Week's Events": "date:next_week",
    "This Weekend": "date:weekend",
    "This Month's Events": "date:month",
    "Next Month's Events": "date:next_month"
}


def validate_filters(v: str) -> str:
    allowed_values = ""
    for k, v in ALLOWED_FILTERS.items():
        allowed_values += str(v) + " for " + k + ", "

    assert v in ALLOWED_FILTERS.values(), "Allowed values are: " + allowed_values
    return v


class SerpAPIGoogleEventsInput(BaseModel):
    q: str = Field(description="Parameter defines the query you want to search. To search for events in a specific "
                               "location, just include the location inside your search query (e.g. Events in Paris, "
                               "France).")
    htichips: Optional[str] = Field(description="Parameter allows the use of different filters. Allowed values are: "
                                                "date:today for Today's Events, date:tomorrow for Tomorrow's Events, "
                                                "date:week for This Week's Events, date:next_week for Next Week's "
                                                "Events, date:weekend for This Weekend, date:month for This Month's "
                                                "Events, date:next_month for Next Month's Events.")


class SerpAPIGoogleEventsWrapper(BaseModel):
    """Wrapper around SerpAPI.

    To use, you should have the ``google-search`` python package installed,
    and the environment variable ``SERPAPI_API_KEY`` set with your API key, or pass
    `serpapi_api_key` as a named parameter to the constructor.

    Example:
        .. code-block:: python

            import SerpAPIGoogleEventsWrapper
            serpapimaps = SerpAPIGoogleEventsWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google_events",
            "gl": "us",
            "hl": "en",
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

    async def arun(self, q: str, htichips: Optional[str] = None, **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result async."""
        if htichips is not None:
            validate_filters(htichips)
        return self._process_response(await self.aresults(q, htichips))

    def run(self, q: str, htichips: Optional[str] = None, **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result."""
        if htichips is not None:
            validate_filters(htichips)
        return self._process_response(self.results(q, htichips))

    def results(self, q: str, htichips: Optional[str] = None) -> dict:
        """Run query through SerpAPI and return the raw result."""
        if htichips is not None:
            validate_filters(htichips)
        params = self.get_params(q, htichips)
        with HiddenPrints():
            search = self.search_engine.search(params)
            res = search.as_dict()
        return res

    async def aresults(self, q: str, htichips: Optional[str] = None) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""
        if htichips is not None:
            validate_filters(htichips)

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(q, htichips)
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

    def get_params(self, q: str, htichips: Optional[str] = None) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        _params = {
            "api_key": self.serpapi_api_key,
            "q": q,
            "htichips": htichips
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")

        if "events_results" in res.keys() and res["events_results"]:
            print(res["events_results"])
            return res["events_results"]
