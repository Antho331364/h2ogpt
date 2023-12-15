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


class Beds24APIWrapper(BaseModel):
    client: Any
    beds24api_token: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    params: dict = Field(
        default={
        }
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that token and python package exists in environment."""
        beds24api_token = get_from_dict_or_env(
            values, "beds24api_token", "beds24api_token"
        )
        values["beds24api_token"] = beds24api_token
        try:
            from beds_24_api_v2_client import AuthenticatedClient
            client = AuthenticatedClient(base_url="https://beds24.com/api/v2/", token=beds24api_token)

            values["client"] = client
        except ImportError:
            raise ValueError(
                "Could not import beds_24_api_v2_client python package. "
                "Please install it with `pip install --index-url https://{{TOKEN_NAME}}:{{"
                "TOKEN_PWD}}@gitlab.com/api/v4/projects/52750088/packages/pypi/simple --no-deps beds-24-api-v2-client`."
            )
        return values

    async def arun(self, q: str, **kwargs: Any) -> str:
        """Run query through Beds24API and parse result async."""
        return self._process_response(await self.aresults(q))

    def run(self, q: str, **kwargs: Any) -> str:
        """Run query through Beds24API and parse result."""
        return self._process_response(self.results(q))

    def results(self, q: str) -> dict:
        """Run query through Beds24API and return the raw result."""
        params = self.get_params(q)
        with HiddenPrints():
            search = self.client.search(params)
            res = search.as_dict()
        return res

    async def aresults(self, q: str) -> dict:
        """Use aiohttp to run query through Beds24API and return the results async."""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(q)
            params["source"] = "python"
            if self.beds24api_token:
                params["beds24api_token"] = self.beds24api_token
            params["output"] = "json"
            url = "https://beds24.com/api/v2/"
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

    def get_params(self, q: str) -> Dict[str, str]:
        """Get parameters for Beds24API."""
        _params = {
            "token": self.beds24api_token,
            "q": q,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from Beds24API."""
        if "error" in res.keys():
            raise ValueError(f"Got error from Beds24API: {res['error']}")

        return ""
