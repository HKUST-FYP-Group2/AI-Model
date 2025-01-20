from ._HTTP_Client import BaseFetcher
from AI_logger.logger import logger
import asyncio


class Fetcher(BaseFetcher):
    def __init__(self, url: str, limit: int, range: int):
        super().__init__(url)
        self.limit = limit
        self.range = min(max(int(range), 0), 250.0)

    def getLinks(self, response: dict) -> list[int]:
        linkArray = []
        for info in response["webcams"]:
            linkArray.append(info["webcamId"])

        return linkArray

    def fetch(self, key=None, lat: float = None, lon: float = None):
        if not lat or not lon:
            logger.error(f"{__file__}: Please provide both lat and lon")
            return [], False

        _param = {
            "limit": self.limit, "nearby": f"{lat},{lon},{int(self.range)}", "include": "urls"}

        logger.info(
            f"{__file__}: Calling the api {self.url}, passed params {_param}")
        _header = {"x-windy-api-key": key}
        response, status = asyncio.run(self._fetch(_header, _param))

        if not status:
            return [], status

        return self.getLinks(response), status
