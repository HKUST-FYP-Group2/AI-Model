from ._HTTP_Client import BaseFetcher
from AI_logger.logger import logger
import asyncio
import aiohttp


class _Fetcher(BaseFetcher):
    def __init__(self, url: str):
        super().__init__(url)

    def _call_API(self, key: str, **param):
        _header = {"x-windy-api-key": key}
        logger.info(
            f"{__file__}: Fetching data from {self.url}, params: {param}")
        response, status = asyncio.run(super()._fetch(_header, param))
        return response, status


class ImageLink_Fetcher(_Fetcher):
    def __init__(self, url, limit: int, dist_range: float):
        super().__init__(url)
        self.limit = limit
        self.dist_range = dist_range

    def _getLinks(self, response: dict) -> list[int]:
        linkArray = []
        for info in response["webcams"]:
            linkArray.append(info["webcamId"])

        return linkArray

    def fetch(self, key=None, lat: float = None, lon: float = None):
        if not lat or not lon:
            logger.error(f"{__file__}: Please provide both lat and lon")
            return [], False

        response, status = super()._call_API(
            key, lat=lat, lon=lon, limit=self.limit, nearby=f"{lat},{lon},{int(self.dist_range)}", include="urls")

        if not status:
            return [], status

        return self._getLinks(response), status


class Image_Fetcher(_Fetcher):
    def __init__(self, url: str, outputPath: str):
        super().__init__(url)
        self.outputPath = outputPath

    async def _download(self, response, id: int):
        imgLink = response["images"]["current"]["preview"]
        outputPath = f"{self.outputPath}/{id}.jpg"

        logger.info(f"{__file__}: Downloading the image for {id}")

        with aiohttp.ClientSession() as session:
            async with session.get(imgLink) as response:
                if response.status == 200:
                    with open(outputPath, 'wb') as f:
                        f.write(await response.read())
                    logger.info(f"{__file__}: Image downloaded for {id}")
                else:
                    logger.error(
                        f"{__file__}: Error in downloading the image for {id}")

    def fetch(self, key=None, webcamIds: list[int] = None):
        if not webcamIds:
            logger.error(f"{__file__}: Please provide the webcamId(s)")
            return False

        for webcamId in webcamIds:
            response, status = super()._call_API(key, webcamId=webcamId, include="images")

            if not status:
                logger.error(
                    f"{__file__}: Error in fetching the Image link for {webcamId}")
                continue

            logger.info(f"{__file__}: Downloading the image for {webcamId}")
            asyncio.run(self._download(response))

        return True
