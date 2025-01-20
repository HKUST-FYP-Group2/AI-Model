from ._HTTP_Client import HttpClient
from AI_logger.logger import logger
import asyncio
import json


class WebCamClient(HttpClient):
    def __init__(self, limit: int, dist_range: int, outputPath: str):
        super().__init__()
        self.url = "https://api.windy.com/webcams/api/v3/webcams"
        self.limit = limit
        self.dist_range = dist_range
        self.outputPath = outputPath

    def _call_API(self, url: str, key: str, **param):
        _header = {"x-windy-api-key": key}
        logger.info(
            f"{__file__}: Fetching data from {self.url}, params: {param}")
        response, success = asyncio.run(super()._fetch(url, _header, param))
        return response, success

    def _getImageLinks(self, key=None, lat: float = None, lon: float = None) -> list[int]:
        response, status = self._call_API(self.url, key, lat=lat, lon=lon,
                                          limit=self.limit, nearby=f"{lat},{lon},{int(self.dist_range)}", include="urls")

        if not status:
            return [], status

        with open("webcam.json", "w") as f:
            f.write(json.dumps(response, indent=4))

        linkArray = []
        for info in response["webcams"]:
            linkArray.append(info["webcamId"])

        return linkArray, status

    async def _download(self, response, id: int):
        imgLink = response["images"]["current"]["preview"]
        outputPath = f"{self.outputPath}/{id}.jpg"

        logger.info(f"{__file__}: Downloading the image for {id}")

        with open(outputPath, 'wb') as f:
            response, status = await super()._fetch(imgLink, return_json=False)

            if not status:
                logger.error(
                    f"{__file__}: Error in downloading the image for {id}")
                return

            f.write(response)

    def downloadImages(self, key=None, lat: float = None, lon: float = None):
        if not lat or not lon:
            logger.error(f"{__file__}: Please provide both lat and lon")
            return False

        webcamIds, success = self._getImageLinks(key, lat, lon)

        if not success:
            logger.error(f"{__file__}: unable to fetch the webcamIds")
            return False

        for webcamId in webcamIds:
            downloadURL = f"{self.url}/{webcamId}"
            response, success = self._call_API(
                downloadURL, key, webcamId=webcamId, include="images")

            if not success:
                logger.error(
                    f"{__file__}: Error in fetching the Image link for {webcamId}")
                continue

            logger.info(f"{__file__}: Downloading the image for {webcamId}")
            asyncio.run(self._download(response, webcamId))

        return True
