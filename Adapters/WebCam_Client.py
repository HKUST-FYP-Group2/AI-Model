from ._HTTP_Client import HttpClient
import asyncio
import os
from Logger.Logger import logger
from PIL import Image
from io import BytesIO


class WebCamClient(HttpClient):
    def __init__(self, limit: int, dist_range: int, outputImageSize: tuple[int, int]):
        super().__init__()
        self.url = "https://api.windy.com/webcams/api/v3/webcams"
        self.limit = limit
        self.dist_range = dist_range
        self.outputImageSize = outputImageSize

    def _call_API(self, url: str, key: str, **param):
        _header = {"x-windy-api-key": key}
        logger.info(
            f"{__file__}: Fetching data from {self.url}, params: {param}")
        response, success = asyncio.run(super()._fetch(url, _header, param))
        return response, success

    def _getImageLinks(self, key=None, lat: float = None, lon: float = None):

        response, status = self._call_API(self.url, key, lat=lat, lon=lon,
                                          limit=self.limit, nearby=f"{lat},{lon},{int(self.dist_range)}", include="urls")

        if not status:
            return [], status

        linkArray: list[int] = []
        for info in response["webcams"]:
            linkArray.append(info["webcamId"])

        return linkArray, status

    async def _download(self, response, id: int, outputPath: str):
        imgLink = response["images"]["current"]["preview"]
        outputPath = f"{outputPath}/{id}.jpg"
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)

        logger.info(f"{__file__}: Downloading the image for {id}")

        with open(outputPath, 'wb') as f:
            response, status = await super()._fetch(imgLink, return_json=False)

            if not status:
                logger.error(
                    f"{__file__}: Error in downloading the image for {id}")
                return

            # Resize the image
            image = Image.open(BytesIO(response))
            resized_image = image.resize(self.outputImageSize)

            # Save the resized image
            resized_image.save(outputPath)

    def getImageIds(self, key=None, lat: float = None, lon: float = None):
        if not lat or not lon or not key:
            logger.error(f"{__file__}: Missing arguement")
            return [], False

        webcamIds, success = self._getImageLinks(key, lat, lon)
        return webcamIds, success

    def downloadImages(self, key=None, webcamIds: list[int] = None, outputPath: str = None):
        if not webcamIds or not key or not outputPath:
            logger.error(f"{__file__}: Missing arguement")
            return False, 0

        for webcamId in webcamIds:
            downloadURL = f"{self.url}/{webcamId}"
            response, success = self._call_API(
                downloadURL, key, webcamId=webcamId, include="images")

            if not success:
                logger.error(
                    f"{__file__}: Error in fetching the Image link for {webcamId}")
                continue

            logger.info(f"{__file__}: Downloading the image for {webcamId}")
            asyncio.run(self._download(response, webcamId, outputPath))

        return True, len(webcamIds)
