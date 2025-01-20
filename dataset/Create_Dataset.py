import ijson.dump
from __init__ import *
import ijson
from AI_logger.logger import logger
from Adpaters.WebCam_Client import WebCamClient
from Adpaters.Weather_Client import WeatherClient


class CreateDataset():
    def __init__(self, mode: str = "json", units: str = "metric", lang: str = "en",
                 limit: int = 5, dist_range: int = 250.0,
                 outputPath: str = "./dataset",
                 regionsToCoverPath: str = "./cities500.json"):
        self.openWeatherFetcher = WeatherClient(
            mode=mode, units=units, lang=lang)
        self.webCamFetcher = WebCamClient(limit=limit, dist_range=dist_range)
        self.outputPath = outputPath
        self.regionsToCoverPath = regionsToCoverPath

    def _getCities(self):
        with open(self.regionsToCoverPath, "r") as f:
            for city in ijson.items(f, "item"):
                yield city["id"], city["name"]

    def _writeData(self, f, info: dict):
        f.write(ijson.dumps(info))

    def downloadDataset(self):
        f = open(f"{self.outputPath}/dataset.json", "w")
        for id, city in self._getCities():
            logger.info(f"{__file__}: Creating dataset for {city}")
            logger.info(f"{__file__}: Fetching weather info for {city}")

            weatherInfo, weather_status = self.openWeatherFetcher.get_weather_info(
                os.getenv("openWeather_api_key"), location=city)

            if not weather_status:
                logger.error(
                    f"{__file__}: Error in fetching weather info for {city}")
                continue

            webCamIds, success = self.webCamFetcher.getImageIds(
                key=os.getenv("Windy_webCam_api_key"), lat=weatherInfo.lat, lon=weatherInfo.lon)

            if not success:
                logger.error(
                    f"{__file__}: Error in fetching webcam ids for {city}")
                continue

            success, count = self.webCamFetcher.downloadImages(
                key=os.getenv("Windy_webCam_api_key"),  webcamIds=webCamIds, outputPath=f"{self.outputPath}/{id}")

            if not success:
                logger.error(
                    f"{__file__}: Error in fetching webcam images for {city}")
                continue

            logger.info(f"{__file__}: Dataset for {city} created successfully")
            writeDict = weatherInfo.__dict__.update(
                {"city_id": id, "image_id": webCamIds})
            if count > 0:
                self._writeData(f, writeDict)


CreateDataset().downloadDataset()
