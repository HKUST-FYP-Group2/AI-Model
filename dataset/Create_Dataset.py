import json
import ijson
from AI_logger.logger import logger
from Adapters.WebCam_Client import WebCamClient
from Adapters.Weather_Client import WeatherClient
from time import sleep
from ftfy import fix_text
from json_stream.dump import JSONStreamEncoder, default


class CreateDataset():
    def __init__(self, mode: str = "json", units: str = "metric", lang: str = "en",
                 limit: int = 2, dist_range: int = 250.0,
                 outputPath: str = "./dataset",
                 regionsToCoverPath: str = "./cities500.json"):
        self.openWeatherFetcher = WeatherClient(
            mode=mode, units=units, lang=lang)
        self.webCamFetcher = WebCamClient(limit=limit, dist_range=dist_range)
        self.outputPath = outputPath
        self.regionsToCoverPath = regionsToCoverPath

    def _getCities(self):
        with open(self.regionsToCoverPath, "r") as f:
            count = 0
            for city in ijson.items(f, "item"):
                yield city["id"], fix_text(city["name"])

    def _downloadInstance(self, openWeatherKey: str, WindyKey: str, id: int, city: str):
        logger.info(f"{__file__}: Creating dataset for {city}")
        logger.info(f"{__file__}: Fetching weather info for {city}")

        weatherInfo, weather_status = self.openWeatherFetcher.get_weather_info(
            openWeatherKey, location=city)

        if not weather_status:
            logger.error(
                f"{__file__}: Error in fetching weather info for {city}")
            return None

        webCamIds, success = self.webCamFetcher.getImageIds(
            key=WindyKey, lat=weatherInfo.lat, lon=weatherInfo.lon)

        if not success:
            logger.error(
                f"{__file__}: Error in fetching webcam ids for {city}")
            return None

        success, count = self.webCamFetcher.downloadImages(
            key=WindyKey, webcamIds=webCamIds, outputPath=f"{self.outputPath}/images/{id}")

        if not success:
            logger.error(
                f"{__file__}: Error in fetching webcam images for {city}")
            return None

        logger.info(
            f"{__file__}: Dataset for {city} created successfully")
        writeDict = weatherInfo.__dict__.copy()

        return writeDict

    def downloadDataset(self, openWeatherKey: str, WindyKey: str):
        f = open(f"{self.outputPath}/dataset.json", "w", buffering=1)
        f.write("{")  # Start of JSON array
        first = True
        for id, city in self._getCities():
            infoDict = self._downloadInstance(
                openWeatherKey, WindyKey, id, city)
            if infoDict is None:
                continue

            first = False
            f.write(f'"{id}":')
            json.dump(infoDict, f, cls=JSONStreamEncoder)
            if not first:
                f.write(",")
            f.write("\n")

            sleep(0.1)
        f.write("}")  # End of JSON array
