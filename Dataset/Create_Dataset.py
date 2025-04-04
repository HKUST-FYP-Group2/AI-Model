import csv
import os
from time import sleep
from Logger.Logger import logger
from Adapters import WeatherClient, WebCamClient, WeatherData

class CreateDataset():
    def __init__(self, mode: str = "json", units: str = "standard", lang: str = "en",
                 limit: int = 2, dist_range: int = 250.0,
                 basePath = "",
                 outputPath: str = "dataset",
                 imageTargetSize: tuple[int, int] = (256, 256),
                 regionsToCoverName: str = "worldcities.csv"):
        
        self.openWeatherFetcher = WeatherClient(
            mode=mode, units=units, lang=lang)
        self.webCamFetcher = WebCamClient(
            limit=limit, dist_range=dist_range, outputImageSize=imageTargetSize)
        self.outputPath = f"{basePath}/{outputPath}"
        self.regionsToCoverPath = f"{basePath}/{regionsToCoverName}"

    def _getCities(self):
        with open(self.regionsToCoverPath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield int(row["id"]), row["city"]

    def _downloadInstance(self, openWeatherKey: str, WindyKey: str, id: int, city: str):
        logger.info(f"{__file__}: Creating dataset for {city}")
        logger.info(f"{__file__}: Fetching weather info for {city}")

        weatherInfo, weather_info_fetch_success = self.openWeatherFetcher.get_weather_info(
            openWeatherKey, location=city) # this gets the weather info from the openWeatherFetcher

        if not weather_info_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching weather info for {city}")
            return None, 0 # None is returned, because weather information did not exist for the city, so noo webcams are searched for it

        webCamIds, webcam_ID_fetch_success = self.webCamFetcher.getImageIds(
            key=WindyKey, lat=weatherInfo.lat, lon=weatherInfo.lon) # this gets the webcam ids from the webCamFetcher

        if not webcam_ID_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching webcam ids for {city}")
            return None, 0 # None is returned, because the webcam ids did not exist for the city, so no information is not returned (no point)

        webcam_ID_fetch_success, count = self.webCamFetcher.downloadImages( # this part downloads the images from the webCamFetcher
            key=WindyKey, webcamIds=webCamIds, outputPath=f"{self.outputPath}/images/{id}")

        if not webcam_ID_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching webcam images for {city}")
            return None, 0

        logger.info(
            f"{__file__}: Dataset for {city} created successfully")
        writeDict = weatherInfo.__dict__.copy()

        if count == 0:
            logger.warning(
                f"{__file__}: No webcam images found for {city}")
            return None, 0

        return writeDict, len(webCamIds)
    
    def __getExstingIds(self):
        if not os.path.exists(f"{self.outputPath}/dataset.csv"):
            def noFileThereFunction():
                while True:
                    yield None
            return noFileThereFunction, False
        
        def FileThereFunction():
            with open(f"{self.outputPath}/dataset.csv", "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    yield int(row["id"])
                    
        return FileThereFunction, True

    def downloadDataset(self, openWeatherKey: str, WindyKey: str):

        iteratorFunc, fileExists = self.__getExstingIds()

        with open(f"{self.outputPath}/dataset.csv", "a", newline='') as csvfile:
            fieldnames = ["id"] + WeatherData().getNames + ["num_images"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if the file is new
            if not fileExists:
                writer.writeheader()

            for id, city in self._getCities():
                if id == next(iteratorFunc()):
                    logger.info(
                        f"{__file__}: skipping Dataset for {city}, since it already exists")
                    continue

                infoDict, numImages = self._downloadInstance(openWeatherKey, WindyKey, id, city)
                if infoDict is None:
                    continue

                writer.writerow({'id': id, **infoDict, 'num_images': numImages})
                csvfile.flush()
                sleep(0.1)