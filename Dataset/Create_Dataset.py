import csv
import os
import pandas as pd
import torch
from PIL import Image
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
                
class DatasetProcessor:
    def __init__(self,datasetRootPATH:str, transformer:callable, device:torch.device):
        self.root = datasetRootPATH
        self.dataset = pd.read_csv(f"{datasetRootPATH}/dataset.csv", dtype=float)
        self.dataset["num_images"].astype(int)
        self.transformer = transformer
        self.device = device
        self.__fixDataset()
    
    def __fixDataset(self):
        self.dataset = self.dataset.dropna()
    
    @property
    def shape(self):
        return self.dataset.shape
    
    def __len__(self):
        return int(self.dataset.iloc[:,-1].sum())
    
    def _calcHotCold(self, temp:float, snow:float):
        if snow > 0:
            return 0
        return (temp >= 0)*1 + (temp >= 10)*1 + (temp >= 25)*1 + (temp >= 35)*1
    
    def _calcDryWet(self, rain:float, snow:float):
        return (rain > 0 or snow > 0)*1 + (rain >= 30 or snow >= 2)*1 + (rain >= 50 or snow >= 5)*1 + (rain >= 70 or snow >= 10)*1
    
    def _calcClearCloudy(self, cloud:float):
        return (cloud > 0)*1 + (cloud >= 10)*1 + (cloud >= 30)*1 + (cloud >= 70)*1
    
    def _calcCalmStormy(self, wind:float, rain:float, snow:float):
        return (wind >= 0.556)*1 + (wind >= 3.333)*1 + (wind >= 8.333 or rain >= 30 or snow >= 2)*1 + (wind >= 11.111 or rain >= 50 or snow >= 5)*1
    
    def __getTheClass(self, data):
        """
            I will have temperature, humidity, wind_speed, cloud_cover, visibility, gust, rain_1h, snow_1h
            need to use this information to classify it 5 levels for each category of cold-hot, dry-wet, calm-stormy, clear-cloudy
            
            cold-hot: 
                0: temperature < 0 or snow_1h > 0
                1: temperature >= 10
                2: temperature >= 20
                3: temperature >= 25
                4: temperature >= 35
            dry-wet:
                0: rain_1h == 0 and snow_1h == 0
                1: rain_1h > 0 or snow_1h > 0
                2: rain_1h >= 30 or snow_1h >= 2
                3: rain_1h >= 50 or snow_1h >= 5
                4: rain_1h >= 70 or snow_1h >= 10
            clear-cloudy:
                0: cloud_cover = 0
                1: cloud_cover > 0
                2: cloud_cover >= 10
                3: cloud_cover >= 30
                4: cloud_cover > 70
            calm-stormy:
                0: wind_speed < 0.556
                1: wind_speed >= 0.556
                2: wind_speed >= 3.333
                3: wind_speed >= 8.33333 or rain_1h:>=30 or snow_1h:>=2
                4: wind_speed >= 11.111 or rain_1h:>=50 or snow_1h:>=5
        """

        coldhotval = self._calcHotCold(data[0], data[5])
        drywetval = self._calcDryWet(data[4], data[5])
        clearcloudyval = self._calcClearCloudy(data[2])
        calmstormyval = self._calcCalmStormy(data[1], data[4], data[5])
        
        return coldhotval + 5*drywetval + 25*clearcloudyval + 125*calmstormyval # penta-nary classification, turning a multi-label into a classificaiton problem
    
    def __getAllIdx(self, idx):
        cumalativeSum = self.dataset.iloc[:, -1].cumsum() < idx
        prevLimit = self.dataset[cumalativeSum].iloc[:, -1].sum()
        return int(cumalativeSum.sum()), int(idx - prevLimit - 1)
    
    def __getitem__(self, globalImageIdx):
        cityIdx, localImageIdx = self.__getAllIdx(globalImageIdx)
        row = self.dataset.iloc[cityIdx]
        
        cityId = int(row["id"])
        Y = torch.tensor((
                                    row["temperature"], row["wind_speed"], 
                                    row["cloud_cover"],row["visibility"],
                                    row["rain_1h"],row["snow_1h"]
                                ), 
            device=self.device)
        
        image_path = f"{self.root}/images/{cityId}/"
        choosenImage = os.listdir(image_path)[localImageIdx]
        image = Image.open(f"{image_path}{choosenImage}")
        image = self.transformer(image)
        
        image_tensor = image.to(self.device)
        return image_tensor, self.__getTheClass(Y)
