import csv
import ijson
import os
import pandas as pd
import torch
from ftfy import fix_text
from PIL import Image
from time import sleep

from Logger.Logger import logger
from Adapters import WeatherClient, WebCamClient, WeatherData

class CreateDataset():
    def __init__(self, mode: str = "json", units: str = "standard", lang: str = "en",
                 limit: int = 2, dist_range: int = 250.0,
                 outputPath: str = "./dataset",
                 imageTargetSize: tuple[int, int] = (256, 256),
                 regionsToCoverPath: str = "./cities500.json"):
        self.openWeatherFetcher = WeatherClient(
            mode=mode, units=units, lang=lang)
        self.webCamFetcher = WebCamClient(
            limit=limit, dist_range=dist_range, outputImageSize=imageTargetSize)
        self.outputPath = outputPath
        self.regionsToCoverPath = regionsToCoverPath

    def _getCities(self):
        with open(self.regionsToCoverPath, "r") as f:
            for city in ijson.items(f, "item"):
                yield city["id"], fix_text(city["name"])

    def _downloadInstance(self, openWeatherKey: str, WindyKey: str, id: int, city: str):
        logger.info(f"{__file__}: Creating dataset for {city}")
        logger.info(f"{__file__}: Fetching weather info for {city}")

        weatherInfo, weather_info_fetch_success = self.openWeatherFetcher.get_weather_info(
            openWeatherKey, location=city) # this gets the weather info from the openWeatherFetcher

        if not weather_info_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching weather info for {city}")
            return None # None is returned, because weather information did not exist for the city, so noo webcams are searched for it

        webCamIds, webcam_ID_fetch_success = self.webCamFetcher.getImageIds(
            key=WindyKey, lat=weatherInfo.lat, lon=weatherInfo.lon) # this gets the webcam ids from the webCamFetcher

        if not webcam_ID_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching webcam ids for {city}")
            return None # None is returned, because the webcam ids did not exist for the city, so no information is not returned (no point)

        webcam_ID_fetch_success, count = self.webCamFetcher.downloadImages(
            key=WindyKey, webcamIds=webCamIds, outputPath=f"{self.outputPath}/images/{id}")

        if not webcam_ID_fetch_success:
            logger.error(
                f"{__file__}: Error in fetching webcam images for {city}")
            return None

        logger.info(
            f"{__file__}: Dataset for {city} created successfully")
        writeDict = weatherInfo.__dict__.copy()

        if count == 0:
            logger.warning(
                f"{__file__}: No webcam images found for {city}")
            return None

        return writeDict

    def downloadDataset(self, openWeatherKey: str, WindyKey: str):
        with open(f"{self.outputPath}/dataset.csv", "w", newline='') as csvfile:
            fieldnames = ["id"] + WeatherData().getNames
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for id, city in self._getCities():
                infoDict = self._downloadInstance(
                    openWeatherKey, WindyKey, id, city)
                if infoDict is None:
                    continue

                writer.writerow(
                    {'id': id, **infoDict})
                csvfile.flush()
                print(f"Dataset for {city} created successfully")
                sleep(0.1)
                
class DatasetProcessor:
    def __init__(self,datasetRootPATH:str, transformer:callable, device:torch.device):
        self.root = datasetRootPATH
        self.dataset = pd.read_csv(f"{datasetRootPATH}/dataset.csv",header=None).loc[1:]
        self.transformer = transformer
        self.device = device
    
    @property
    def shape(self):
        return self.dataset.shape
    
    def __convToFloat(self, row: pd.Series, indices:list[int]):
        nums = row.loc[indices]
        nums = [float(num) for num in nums]
        return nums
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        row = self.dataset.iloc[idx]
        image_index = row[0]
        
        X = self.__convToFloat(row, [5,10,11])
        Y= self.__convToFloat(row, [1,2,3,4,6,7,8,9])
        
        image_path = f"{self.root}/images/{image_index}/"
        img_store = []
        for img in os.listdir(image_path):
            image = Image.open(f"{image_path}{img}")
            image = self.transformer(image)
            img_store.append(image)
        
        image_tensor = torch.stack(img_store).to(self.device)
        return image_tensor, torch.tensor(X,device=self.device), torch.tensor(Y,device=self.device)
