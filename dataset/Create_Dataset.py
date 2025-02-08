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
                yield int(row["id"]), row["city_ascii"]

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

        webcam_ID_fetch_success, count = self.webCamFetcher.downloadImages( # this part downloads the images from the webCamFetcher
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
        existing_ids = set()

        # Check if the dataset.csv file exists and read existing city IDs
        if os.path.exists(f"{self.outputPath}/dataset.csv"):
            with open(f"{self.outputPath}/dataset.csv", "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_ids.add(int(row["id"]))

        with open(f"{self.outputPath}/dataset.csv", "a", newline='') as csvfile:
            fieldnames = ["id"] + WeatherData().getNames
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if the file is new
            if not existing_ids:
                writer.writeheader()

            for id, city in self._getCities():
                if id in existing_ids:
                    logger.info(
                        f"{__file__}: skipping Dataset for {city}, since it already exists")
                    continue

                infoDict = self._downloadInstance(openWeatherKey, WindyKey, id, city)
                if infoDict is None:
                    continue

                writer.writerow({'id': id, **infoDict})
                csvfile.flush()
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
            break
        
        image_tensor = torch.stack(img_store).to(self.device)
        return image_tensor, torch.tensor(X,device=self.device), torch.tensor(Y,device=self.device)
