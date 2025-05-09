# import kagglehub
import json
import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from random import randint

from Models import SE_CNN, image_transformer as transformer
from utils import decimal_to_pentanary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download latest version
parent_dir = os.path.dirname(__file__) + "/dataset/"

num_rain_images = len(os.listdir(parent_dir + "rain/"))
num_rime_images = len(os.listdir(parent_dir + "rime/"))
num_snow_images = len(os.listdir(parent_dir + "snow/"))

all_types_start_index = {"rain": 0, "rime": num_rain_images, "snow": num_rain_images + num_rime_images, "normal": num_rain_images + num_rime_images + num_snow_images}
        
model = SE_CNN(3,64,
                64, 625).to(device)

model.load_state_dict(torch.load(os.path.dirname(__file__) + "/TrainedWeights/CNN/24_3.pth"))

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
                3: wind_speed >= 8.33333 or rain_1h:>=30
                4: wind_speed >= 11.111 or rain_1h:>=50 
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