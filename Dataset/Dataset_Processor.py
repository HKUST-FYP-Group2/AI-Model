import pandas as pd
import torch
import os
from PIL import Image


class DatasetProcessor:
    def __init__(
        self, datasetRootPATH: str, transformer: callable, device: torch.device
    ):
        self.root = datasetRootPATH
        self.dataset = pd.read_csv(f"{datasetRootPATH}/dataset.csv", dtype=float)
        self.dataset["num_images"].astype(int)
        self.transformer = transformer
        self.device = device
        self.__fixDataset()

    def __fixDataset(self):
        self.dataset = self.dataset.dropna()
        self.dataset.reset_index(drop=True, inplace=True)  # I hate you pandas
        self.dataset["cumsum"] = self.dataset["num_images"].cumsum()

    @property
    def shape(self):
        return self.dataset.shape

    def __len__(self):
        return int(self.dataset.iloc[-1, -1])

    def _calcHotCold(self, temp: float, snow: float):
        if snow > 0:
            return 0
        return (temp >= 0) * 1 + (temp >= 10) * 1 + (temp >= 25) * 1 + (temp >= 35) * 1

    def _calcDryWet(self, rain: float, snow: float):
        return (
            (rain > 0 or snow > 0) * 1
            + (rain >= 30 or snow >= 2) * 1
            + (rain >= 50 or snow >= 5) * 1
            + (rain >= 70 or snow >= 10) * 1
        )

    def _calcClearCloudy(self, cloud: float):
        return (
            (cloud > 0) * 1 + (cloud >= 10) * 1 + (cloud >= 30) * 1 + (cloud >= 70) * 1
        )

    def _calcCalmStormy(self, wind: float, rain: float, snow: float):
        return (
            (wind >= 0.556) * 1
            + (wind >= 3.333) * 1
            + (wind >= 8.333 or rain >= 30 or snow >= 2) * 1
            + (wind >= 11.111 or rain >= 50 or snow >= 5) * 1
        )

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

        return (
            coldhotval + 5 * drywetval + 25 * clearcloudyval + 125 * calmstormyval
        )  # penta-nary classification, turning a multi-label into a classificaiton problem

    def __getAllIdx(self, idx):
        # Find the row index where cumulative sum is just greater than idx
        row_index = (self.dataset["cumsum"] <= idx).sum()
        # Get the city ID for the corresponding row
        city_id = self.dataset.iloc[row_index, 0]

        # Calculate the previous cumulative sum (prev_count)
        if row_index == 0:
            prev_count = 0
        else:
            first_occurance_idx = self.dataset.index[
                self.dataset.iloc[:, 0] == city_id
            ][0]
            prev_count = self.dataset.iloc[first_occurance_idx - 1, -1]
        # Calculate the relative index (idx - prev_count)
        relative_idx = idx - prev_count
        return int(row_index), int(relative_idx)

    def __getitem__(self, globalImageIdx):
        cityIdx, localImageIdx = self.__getAllIdx(globalImageIdx)
        row = self.dataset.iloc[cityIdx]
        cityId = int(row["id"])
        Y = torch.tensor(
            (
                row["temperature"],
                row["wind_speed"],
                row["cloud_cover"],
                row["visibility"],
                row["rain_1h"],
                row["snow_1h"],
            ),
            device=self.device,
        )

        image_path = f"{self.root}/images/{cityId}/"
        choosenImage = os.listdir(image_path)[localImageIdx]
        image = Image.open(f"{image_path}{choosenImage}").convert("RGB")
        image = self.transformer(image)

        image_tensor = image.to(self.device)
        return image_tensor, self.__getTheClass(Y)
