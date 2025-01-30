from AI_Model.Image_Classifier import FYP_CNN
from dataset import DataLoader
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

NUM_EPOCH = 10
BASE_PATH = "dataset/images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv("dataset/dataset.csv",header=None)

def splitDataset_X_Y(dataset: pd.DataFrame):
    X = dataset.loc[:,["id","local_time","lat","lon"]]
    Y = dataset.loc[:,["temperature","humidity","wind_speed","cloud_cover","visibility","gust","rain_1h","snow_1h"]]
    
    return X, Y

trainData, testData = random_split(torch.tensor(dataset), [int(len(dataset)*0.8), int(len(dataset)*0.2)])
trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)

model = FYP_CNN(3,64,
                3,64,
                256,8)

for epoch in range(NUM_EPOCH):
    for data in trainLoader:
        print(data)