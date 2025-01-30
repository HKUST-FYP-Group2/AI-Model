from AI_Model.Image_Classifier import FYP_CNN
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

NUM_EPOCH = 10
BASE_PATH = "dataset/images"

dataset = pd.read_csv("dataset/dataset.csv")

print(dataset.isnull().sum(axis=0))
