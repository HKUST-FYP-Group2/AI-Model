from AI_Model.Image_Classifier import FYP_CNN
from dataset import DatasetProcessor
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
BASE_PATH = "./dataset/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DatasetProcessor(BASE_PATH, transformer)
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

for image, X, Y in trainLoader:
    print(image.shape, X.shape, Y.shape)
    break