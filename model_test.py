import torch
from torchvision import transforms
from Models import SE_CNN
import os
from PIL import Image
from Dataset import DatasetProcessor
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

weights = torch.load("TrainedWeights/CNN/44.pth",weights_only=True)
model = SE_CNN(3,64,
                64, 625).to(device)
model.load_state_dict(weights)


BASE_PATH = os.path.dirname(__file__) + "/dataset"
dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

for img, X, Y in trainLoader:
    # image = Image.open(f"TestImages/{img}")
    # image = transformer(image).unsqueeze(0)
    img = img[:,0,...]
    output = model(img)

    print(output)

    print(f"Image {img} done")

