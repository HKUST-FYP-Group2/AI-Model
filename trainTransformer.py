from dataset import DatasetProcessor
from AI_Model.Image_Transformer import PerceiverIO
from AI_Model.Image_Classifier import LossFunction

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

NUM_EPOCH = 10
BASE_PATH = "./dataset/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PerceiverIO(128, 
                    256, 
                    8, 3, 8,
                    128).to(device)
loss_fn = LossFunction(device=device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)


for images, X, Y in trainLoader:
    img1 = images[:, 0, ...]
    output1 = model(img1)
    loss = loss_fn(output1, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
