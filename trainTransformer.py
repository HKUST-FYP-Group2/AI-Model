from dataset import DatasetProcessor
from AI_Model.Image_Transformer import PerceiverIO
from AI_Model.Image_Classifier import LossFunction

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import os

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

NUM_EPOCH = 10
BASE_PATH = os.path.join(os.path.dirname(__file__), "/dataset")
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PerceiverIO(128, 
                    256, 
                    8, 3, 8,
                    128).to(device)
loss_fn = LossFunction(device=device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCH):
    for images, X, Y in trainLoader:
        img1 = images[:, 0, ...]
        output1 = model(img1)
        loss = loss_fn(output1, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "./transformer.pth")
    
