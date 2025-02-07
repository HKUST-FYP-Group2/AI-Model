from AI_Model.Image_Classifier import FYP_CNN, LossFunction
from dataset import DatasetProcessor

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
BASE_PATH = os.path.dirname(__file__) + "/dataset"
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FYP_CNN(3,128,
                128, 8).to(device)

loss_fn = LossFunction(device=device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(NUM_EPOCH):
    i = 0
    for images, X, Y in trainLoader:
        optimizer.zero_grad()
        
        img1 = images[:, 0, ...]
        output1 = model(img1)
        loss = loss_fn(output1, Y)
        
        loss.backward()
        optimizer.step()
        
        i+= 1
        print(f"batches done for epoch {epoch+1}: {i}/{len(trainLoader)}")
        
    print(f"Epoch {epoch+1} completed")
    
torch.save(model.state_dict(), "./CNN.pth")