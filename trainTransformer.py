from dataset import DatasetProcessor
from Models import PerceiverIO

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import os
from torch.nn import CrossEntropyLoss

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

NUM_EPOCH = 104
BASE_PATH = os.path.dirname(__file__) + "/dataset"
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
print(len(dataset))
trainLoader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PerceiverIO(128, 
                    256, 
                    625, 32, 16,
                    512).to(device)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4)

model.train()
for epoch in range(NUM_EPOCH):
    i = 0
    cumalative_loss = 0
    for image, Y in trainLoader:
        optimizer.zero_grad()
        output1 = model(image)
        loss = loss_fn(output1, Y)
        print(loss.item())
        loss.backward()
        optimizer.step()
        
        i+= 1
        cumalative_loss += loss.item()
        print(f"batches done for epoch {epoch+1}: {i}/{len(trainLoader)}")
        
    print(f"Loss for epoch {epoch+1}: {cumalative_loss/len(trainLoader)}")
    
    if (epoch+1) % 4 == 0:
        savePath = os.path.dirname(__file__) + f"/TrainedWeights/Transformer/{epoch+1}.pth"
        torch.save(model.state_dict(), savePath)
    
