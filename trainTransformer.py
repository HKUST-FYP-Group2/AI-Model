from Dataset import DatasetProcessor
from Models import PerceiverIO, image_transformer as transformer

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

NUM_EPOCH = 104
BASE_PATH = os.path.dirname(__file__) + "/dataset"
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
print(len(dataset))
trainLoader = DataLoader(dataset, batch_size=64, shuffle=True)

model = PerceiverIO(256, 
                    256, 
                    625, 32, 16,
                    512).to(device)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

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
    
    scheduler.step()
    print(f"Loss for epoch {epoch+1}: {cumalative_loss/len(trainLoader)}")
    
    if (epoch+1) % 4 == 0:
        savePath = os.path.dirname(__file__) + f"/TrainedWeights/Transformer/{epoch+1}.pth"
        torch.save(model.state_dict(), savePath)
    
