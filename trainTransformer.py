from dataset import DatasetProcessor
from Models import PerceiverIO, LossFunction

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

NUM_EPOCH = 100
BASE_PATH = os.path.dirname(__file__) + "/dataset"
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=128, shuffle=True)

model = PerceiverIO(128, 
                    256, 
                    8, 3, 8,
                    128).to(device)
loss_fn = LossFunction(1/1000000, 1/100,
                       1/100,device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(NUM_EPOCH):
    i = 0
    cumalative_loss = 0
    for idx, image, Y in trainLoader:
        optimizer.zero_grad()
        
        output1 = model(image)
        print(idx)
        
        loss = loss_fn(output1, Y)
        print(loss.item())
        loss.backward()
        optimizer.step()
        
        i+= 1
        cumalative_loss += loss.item()
        print(f"batches done for epoch {epoch+1}: {i}/{len(trainLoader)}")
        
    print(f"Loss for epoch {epoch+1}: {cumalative_loss/len(trainLoader)}")
    
    if epoch % 10 == 0:
        savePath = os.path.dirname(__file__) + f"/TrainedWeights/transformer_{epoch}_{epoch+1}.pth"
        torch.save(model.state_dict(), savePath)

savePath = os.path.dirname(__file__) + "/transformer.pth"
torch.save(model.state_dict(), savePath)
    
