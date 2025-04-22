from Models import SE_CNN, image_transformer as transformer, CustomLoss
from Dataset import DatasetProcessor

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from torch.nn import CrossEntropyLoss


NUM_EPOCH = 24
BASE_PATH = os.path.dirname(__file__) + "/Dataset"
print(BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = DatasetProcessor(BASE_PATH, transformer, device)
trainLoader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SE_CNN(3,64,
                64, 625).to(device)
# model.load_state_dict(torch.load(os.path.dirname(__file__) + "/TrainedWeights/CNN/24_3.pth"))

loss_fn = CustomLoss()
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(NUM_EPOCH):
    i = 0
    cumalative_loss = 0
    for image, Y, class_type in trainLoader:
        optimizer.zero_grad()
        
        output1 = model(image)
        
        loss = loss_fn(output1, Y, class_type)
        print(loss.item())
        loss.backward()
        optimizer.step()
        
        i+= 1
        cumalative_loss += loss.item()
        print(f"batches done for epoch {epoch+1}: {i}/{len(trainLoader)}")
        
    print(f"Loss for epoch {epoch+1}: {cumalative_loss/len(trainLoader)}")
    
    if (epoch+1) % 4 == 0:
        savePath = os.path.dirname(__file__) + f"/TrainedWeights/CNN/{epoch+1}_fresh.pth"
        torch.save(model.state_dict(), savePath)
