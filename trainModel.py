from AI_Model.Image_Classifier import FYP_CNN, LossFunction
from dataset import DatasetProcessor

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

model = FYP_CNN(3,32,
                3, 32,
                128, 8).to(device)

loss_fn = LossFunction(device=device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(NUM_EPOCH):
    for images, X, Y in trainLoader:
        optimizer.zero_grad()
        
        img1 = images[:, 0, ...]
        output1 = model(img1, X)
        
        img2 = images[:, 1, ...]       
        output2 = model(img2, X)
        
        loss1 = loss_fn(output1, Y)
        loss2 = loss_fn(output2, Y)
        loss = (loss1 + loss2)/2
        
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} completed")
    
torch.save(model.state_dict(), "./model.pth")