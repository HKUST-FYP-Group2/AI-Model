import kagglehub
import json
import os
import torch

from Models import SE_CNN
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from random import randint

from utils import decimal_to_pentanary
from Models import image_transformer as transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download latest version
path = kagglehub.dataset_download("wjybuqi/weathertime-classification-with-road-images")

BASE_DIR = "/home/hvgupta/.cache/kagglehub/datasets/wjybuqi/weathertime-classification-with-road-images/versions/2/train_dataset"
with open(f"{BASE_DIR}/train.json") as f:
    data = json.load(f)
    
data = data["annotations"]
rainy_images_name = []
for item in data:
    if item["weather"] == "Rainy":
        rainy_images_name.append(item["filename"].split("\\")[-1])
        
model = SE_CNN(3,64,
                64, 625).to(device)

model.load_state_dict(torch.load(os.path.dirname(__file__) + "/TrainedWeights/CNN/24_1.pth"))
model.eval()

all_images = []     
classification = []
for image in rainy_images_name:
    src = f"{BASE_DIR}/train_images/{image}"
    image_file = Image.open(src)
    image_file = transformer(image_file).to(device)
    model_classification = model(image_file.unsqueeze(0))
    model_classification = torch.argmax(model_classification, dim=1)
    classification.append(model_classification.item())
    all_images.append(image_file)

# Define the loss function and optimizer

all_images = torch.stack(all_images).to(device)
classification = torch.tensor(classification).to(device)

class custom_loss(torch.nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        # Custom loss calculation
        new_label = []
        for label in labels:
            penatary_number: str = decimal_to_pentanary(label.item())
            penatary_list = list(penatary_number)
            penatary_list[2] = str(randint(1, 4))
            penatary_number = ''.join(penatary_list)
            classification = int(penatary_number[0])*125 + int(penatary_number[1])*25 + int(penatary_number[2])*5 + int(penatary_number[3])
            new_label.append(classification)
        new_label = torch.tensor(new_label).to(device)
        loss = self.cross_entropy(outputs, new_label)
        return loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = custom_loss()

# Training loop
epochs = 24
# Create a DataLoader for batching
dataset = TensorDataset(all_images, classification)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_images, batch_labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_images)
        
        loss = criterion(outputs, batch_labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    

torch.save(model.state_dict(), os.path.dirname(__file__) + "/TrainedWeights/CNN/24_3.pth")



    