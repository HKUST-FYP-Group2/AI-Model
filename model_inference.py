from Models import SE_CNN
import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

model = SE_CNN(3,64,
                64, 625).to(device)
model.load_state_dict(torch.load("/home/hvgupta/FYP/AI-Model/deployedModel.pth"))
model.eval()

def decimal_to_pentanary(decimal_number):
    if decimal_number == 0:
        return "0"
    
    pentanary_number = ""
    while decimal_number > 0:
        remainder = decimal_number % 5
        pentanary_number = str(remainder) + pentanary_number
        decimal_number //= 5
    
    return pentanary_number

def classify_image(images):
    model.eval()
    images_tensor = []
    for image in images:
        image = transformer(image).unsqueeze(0)
        images_tensor.append(image)
    images_tensor = torch.cat(images_tensor, dim=0)
    with torch.no_grad():
        outputs = model(images_tensor)
    
    converted_outputs = []
    for output in outputs:
        converted_outputs.append(decimal_to_pentanary(output.item()))
    
    return converted_outputs
    