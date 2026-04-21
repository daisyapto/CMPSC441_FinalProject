# logic/predict.py

import torch
from torchvision import transforms
from PIL import Image
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from Models.EnsembleCNN import Ensemble
from torch.functional import F

# logic/predict.py

import torch
from torchvision import transforms
from PIL import Image
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from torch.functional import F

def predict_image(image_path, model_num):
    classes = ["brain_tumor", "healthy"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_num == 1:
        CNN = BrainCNN()
        CNN.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    elif model_num == 2:
        CNN = NN()
        CNN.load_state_dict(torch.load("saved_models/CNN_Model2.pth", map_location=device))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    elif model_num == 3:
        CNN1 = BrainCNN()
        CNN1.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))

        CNN2 = NN()
        CNN2.load_state_dict(torch.load("saved_models/CNN_Model2.pth", map_location=device))

        CNN = Ensemble(CNN1, CNN2)

        transform1 = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    CNN.to(device)
    CNN.eval()

    # Image transform (same as training)

    # Load image
    image = Image.open(image_path).convert("RGB")
    if model_num == 1 or model_num == 2:
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)

    if model_num == 3:
        image1 = transform1(image)
        image2 = transform2(image)

        image1 = image1.unsqueeze(0)  # Add batch dimension
        image1 = image1.to(device)

        image2 = image2.unsqueeze(0)  # Add batch dimension
        image2 = image2.to(device)

    with torch.no_grad():
        if model_num == 1 or model_num == 2:
            outputs = CNN(image)
            prob = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(prob, 1)

        elif model_num == 3:
            outputs = CNN(image1, image2)
            confidence, predicted = torch.max(outputs, 1)

    return confidence.item(), classes[predicted.item()]