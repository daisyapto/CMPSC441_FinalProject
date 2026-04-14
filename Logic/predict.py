# logic/predict.py

import torch
from torchvision import transforms
from PIL import Image
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from torch.functional import F

def predict_image(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model == 1:
        model = BrainCNN()
        model.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    elif model == 2:
        model = NN()
        model.load_state_dict(torch.load("saved_models/CNN_Model2.pth", map_location=device))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    model.to(device)
    model.eval()

    # Image transform (same as training)

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["brain_tumor", "healthy"]

    return classes[predicted.item()], confidence.item()
