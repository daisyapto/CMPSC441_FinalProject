# logic/predict.py

import torch
from torchvision import transforms
from PIL import Image
from Models.CNN_Model import BrainCNN

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BrainCNN()
    model.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # Image transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ["brain_tumor", "healthy"]

    return classes[predicted.item()]