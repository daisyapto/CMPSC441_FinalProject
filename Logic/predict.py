import os
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random

from Models.EnsembleCNN import Ensemble
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN

CLASSES = ["Brain Tumor", "Healthy"]
TEST_DIR = "Data/test2"


# LOAD MODEL + TRANSFORMS
def load_model(model_type, device):
    if model_type == "cnn1":
        model = BrainCNN()
        model.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        return model, (transform,)

    elif model_type == "cnn2":
        model = NN()
        model.load_state_dict(torch.load("saved_models/CNN_Model2.pth", map_location=device))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        return model, (transform,)

    elif model_type == "ensemble":
        cnn1 = BrainCNN()
        cnn1.load_state_dict(torch.load("saved_models/brain_cnn.pth", map_location=device))

        cnn2 = NN()
        cnn2.load_state_dict(torch.load("saved_models/CNN_Model2.pth", map_location=device))

        model = Ensemble(cnn1, cnn2)

        transform1 = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        return model, (transform1, transform2)

    else:
        raise ValueError("Invalid model_type")


# RUN SINGLE IMAGE
def run_inference(model, transforms_tuple, image, device):
    with torch.no_grad():
        if len(transforms_tuple) == 1:
            img = transforms_tuple[0](image).unsqueeze(0).to(device)
            outputs = model(img)

        else:  # ensemble
            img1 = transforms_tuple[0](image).unsqueeze(0).to(device)
            img2 = transforms_tuple[1](image).unsqueeze(0).to(device)
            outputs = model(img1, img2)

        # ensure consistent probability output
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return predicted.item(), confidence.item()


# DATABASE TESTING
def predict_database(model_type, max_images=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transforms_tuple = load_model(model_type=model_type, device=device)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    results = []

    for label_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(TEST_DIR, class_name)

        if not os.path.exists(class_path):
            print(f"Path not found: {class_path}")
            continue

        # GET + RANDOMIZE FILES
        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        random.shuffle(files)  # RANDOMIZE ORDER

        for file in files:
            if max_images is not None and total >= max_images:
                break

            image_path = os.path.join(class_path, file)
            image = Image.open(image_path).convert("RGB")

            pred_label, confidence = run_inference(
                model, transforms_tuple, image, device
            )

            total += 1
            if pred_label == label_idx:
                correct += 1

            results.append({
                "file": file,
                "true": class_name,
                "predicted": CLASSES[pred_label],
                "confidence": confidence
            })

        # BREAK OUTER LOOP TOO
        if max_images is not None and total >= max_images:
            break

    accuracy = correct / total if total > 0 else 0

    return {
        "mode": "database",
        "prediction": None,
        "confidence": None,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "results": results
    }


# MANUAL IMAGE PREDICTION
def predict_image(model_type):
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an X-ray Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transforms_tuple = load_model(model_type, device)

    model.to(device)
    model.eval()

    image = Image.open(file_path).convert("RGB")

    pred_label, confidence = run_inference(
        model, transforms_tuple, image, device
    )

    return {
        "mode": "manual",
        "prediction": CLASSES[pred_label],
        "confidence": confidence,
        "accuracy": None,
        "total": None,
        "correct": None,
        "results": None
    }


def predict(model_type, test_type, max_images=None):
    if test_type == "manual":
        return predict_image(model_type=model_type)
    else:
        return predict_database(model_type=model_type, max_images=max_images)
