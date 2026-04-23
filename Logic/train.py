# Unused in main GUI running - used predict.py for all
import os

import torch
import torch.optim as optim
import torch.nn as nn

from Logic.test import test_model
from Models.CNN_Model import BrainCNN
from Utils.data_loader import get_data_loaders


def train_model(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss}")


def train_and_save():
    train_loader, test_loader = get_data_loaders()

    model = BrainCNN()

    train_model(model, train_loader)
    test_model(model, test_loader)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/brain_cnn.pth")

    print("Model trained and saved.\n")