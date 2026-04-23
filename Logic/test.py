# Unused in main GUI running - used predict.py for all
import torch
import torchvision
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchvision import transforms
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from Models.EnsembleCNN import Ensemble
import torch.nn.functional as F


def test_model(model_type):
    TEST_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test2/" # Replace with file path
    test_loader2 = None # Set as None in case unused

    if model_type == "cnn1":
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((128, 128))
        ])
        test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
        model = BrainCNN()
        model.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth"))

    elif model_type == "cnn2":
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])
        test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
        model = NN()
        model.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth"))

    elif model_type == "ensemble":
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((128, 128))
        ])
        test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
        model1 = BrainCNN()
        model1.load_state_dict(
            torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth"))

        transform2 = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])
        test_data2 = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
        test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=64, shuffle=True)
        model2 = NN()
        model2.load_state_dict(
            torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth"))

        model = Ensemble(model1, model2)

    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    model.eval()

    with torch.no_grad():
        if test_loader2 is None:
            for images, labels in test_loader:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

        if test_loader2 is not None:
            for (images, labels), (images2, labels2) in zip(test_loader, test_loader2):
                outputs = model(images, images2)
                probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

    print(f"{model.__class__.__name__} Test Accuracy: ", accuracy(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Precision: ", precision(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Recall: ", recall(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test F1 Score: ", f1(predicted, labels).item() * 100, "%")
    print()
    return {precision : precision(predicted, labels).item() * 100,
            recall : recall(predicted, labels).item() * 100,
            f1 : f1(predicted, labels).item() * 100}
