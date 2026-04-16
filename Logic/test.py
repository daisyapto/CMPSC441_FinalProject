import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

def test_model(model, test_loader):
    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if model.__class__.__name__ != "Ensemble":
                outputs = model(images)
            else:
                outputs = model(images, images)
            _, predicted = torch.max(outputs, 1)

    print(f"{model.__class__.__name__} Test Accuracy: ", accuracy(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Precision: ", precision(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Recall: ", recall(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test F1 Score: ", f1(predicted, labels).item() * 100, "%")