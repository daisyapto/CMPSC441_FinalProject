import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

def test_model(model, test_loader, test_loader2=None):
    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    model.eval()

    with torch.no_grad():
        if test_loader2 is None:
            for images, labels in test_loader:
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        if test_loader2 is not None:
            for (images, labels), (images2, labels2) in zip(test_loader, test_loader2):
                outputs = model(images, images2)
            _, predicted = torch.max(outputs, 1)

    print(f"{model.__class__.__name__} Test Accuracy: ", accuracy(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Precision: ", precision(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test Recall: ", recall(predicted, labels).item() * 100, "%")
    print(f"{model.__class__.__name__} Test F1 Score: ", f1(predicted, labels).item() * 100, "%")