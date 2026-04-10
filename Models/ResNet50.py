# Does not work yet
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torchvision import models, transforms

class ResNet50:
    NUM_EPOCHS = 10
    TRAIN_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/train/"
    TEST_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test/"

    def data(self):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])

        train_data = torchvision.datasets.ImageFolder(root=self.TRAIN_PATH, transform=transform)
        test_data = torchvision.datasets.ImageFolder(root=self.TEST_PATH, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

        return train_data, test_data, train_loader, test_loader

    def transfer_learning(self):
        train_data, test_data, train_loader, test_loader = self.data()
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_data.classes))
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        model.train()

        for epoch in range(self.NUM_EPOCHS):
            train_loss = 0
            for images, labels in train_loader:

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        torch.save(model.state_dict(), '/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/resnet50.pth')
        print("Model successfully saved!")

    def testing(self):
        train_data, test_data, train_loader, test_loader = self.data()
        model = torch.load_state_dict('/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/resnet50.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Test Accuracy:", 100 * correct / total)

# The following code statements should perform the training, save the model, load the model, and test its accuracy
resnet50 = ResNet50()
resnet50.transfer_learning()
#resnet50.testing()

# Main source code inspo + CNN_Model.py: https://medium.com/@engr.akhtar.awan/how-to-fine-tune-the-resnet-50-model-on-your-target-dataset-using-pytorch-187abdb9beeb
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
# https://www.youtube.com/watch?v=CtzfbUwrYGI
# Gemini - step-by-step on how to implement models in pytorch and utilize transfer learning for maximum results