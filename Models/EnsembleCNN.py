import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from torchvision import transforms
import torchvision
from Logic.test import test_model

# https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/2

class Ensemble(nn.Module):
    def __init__(self, CNN1, CNN2):
        super(Ensemble, self).__init__()
        self.CNN1 = CNN1
        self.CNN2 = CNN2
        self.classifier = nn.Linear(43264,12544)

    def forward(self, x1, x2):
        x1 = self.CNN1(x1)
        x2 = self.CNN2(x2)
        x = torch.cat((x1, x2), 1)
        print(x)
        x = self.classifier(F.relu(x))
        return x

# Testing
TEST_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test/"

transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])

test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

CNN1 = BrainCNN()
CNN2 = NN()

CNN1.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth"))
CNN2.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth"))

model = Ensemble(CNN1, CNN2)
test_model(model, test_loader)

