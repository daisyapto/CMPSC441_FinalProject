import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/2

class Ensemble(nn.Module):
    def __init__(self, CNN1, CNN2):
        super(Ensemble, self).__init__()
        self.CNN1 = CNN1
        self.CNN2 = CNN2
        self.classifier = nn.Linear(4,2)

    def forward(self, x1, x2):
        x1 = F.softmax(self.CNN1(x1))
        x2 = F.softmax(self.CNN2(x2))
        x = (x1 * 0.7 + x2 * .3)/2 # Fixed proper concatenation; also add weight to better performing model
        #print(x)
        #x = self.classifier(F.relu(x))
        return x

# Testing
"""
TEST_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test/"

transform1 = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((128, 128))
        ])
transform2 = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])

test_data1 = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform1)
test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=64, shuffle=True)
test_data2 = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform2)
test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=64, shuffle=True)

CNN1 = BrainCNN()
CNN2 = NN()

CNN1.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth"))
CNN2.load_state_dict(torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth"))

model = Ensemble(CNN1, CNN2)
test_model(model, test_loader1, test_loader2)
"""
