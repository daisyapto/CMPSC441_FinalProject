from Models.CNN_Model import BrainCNN
from Models.CNN_Model2 import NN
from Models.EnsembleCNN import Ensemble
from Logic.test import test_model
from Logic.predict import predict_image
from Manual_Test import manual_test
import torch
import torchvision
from torchvision import transforms

def main():
    NUM_EPOCHS = 10
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
    test_data2 = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform2)
    test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=64, shuffle=True)
    test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=64, shuffle=True)

    CNN1 = BrainCNN()
    CNN2 = NN()

    state1 = torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth")
    CNN1.load_state_dict(state1)
    CNN1.eval()

    state2 = torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth")
    CNN2.load_state_dict(state2)
    CNN2.eval()

    ensemble = Ensemble(CNN1, CNN2)

    test_model(CNN1, test_loader1)
    test_model(CNN2, test_loader2)
    test_model(ensemble, test_loader1, test_loader2)

    more = int(input("\n0) Exit\n1) Enter an image to classify\n>>>"))
    while more == 1:
        model_num = int(input("\n1) CNN Model 1\n2) CNN Model 2\n3) Ensemble (CNN Model 1 + CNN Model 2)\n>>>"))
        manual_test(model_num)
        more = int(input("\n0) Exit\n1) Enter another image to classify\n>>>"))

if __name__ == '__main__':
    main()