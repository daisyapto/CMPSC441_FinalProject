import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.fc1 = nn.Linear(25600, 2)
        self.forwardCall = 0

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        # Programmatically calculate FC input size
        #print(x.size())
        x = self.fc1(x)
        #print("Forward Call #: ", self.forwardCall)
        self.forwardCall += 1
        return x

####### Training process #######
"""
NUM_EPOCHS = 10
TRAIN_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/train/"
TEST_PATH = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test/"

transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224))
        ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

network = NN()
mod = BrainCNN()

crit = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    train_loss = 0
    for images, labels in train_loader:

        optimizer.zero_grad()
        outputs = network(images)
        loss = crit(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {train_loss}")

torch.save(network.state_dict(), '/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth')
print("Model successfully saved!")"""


###### Testing ######
"""
test_data = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

state = torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/CNN_Model2.pth")
network.load_state_dict(state)
network.eval()

state2 = torch.load("/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Saved_Models/brain_cnn.pth")
mod.load_state_dict(state2)
mod.eval()

image_path = "/Users/daisyaptovska/Desktop/CMPSC441_FinalProject/Data/test/Brain_Tumor/Cancer (1).jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    outputs = network(image)
    _, predicted = torch.max(outputs, 1)

classes = ["brain_tumor", "healthy"]

print(classes[predicted.item()], round(_.item() * 100, 2), "%")"""
"""
total = 0
correct = 0

for images, labels in test_loader:
    outputs = network(images)
    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Test Accuracy:", 100 * correct / total)

total = 0
correct = 0

for images, labels in test_loader:
    outputs = network(images)
    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Test Accuracy:", 100 * correct / total)"""