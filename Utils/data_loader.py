from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder("Data/train", transform=transform)
    test_dataset = datasets.ImageFolder("Data/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader