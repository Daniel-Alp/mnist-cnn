import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x) -> torch.tensor:
        return self.layers(x)

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> None:
    for inputs, label in dataloader:
        pred = model(inputs)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, label in dataloader:
            inputs, label = inputs.to(device), label.to(device)
            pred = model(inputs)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run() -> None:
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform)

    batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "mnistmodel.pth")

if __name__ == "__main__":
    run()