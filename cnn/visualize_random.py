import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

from main import CNN

def run() -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = CNN().to(device)
    model.load_state_dict(torch.load("mnistmodel.pth", weights_only=True))

    test_idx = random.randint(0, 10_000)

    model.eval()
    inputs, label = test_dataset[test_idx][0], test_dataset[test_idx][1]
    with torch.no_grad():
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(0)  
        pred = model(inputs)
        img = inputs.squeeze().cpu() * 0.3081 + 0.1307  
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f'Predicted: "{pred.argmax(1).item()}", Actual: "{label}"')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run()