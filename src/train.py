import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F  # Add this line

from tqdm import tqdm
from model import SimpleMNISTNet

# Global lists for tracking metrics
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        loss.backward()
        optimizer.step()

        # Update metrics
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        running_loss += loss.item()

        # Update progress bar
        pbar.set_description(
            desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}'
        )

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100. * correct / processed

    train_losses.append(epoch_loss)
    train_acc.append(epoch_accuracy)

    return epoch_loss, epoch_accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)  # Calculate accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))  # Use the calculated accuracy

    test_acc.append(accuracy)  # Append accuracy to the list
    return accuracy  # Return the accuracy value

def validate_model(model, device, train_loader):
    # Check the number of parameters
    num_params = model.count_parameters()
    print(f"Number of parameters: {num_params}")

    # Validate only parameter count
    if num_params < 8000:
        print("Model validation successful: Parameters within limit")
        return True
    else:
        raise ValueError(
            f"Model validation failed: Parameters={num_params} (limit: 8000)"
        )

def train_and_test():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    network = SimpleMNISTNet().to(device)
    network.print_model_summary()

    # Calculate dataset statistics
    initial_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=initial_transform
    )
    mean, std = calculate_dataset_statistics(temp_dataset)
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    transform_train = transforms.Compose([
        transforms.RandomRotation((-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    # Datasets and Loaders
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Validate the model
    validate_model(network, device, train_loader)

    # Optimizer and Scheduler
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    # Training Loop
    best_accuracy = 0

    for epoch in range(15):
        print(f"\nEpoch {epoch+1}/15")

        # Training phase
        train_loss, train_accuracy = train(network, device, train_loader, optimizer, epoch)

        # Testing phase
        test_accuracy = test(network, device, test_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Testing  - Accuracy: {test_accuracy:.2f}%")

        scheduler.step()

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"Best Test Accuracy: {best_accuracy:.2f}%")
            torch.save(network.state_dict(), 'best_model.pth')

    print("\nTraining completed!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

def calculate_dataset_statistics(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=4
    )

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc="Calculating dataset statistics"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.item(), std.item()

if __name__ == "__main__":
    train_and_test()
