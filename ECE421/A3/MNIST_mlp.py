import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class CustomDataset(torchvision.datasets.FashionMNIST):
    """Reshapes image to single dimensional vector of dim 784"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img.reshape(-1), target

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 256) 
        self.l2 = nn.Linear(256, 10) 

        self.relu = nn.ReLU()
        
    def forward(self, x):
        l1 = self.relu(self.l1(x))
        l2 = self.l2(l1)
        return l2
        
def load_data():
    # Defining transformation
    transform = torchvision.transforms.ToTensor() 

    # Use custom dataset to load
    training_data = CustomDataset(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    validation_data = CustomDataset(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    return training_data, validation_data

def plot_metrics(training_losses, training_accs, val_losses, val_accs):
    epochs = range(1, len(training_losses) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.savefig("loss_vs_epochs.png", dpi=300)
    plt.show()

    # Plot training and validation accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_accs, label="Training Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()
    plt.savefig("accuracy_vs_epochs.png", dpi=300)
    plt.show()



def validate(model, criterion, val_loader, num_val_pts, device):
    # Validate on val set
    with torch.no_grad():
        model.eval() # Put model in eval mode
        num_correct = 0
        for x, y in val_loader:
            x, y = x.float().to(device), y.long().to(device)
            pred = model(x)

            loss = criterion(pred, y)

            _, pred_indices = torch.max(pred, dim=1)
            num_correct += torch.sum(pred_indices == y).item()
        acc = num_correct / num_val_pts
        print("Validation Accuracy:", round(acc, 4))

        model.train() # Put model back in train mode
    return loss.item(), acc

def main():
    # Define device and training params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    epochs = 20
    batch_size = 128
    learning_rate = 0.001

    # Load datasets and create dataloaders
    train, val = load_data()
    num_val_pts = len(val)
    num_train_pts = len(train)

    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val, batch_size, num_workers=1)

    # Create model on device
    model = MLP(784).to(device)

    # Define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Put model in training mode
    model.train()

    # Run training loop epoch times
    training_losses = []
    training_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        tqdm.write(f"Epoch {epoch + 1}/{epochs}")
        training_loss = []
        num_correct = 0
        # Training loop
        for x, y in tqdm(train_loader, unit="batch"):
            x, y = x.float().to(device), y.long().to(device)
            optimizer.zero_grad() 
            pred = model(x)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            _, pred_indices = torch.max(pred, dim=1)
            num_correct += torch.sum(pred_indices == y).item()
            training_loss.append(loss.item())

        # Compute training and val acc and loss
        train_loss = np.mean(training_loss)
        train_acc = num_correct / num_train_pts
        val_loss, val_acc = validate(model, criterion, val_loader, num_val_pts, device)

        # Store training and val acc and loss 
        training_losses.append(train_loss)
        training_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)      

        print(f"Epoch {epoch+1}. training loss: {train_loss:2f}, training acc: {train_acc:2f}, val loss: {val_loss:2f}, val acc: {val_acc:2f}"            )

    plot_metrics(training_losses, training_accs, val_losses, val_accs)
    torch.save(model.state_dict(), "model_smaller_AdamW.pth")


if __name__ == "__main__":
    main()


