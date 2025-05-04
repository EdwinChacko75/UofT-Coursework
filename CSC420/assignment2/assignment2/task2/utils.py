
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def split_data(dataset, test_pct, val_pct):
    test_pct = 0.3
    test_size = int(len(dataset)*test_pct)
    dataset_size = len(dataset) - test_size

    val_pct = 0.1
    val_size = int(dataset_size*val_pct)
    train_size = dataset_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])

def plot_acc(history, img_save_path, stage='Validation'):
    name = 'val' if stage == "Validation" else 'test'
    val_acc = [ep[f'{name}_acc'] for ep in history]
    train_acc = [ep['train_acc'] for ep in history]

    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, val_acc, label=f"{stage} Accuracy", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training vs {stage} Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    
