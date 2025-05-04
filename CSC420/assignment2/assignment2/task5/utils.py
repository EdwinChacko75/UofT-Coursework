
from torch.utils.data import random_split, ConcatDataset
import matplotlib.pyplot as plt
import torch

def load_joint_dataset(dbiDataset, sddDataset):
    # Manually change labels: Assign DBI -> 0, SDD -> 1
    dbiDataset.targets = [0] * len(dbiDataset)  
    sddDataset.targets = [1] * len(sddDataset)  
    dbiDataset.samples = [(path, 0) for path, _ in dbiDataset.samples]
    sddDataset.samples = [(path, 1) for path, _ in sddDataset.samples]

    # Merge datasets
    merged_dataset = ConcatDataset([dbiDataset, sddDataset])
    return merged_dataset

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def split_data(dataset, test_pct, val_pct):
    test_size = int(len(dataset)*test_pct)
    dataset_size = len(dataset) - test_size

    val_size = int(dataset_size*val_pct)
    train_size = dataset_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])

def plot_acc(history, img_save_path, stage='Validation'):
    test_acc = [ep[f'test_acc'] for ep in history]
    val_acc = [ep[f'val_acc'] for ep in history]
    train_acc = [ep['train_acc'] for ep in history]

    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, val_acc, label=f"Validation Accuracy", marker='s')
    plt.plot(epochs, test_acc, label=f"Test Accuracy", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training, Validation, and Test Accuracies")
    plt.legend()
    plt.grid(True)

    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    
