import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import DogBreedDataset, TrainDogBreedDataset
from utils import split_data, plot_acc
from model import get_resnet18, get_resnet34, get_resnext32
from train import fit_one_cycle, evaluate

torch.manual_seed(42)

def main():
    dataset = ImageFolder('../data/DBIsubset/')
    sddDataset = ImageFolder('../data/SDDsubset/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Split data
    train_ds, val_ds, test_ds = split_data(dataset, test_pct=0.3, val_pct=0.1)

    batch_size = 64

    # Create datasets
    train_dataset = TrainDogBreedDataset(train_ds)
    val_dataset = DogBreedDataset(val_ds)
    test_dataset = DogBreedDataset(test_ds)

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)


    # Create Model
    model = get_resnext32(num_labels=7).to(device)
    print(sum(p.numel() for p in model.parameters()))

    # Hyperparms
    num_epochs = 10
    opt_func = torch.optim.AdamW
    criterion = nn.CrossEntropyLoss()
    max_lr = 1e-4
    grad_clip = 0.1
    weight_decay = 1e-4

    # train loop
    history = fit_one_cycle(num_epochs, max_lr, model, train_dl, val_dl, test_dl, device, criterion, weight_decay, grad_clip, opt_func)
    
    # plot
    # plot_acc(history, f'./plots/{max_lr}_{num_epochs}.png')

    # evaluate on sdd dataset
    _, _, test_ds = split_data(sddDataset, test_pct=1, val_pct=0)
    
    sdd_dataset = DogBreedDataset(test_ds)
    sdd_loader = DataLoader(sdd_dataset, batch_size*2, num_workers=2, pin_memory=True)
    metrics = evaluate(model, sdd_loader,criterion, device, stage='sdd')

    print("SDD Dataset Metrics")
    print(f"Accuracy: {metrics['sdd_acc']}")
    print(f"Loss: {metrics['sdd_loss']}")

if __name__ =="__main__":
    main()