import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import DogBreedDataset, TrainDogBreedDataset
from utils import split_data, plot_acc, load_joint_dataset 
from model import get_resnet18, get_resnet34, get_resnext32
from train import fit_one_cycle

torch.manual_seed(42)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Alternative if using preprocessed dir
    # do not call load_join_dataset if using this
    # merged_dataset =ImageFolder('../task5_data/')
    
    dbiDataset = ImageFolder('../data/DBIsubset/')
    sddDataset = ImageFolder('../data/SDDsubset/')

    merged_dataset = load_joint_dataset(dbiDataset, sddDataset)

    # # Split data
    train_ds, val_ds, test_ds = split_data(merged_dataset, test_pct=0.3, val_pct=0.1)

    batch_size = 64

    # Create datasets
    train_dataset = TrainDogBreedDataset(train_ds)
    val_dataset = DogBreedDataset(val_ds)
    test_dataset = DogBreedDataset(test_ds)

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)

    # Here you can pick and run any model

    # Create Model
    # model = get_resnet18(num_labels=2).to(device)
    model = get_resnet34(num_labels=2).to(device)
    # model = get_resnext32(num_labels=2).to(device)

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


if __name__ =="__main__":
    main()