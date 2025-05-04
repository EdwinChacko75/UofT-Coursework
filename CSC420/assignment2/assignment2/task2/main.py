import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import DogBreedDataset, TrainDogBreedDataset
from utils import split_data, plot_acc
from model import DogBreedClassificationCNN
from train import fit_one_cycle


torch.manual_seed(42)

def main(model_number=None,dropout=None):
    dataset = ImageFolder('../data/DBIsubset/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### THIS IS WERE WE SPECIFY MODEL AND DROPOUT
    # Models as defined in "./task2/model.py"
    model_number = 2
    dropout = 0.3
    batch_size = 64
    ### END 

    # Split data
    train_ds, val_ds, test_ds = split_data(dataset, test_pct=0.3, val_pct=0.1)

    # Create datasets
    train_dataset = TrainDogBreedDataset(train_ds, model_number)
    val_dataset = DogBreedDataset(val_ds, model_number)
    test_dataset = DogBreedDataset(test_ds, model_number)

    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)

    # Create Model
    model = DogBreedClassificationCNN(model_number, dropout).to(device)
    # print(sum(p.numel() for p in model.parameters()))

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
    plot_acc(history, f'./plots/m{model_number}_dropout_{dropout}.png', "Test")


if __name__ =="__main__":
    ### BELOW IS CODE TO GENERATE PLOTS
    # Set line 43 (epochs) to 10 if running this
    # mns = [1,2]
    # dps = [0,0.1,0.3,0.5]
    # for mn in mns:
    #     for dp in dps:
    #         print(f'running model {mn} with drpopout {dp}')
    #         main(mn, dp)
    ### END CODE TO GENERATE PLOTS

    ### BELOW IS CODE FOR A SINGLE RUN
    # Note: must specify lines 20, 21
    main()