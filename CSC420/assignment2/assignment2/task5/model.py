import torch
import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_labels):
    # Load ResNet-18 
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final layer 
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_labels)

    return resnet18

def get_resnet34(num_labels):
    # Load ResNet-34 
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Modify the final layer 
    resnet34.fc = nn.Linear(resnet34.fc.in_features, num_labels)

    return resnet34

def get_resnext32(num_labels):
    # Load ResNeXt-32 with pre-trained weights if specified
    resnext32 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    
    # Modify the final layer
    resnext32.fc = nn.Linear(resnext32.fc.in_features, num_labels)
    
    return resnext32

def validation_epoch_end(outputs, stage):
    batch_losses = [x[f'{stage}_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x[f'{stage}_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {f'{stage}_loss':epoch_loss.item(), f'{stage}_acc':epoch_acc.item()}
    
# print result end epoch
def epoch_end(epoch, result):
    print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"], result["test_loss"], result["test_acc"]))
