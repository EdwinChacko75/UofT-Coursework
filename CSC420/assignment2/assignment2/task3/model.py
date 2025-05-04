import torch
import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_labels):
    # Load ResNet-18 architecture
    resnet18 = models.resnet18(weights=None)

    # Modify the final layer 
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_labels)

    return resnet18

def validation_epoch_end(outputs, stage):
    batch_losses = [x[f'{stage}_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x[f'{stage}_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {f'{stage}_loss':epoch_loss.item(), f'{stage}_acc':epoch_acc.item()}
    
# print result end epoch
def epoch_end(epoch, result):
    print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(epoch, result["train_loss"], result["val_loss"], result["val_acc"], result["test_loss"], result["test_acc"]))
