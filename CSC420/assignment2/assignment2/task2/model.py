import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    # # training step
    # def training_step(self, batch):
    #     img, targets = batch
    #     out = self(img)
    #     loss = F.nll_loss(out, targets)
    #     return loss
    
    # # validation step
    # def validation_step(self, batch):
    #     img, targets = batch
    #     out = self(img)
    #     loss = F.nll_loss(out, targets)
    #     acc = accuracy(out, targets)
    #     return {'val_acc':acc.detach(), 'val_loss':loss.detach()}
    
    # validation epoch end
    def validation_epoch_end(self, outputs, stage):
        batch_losses = [x[f'{stage}_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x[f'{stage}_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {f'{stage}_loss':epoch_loss.item(), f'{stage}_acc':epoch_acc.item()}
        
    # print result end epoch
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(epoch, result["train_loss"], result["test_loss"], result["test_acc"]))

class DogBreedClassificationCNN(ImageClassificationBase):
    def __init__(self, model_number, dropout):
        super().__init__()
        if model_number ==1:
            self.network = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=1, padding=0),   # 16*222*222
                nn.ReLU(),  
                nn.BatchNorm2d(num_features=16),  
                nn.Conv2d(16, 16, 3, stride=1, padding=0),  # 16*220*220
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                         # 16*110*110   
                
                nn.Conv2d(16, 8, 3, stride=1, padding=0),   # 8*108*108
                nn.ReLU(), 
                nn.BatchNorm2d(num_features=8),            
                nn.Conv2d(8, 8, 3, stride=1, padding=0),    # 8*106*106

                nn.ReLU(), 
                nn.MaxPool2d(2,2),                          # 8*53*53
                nn.Dropout(dropout),
            
                nn.Flatten(),                               # 8*53*53 = 26912
                nn.Linear(8*53*53, 512),                     # 32
                nn.ReLU(), 
                nn.Dropout(dropout),
                nn.Linear(512, 7),                          # 7 classes    
            )
        if model_number ==2:
            self.network = nn.Sequential(
                # MODEL 2
                nn.Conv2d(3, 16, 5, stride=1, padding=0),   # 16*252*252
                nn.ReLU(),  
                nn.BatchNorm2d(num_features=16),  
                nn.Conv2d(16, 16, 5, stride=1, padding=0),  # 16*248*248
                nn.ReLU(),
                nn.MaxPool2d(2, 2),                         # 16*124*124   
                
                nn.Conv2d(16, 32, 3, stride=1, padding=0),   # 8*122*122
                nn.ReLU(), 
                nn.BatchNorm2d(num_features=32),            
                nn.Conv2d(32, 32, 3, stride=1, padding=0),    # 8*120*120

                nn.ReLU(), 
                nn.MaxPool2d(2,2),                          # 8*60*60

                nn.Conv2d(32, 16, 3, stride=1, padding=0),   # 16*58*58
                nn.ReLU(), 
                nn.BatchNorm2d(num_features=16),            
                nn.Conv2d(16, 8, 3, stride=1, padding=0),    # 16*56*56

                nn.Flatten(),                               # 16*56*56 
                nn.Linear(8*56*56, 1024),                     
                nn.ReLU(),            
                nn.Dropout(dropout),

                nn.Linear(1024, 512),                     
                nn.Dropout(dropout),

                nn.ReLU(),
                nn.Linear(512, 7)                     
            )
    
    def forward(self, xb):
        return self.network(xb)
    
