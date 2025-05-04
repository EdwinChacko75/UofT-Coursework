
import torch
from tqdm import tqdm
import torch.nn as nn
from utils import accuracy

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, test_loader, device, criterion, weight_decay=0, grad_clip=None, opt_func = torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # set up one cycle lr scheduler
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 3 epochs
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(epochs):
        
        # Training phase
        model.train()       
        train_losses = []
        train_accs=[]
        lrs = []
        for (img, tgt) in tqdm(train_loader):
            img, tgt = img.to(device), tgt.to(device)

            pred = model(img)
            loss = criterion(pred, tgt.long())

            train_losses.append(loss.item())
            train_accs.append(accuracy(pred, tgt))
            # calculates gradients
            loss.backward()
            
            # check gradient clipping 
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            # perform gradient descent and modifies the weights
            optimizer.step()

            # reset the gradients
            optimizer.zero_grad()
            
            # record and update lr
            lrs.append(get_lr(optimizer))
            
            # modifies the lr value
            # sched.step()
            
        # Validation phase
        result = evaluate(model, val_loader, criterion, device)
        sched.step(result['val_loss'])

        # test phase
        result = evaluate(model, test_loader, criterion, device, stage="test")
        result['train_loss'] = torch.tensor(train_losses).mean().item()
        result['train_acc'] = torch.tensor(train_accs).mean().item()
        result['lrs'] = lrs


        model.epoch_end(epoch, result)
        history.append(result)
        
        
    return history
        
    

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, stage='val'):
    model.eval()
    outputs = []
    for img, tgt in val_loader:
        img, tgt = img.to(device), tgt.to(device)

        pred = model(img)
        loss = criterion(pred, tgt.long())
        acc = accuracy(pred, tgt)

        outputs.append({f'{stage}_acc':acc, f'{stage}_loss':loss})
    return model.validation_epoch_end(outputs, stage)