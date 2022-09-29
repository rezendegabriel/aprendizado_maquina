#%% LIBRARIES

import warnings

import torch

from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% LOOP TRAIN FUNCTION

def train_loop(model, data_loader, criterion, optimizer, scaler):
    model = model.train()

    num_loss, num_correct, num_samples = 0, 0, 0
    
    for i, data in enumerate(data_loader):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs = inputs.float()

        with autocast():
            # Forward
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward() # Backward
    
        # Optimize
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Calculate performance metrics
        num_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        num_correct += (preds == labels).sum().item()
        num_samples += labels.size(0)

    acc = 100*num_correct/num_samples
    loss = num_loss/num_samples

    return acc, loss

def evaluate(model, data_loader, criterion):
    model = model.eval()

    num_loss, num_correct, num_samples = 0, 0, 0
    
    for data in data_loader:
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        inputs = inputs.float()

        # Forward
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)

        # Calculate performance metrics
        num_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        num_correct += (preds == labels).sum().item()
        num_samples += labels.size(0)

    acc = 100*num_correct/num_samples
    loss = num_loss/num_samples

    return acc, loss