#%% LIBRARIES

import warnings

from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast

import torch

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% LOOP TRAIN FUNCTION

def train_loop(model, data_loader, criterion, optimizer, scaler, num_outputs):
    model = model.train()

    num_loss, num_correct, num_samples = 0, 0, 0
    
    # Inputs of the confusion matrix
    y_pred = []
    y_tg = []

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
        
        # Adds the prediction and label information in two lists that will be used to create the confusion matrix
        y_pred.extend(pred.item() for pred in preds)
        y_tg.extend(y.item() for y in labels)

        del inputs
        del labels
        del outputs
        del loss
        del preds

    acc = 100*num_correct/num_samples
    loss = num_loss/num_samples
    
    missed_targets = []
    
    # Missed target
    if len(set(y_tg)) != num_outputs:
        y_s = list(set(y_tg))
        targets = [c for c in range(0, num_outputs)]
        
        for tg in targets:
            if tg not in y_s:
                missed_targets.append(tg)
    
    # Confusion matrix
    cf_matrix = confusion_matrix(y_tg, y_pred)

    return acc, loss, cf_matrix, missed_targets

def evaluate(model, data_loader, criterion):
    model = model.eval()

    num_loss, num_correct, num_samples = 0, 0, 0

    # Inputs of the confusion matrix
    y_pred = []
    y_tg = []
    
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

        # Adds the prediction and label information in two lists that will be used to create the confusion matrix
        y_pred.extend(pred.item() for pred in preds)
        y_tg.extend(y.item() for y in labels)

    acc = 100*num_correct/num_samples
    loss = num_loss/num_samples

    # Confusion matrix
    cf_matrix = confusion_matrix(y_tg, y_pred)

    return acc, loss, cf_matrix