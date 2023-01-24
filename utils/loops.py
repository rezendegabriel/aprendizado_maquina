#%% LIBRARIES

import warnings

from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast

import numpy as np
import torch

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% LOOP TRAIN FUNCTION

def train_loop(model, data_loader, criterion, optimizer, scaler):
    num_loss, num_correct, num_samples = 0, 0, 0
    
    model = model.train()
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

def evaluate_loop(model, data_loader, criterion):
    num_loss, num_correct, num_samples = 0, 0, 0

    # Inputs of the confusion matrix
    y_pred = []
    y_tg = []

    model = model.eval()
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

    # Main diagonal of confusion matrix
    cf_matrix = confusion_matrix(y_tg, y_pred)
    cf_matrix = cf_matrix/cf_matrix.sum(axis = 1)[:, np.newaxis]
    main_diagonal = cf_matrix.diagonal()
    main_diagonal = [m_d for m_d in main_diagonal]

    return acc, loss, main_diagonal