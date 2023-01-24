#%% LIBRARIES

import warnings

from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% LOOP TRAIN FUNCTION

def train_loop(model, data_loader, criterion, optimizer, scaler):
    loss_tr, num_correct, num_samples = 0, 0, 0
    
    y_tg = []
    y_sm = [] # Softmax prediction
    y_id = []
    confs = []

    model = model.train()
    for i, data in enumerate(data_loader):
        inputs, labels, idx = data
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
        loss_tr += loss.item()
        _, preds = torch.max(outputs.data, 1)
        num_correct += (preds == labels).sum().item()
        num_samples += labels.size(0)

        # Calculate confideces interval
        y_tg.extend(tg.item() for tg in labels)
        y_sm.extend(sm.cpu().detach().numpy() for sm in F.softmax(outputs, dim = 1))
        y_id.extend(id.item() for id in idx)

        del inputs
        del labels
        del outputs
        del loss
        del preds

    acc = 100*num_correct/num_samples
    loss_tr = loss_tr/num_samples

    for i, sm in enumerate(y_sm):
        confs.append(sm[y_tg[i]])

    return acc, loss_tr, confs, y_id

def evaluate_loop(model, data_loader, criterion):
    loss_v, num_correct, num_samples = 0, 0, 0

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
        loss_v += loss.item()
        _, preds = torch.max(outputs.data, 1)
        num_correct += (preds == labels).sum().item()
        num_samples += labels.size(0)

        # Adds the prediction and label information in two lists that will be used to create the confusion matrix
        y_pred.extend(pred.item() for pred in preds)
        y_tg.extend(y.item() for y in labels)

    acc = 100*num_correct/num_samples
    loss_v = loss_v/num_samples

    # Main diagonal of confusion matrix
    cf_matrix_abs = confusion_matrix(y_tg, y_pred)
    cf_matrix_rel = cf_matrix_abs/cf_matrix_abs.sum(axis = 1)[:, np.newaxis]
    main_diagonal_abs = cf_matrix_abs.diagonal()
    main_diagonal_abs = [m_d for m_d in main_diagonal_abs]
    main_diagonal_rel = cf_matrix_rel.diagonal()
    main_diagonal_rel = [m_d for m_d in main_diagonal_rel]

    return acc, loss_v, main_diagonal_abs, main_diagonal_rel