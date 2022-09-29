#%% LIBRARIES

import sys
import warnings

from data.sampling_data import get_dataloaders
from torch.cuda.amp import GradScaler
from utils.checkpoint import save
from utils.loops import train_loop
from utils.setup_hyperparameters import setup_hyperparameters
from utils.setup_model import setup_model

import time
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_summary(device = None, abbreviated = False)

#%% TRAIN FUNCTION

def train(model, logger, hps):
    # Hyper-parameters
    name = hps["name"]
    start_epoch = hps["start_epoch"]
    num_epochs = hps["num_epochs"]
    learning_rate = hps["learning_rate"]
    weight_decay = hps["weight_decay"]
    k_fold = hps["k_fold"]
    
    scaler = GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = .9, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for k in range(1, k_fold+1):
        train_loader, _, _ = get_dataloaders(hps = hps, k = k)
    
        model = model.to(device)
        
        print("Training {} on {}".format(name, device))
        print(model)
    
        for epoch in range(start_epoch, num_epochs):
            t_start = time.time() # Time measure
            
            acc_tr, loss_tr = train_loop(model, train_loader, criterion, optimizer, scaler)
            logger.acc_train.append(acc_tr)
            logger.loss_train.append(loss_tr)
            
            print("Epoch: {} | "
                  "Train Acc.: {:.4f}% | "
                  "Train Loss: {:.4f}  | "
                  "Time: {:.2f}s".format(epoch+1,
                                         acc_tr,
                                         loss_tr,
                                         time.time()-t_start))
    
            #acc_v, loss_v = evaluate(model, train_loader, criterion)
            #logger.acc_val.append(acc_v)
            #logger.loss_val.append(loss_v)
            
            #print("Epoch: {} | "
            #      "Train Acc.: {:.4f}% | "
            #      "Val. Acc.: {:.4f}% | "
            #      "Time: {:.2f}s".format(epoch+1,
            #                             acc_tr, acc_v,
            #                             time.time()-t_start))

        save(model, logger, hps, k)
        logger.save_plt(hps, k)

#%% MAIN

if __name__ == "__main__":
    hps = setup_hyperparameters(sys.argv[1:]) # Import parameters
    
    logger, model = setup_model(hps) # Import model

    train(model, logger, hps) # Training