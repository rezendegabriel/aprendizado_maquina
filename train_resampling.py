#%% LIBRARIES

import sys
import warnings

from data.sampling_data import get_dataloaders
from torch.cuda.amp import GradScaler
from utils.checkpoint import save
from utils.loops_resampling import train_loop
from utils.setup_hyperparameters import setup_hyperparameters
from utils.setup_model import setup_model

import time
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_summary(device = None, abbreviated = False)

#%% TRAIN FUNCTION

def train(model, logger, hps, train_loader, optimizer, criterion, current_epoch, num_outputs):
    # Hyper-parameters
    num_epochs = hps["num_epochs"]
    z = hps["z"]
    
    scaler = GradScaler()

    if current_epoch+z > num_epochs:
        z = num_epochs-current_epoch
    
    for epoch in range(current_epoch, current_epoch+z):
        t_start = time.time()

        acc_tr, loss_tr, cf_matrix, missed_targets = train_loop(model, train_loader, criterion, optimizer, scaler, num_outputs)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)
        
        print("Epoch: {} | "
              "Train Acc.: {:.4f}% | "
              "Train Loss: {:.4f}  | "
              "Time: {:.2f}s".format(epoch+1,
                                     acc_tr,
                                     loss_tr,
                                     time.time()-t_start))
        
        #acc_v, loss_v, cf_matrix = evaluate(net, val_loader, criterion)
        #logger.loss_val.append(loss_v)
        #logger.acc_val.append(acc_v)

        #print("Epoch: {} | "
        #      "Train Acc.: {:.4f}% | "
        #      "Val. Acc.: {:.4f}% | "
        #      "Time: {:.2f}s".format(epoch+1,
        #                             acc_tr, acc_v,
        #                             time.time()-t_start))

    return z, cf_matrix, missed_targets

#%% MAIN

if __name__ == "__main__":
    hps = setup_hyperparameters(sys.argv[1:])
    
    # Hyper-parameters
    name = hps["name"]
    num_epochs = hps["num_epochs"]
    num_outputs = hps["num_outputs"]
    learning_rate = hps["learning_rate"]
    weight_decay = hps["weight_decay"]
    z = hps["z"]
    k_fold = hps["k_fold"]
    
    s = int(num_epochs/z)

    if num_epochs%z != 0:
        s += 1

    for k in range(1, k_fold+1):
        current_epoch = hps["start_epoch"]
        
        logger, model = setup_model(hps)
        
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = .9, weight_decay = weight_decay)

        criterion = nn.CrossEntropyLoss()
        
        train_loader, _, weights = get_dataloaders(hps = hps, k = k, resampling = True)
        logger.weights.append(weights)

        model = model.to(device)
        
        print("Training {} on {}".format(name, device))
        print(model)
    
        z, cf_matrix, missed_targets = train(model, logger, hps, train_loader, optimizer, criterion, current_epoch, num_outputs)
        current_epoch += z
    
        for n_sampling in range(1, s):
            train_loader, _, weights = get_dataloaders(hps = hps, k = k, resampling = True, n_sampling = n_sampling, cf_matrix = cf_matrix, missed_targets = missed_targets)
            logger.weights.append(weights)
            
            z, cf_matrix, missed_targets = train(model, logger, hps, train_loader, optimizer, criterion, current_epoch, num_outputs)
            current_epoch += z
    
        save(model, logger, hps, k)
        logger.save_plt(hps, k)
        
        del model
        del criterion
        del optimizer