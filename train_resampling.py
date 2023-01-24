#%% LIBRARIES

import sys
import warnings

from data.sampling_data import get_dataloaders
from torch.cuda.amp import GradScaler
from utils.checkpoint import save
from utils.loops_resampling import train_loop, evaluate_loop
from utils.setup_hyperparameters import setup_hyperparameters
from utils.setup_model import setup_model

import time
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_summary(device = None, abbreviated = False)

#%% TRAIN FUNCTION

def train(model, logger, hps, train_loader, val_loader, optimizer, criterion, current_epoch):
    # Hyper-parameters
    num_epochs = hps["num_epochs"]
    z = hps["z"]
    verbose = hps["verbose"]
    
    scaler = GradScaler()

    if current_epoch+z > num_epochs:
        z = num_epochs-current_epoch
    
    for epoch in range(current_epoch, current_epoch+z):
        t_start = time.time()

        acc_tr, loss_tr, confs, inputs_idx = train_loop(model, train_loader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)
        
        acc_v, loss_v, main_diagonal_abs, main_diagonal_rel = evaluate_loop(model, val_loader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        if verbose == 0:
            print("Epoch: {}".format(epoch+1))
        elif verbose == 1:
            print("Epoch: {} | Train Acc.: {:.4f}% | Val. Acc.: {:.4f}%".format(epoch+1, acc_tr, acc_v))
        elif verbose == 2:
            print("Epoch: {} | Train Acc.: {:.2f}% | Val. Acc.: {:.2f}% | "
                  "0: {:.2f}% | 1: {:.2f}%".format(epoch+1, acc_tr, acc_v,
                  100*main_diagonal_rel[0], 100*main_diagonal_rel[1]))
        else: # 3
            print("Epoch: {} | Train Acc.: {:.2f}% | Val. Acc.: {:.2f}% | "
                  "0: {:.2f}% | 1: {:.2f}% | Time: {:.2f}s".format(epoch+1, acc_tr, acc_v,
                                                                   100*main_diagonal_rel[0], 100*main_diagonal_rel[1],
                                                                   time.time()-t_start))

    return z, main_diagonal_abs, confs, inputs_idx

#%% MAIN

if __name__ == "__main__":
    hps = setup_hyperparameters(sys.argv[1:])
    
    # Hyper-parameters
    name = hps["name"]
    num_epochs = hps["num_epochs"]
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
        
        train_loader, val_loader, _, class_weights, sampler_weights = get_dataloaders(hps = hps, k = k, resampling = True)
        logger.class_weights.append(class_weights)
        logger.sampler_weights.append(sampler_weights)

        model = model.to(device)
        
        print("Training {} on {}".format(name, device))
        print(model)
    
        z, main_diagonal_abs, confs, inputs_idx = train(model, logger, hps, train_loader, val_loader, optimizer, criterion, current_epoch)
        current_epoch += z
    
        for n_sampling in range(1, s):
            train_loader, val_loader, _, class_weights, sampler_weights = get_dataloaders(hps = hps,
                                                                                          k = k,
                                                                                          resampling = True,
                                                                                          n_sampling = n_sampling,
                                                                                          main_diagonal = main_diagonal_abs,
                                                                                          confs = confs,
                                                                                          inputs_idx = inputs_idx)
            logger.class_weights.append(class_weights)
            logger.sampler_weights.append(sampler_weights)
            
            z, main_diagonal_abs, confs, inputs_idx = train(model, logger, hps, train_loader, val_loader, optimizer, criterion, current_epoch)
            current_epoch += z
    
        save(model, logger, hps, k)
        logger.save_plt(hps, k)
        
        del model
        del criterion
        del optimizer