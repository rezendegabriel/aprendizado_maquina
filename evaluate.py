#%% LIBRARIES

import sys
import warnings

from data.sampling_data import get_dataloaders
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from utils.setup_hyperparameters import setup_hyperparameters
from utils.setup_model import setup_model

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sb
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = torch.device("cpu")

#%% EVALUATE FUNCTION

def evaluate(hps):
    # Hyper-parameters
    model_save_dir = hps["model_save_dir"]
    k_fold = hps["k_fold"]
    
    criterion = nn.CrossEntropyLoss() # Loss function
    
    # Wrongly predicted samples folder
    wrong_pred_ids_folder = os.path.join(model_save_dir, "wrong_pred_ids")

    if os.path.exists(wrong_pred_ids_folder):
        shutil.rmtree(wrong_pred_ids_folder)
        os.mkdir(wrong_pred_ids_folder)
    else:
        os.mkdir(wrong_pred_ids_folder)
        
    num_samples, num_correct, num_loss = 0, 0, 0

    y_pred = []
    y_tg = []

    w_id = []
    w_pred = []
    w_label = []
    
    # Cross validation
    for k in range(1, k_fold+1):
        _, _, test_loader, _, _ = get_dataloaders(hps = hps, k = k) # Get data with no augmentation
        
        # Build model
        _, model = setup_model(hps, k = k)
        model = model.to(device)
    
        print(model)

        model = model.eval()
        for data in test_loader:
            inputs, labels, id_inputs = data
            inputs, labels, id_inputs = inputs.to(device), labels.to(device), id_inputs.to(device)
            
            inputs = inputs.float()
    
            outputs = model(inputs) # Forward
            outputs = outputs.squeeze(1)
            outputs = outputs.squeeze(1)
            
            # Calculated loss
            loss = criterion(outputs, labels)
            num_loss += loss.item()
    
            preds = torch.argmax(F.softmax(outputs, dim = 1), 1) # Predictions
            
            num_correct += (preds == labels).sum() # Updates the number of correct predictions
    
            preds = preds.to("cpu")
            labels = labels.to("cpu")
            
            num_samples += labels.size(0) # Updates the number of samples
    
            y_pred.extend(pred.item() for pred in preds)
            y_tg.extend(tg.item() for tg in labels)
    
            # Store wrongly predicted samples
            w_batch_id = (preds != labels.view_as(preds)).nonzero()[:, 0]
            w_id.extend(i.item() for i in id_inputs[w_batch_id])
            w_label.extend(label.item() for label in labels[w_batch_id])
            w_pred.extend(pred.item() for pred in preds[w_batch_id])
            
    acc = 100*num_correct/num_samples
    loss = num_loss/num_samples

    print("--------------------------------------------------")
    print("Accuracy: %2.6f %%" % acc)
    print("Loss: %2.6f" % loss)
    print("Confusion Matrix:\n", confusion_matrix(y_tg, y_pred), "\n")

    wrong_dict = {"id": w_id, "label": w_label, "pred": w_pred}
    wrong_df = pd.DataFrame(wrong_dict)
    wrong_df.to_csv("{}/wrong_ids.csv".format(wrong_pred_ids_folder), index = False)

    dict_target_pred = {"target": y_tg, "prediction": y_pred}
    df_target_pred = pd.DataFrame(dict_target_pred)
    df_target_pred.to_csv("{}/target_pred.csv".format(model_save_dir), index = False)

    precision = precision_score(y_tg, y_pred, average = None)
    recall = recall_score(y_tg, y_pred, average = None)
    f1 = f1_score(y_tg, y_pred, average = None)
    f2 = fbeta_score(y_tg, y_pred, average = None, beta = 2)
    
    dict_metrics = {"precision": precision, "recall": recall, "f1": f1, "f2": f2}
    df_metrics = pd.DataFrame(dict_metrics)
    df_metrics.to_csv("{}/metrics.csv".format(model_save_dir), index = False)

    ploting_confusion_matrix(y_tg, y_pred, acc)

# Function for ploting the confusion matrix results
def ploting_confusion_matrix(y_tg, y_pred, acc):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]
    })
    
    classes_names = []
    
    for c in set(y_tg):
        classes_names.append(str(c))

    # Hyper-parameters
    model_save_dir = hps["model_save_dir"]

    cf_matrix_path = os.path.join(model_save_dir, "cf_matrix_ft.pdf")
    
    #confusion matrix with absolute values
    cf_matrix = confusion_matrix(y_tg, y_pred)
    group_counts = ["{0:.0f}".format(value) for value in cf_matrix.flatten()]
    
    #confusion matrix with relative values
    cf_matrix = 100*cf_matrix/cf_matrix.sum(axis = 1)[:, np.newaxis]
    group_percentages = ["{0:.2f}".format(value) for value in cf_matrix.flatten()]
    
    #confusion matrix with both absolute and relative values
    annot = [f"{v1}" + r'\%' + f"\n{v2}" for v1, v2 in zip(group_percentages, group_counts)]
    annot = np.asarray(annot).reshape(len(classes_names) , len(classes_names))
    
    #ploting
    fig, ax = plt.subplots(1, figsize = (30, 30))
    
    sb.set(rc = {"text.usetex": True, "font.family": "serif", "font.serif": ["Times"]}, font_scale = 5)
    ax = sb.heatmap(cf_matrix,
                    xticklabels = classes_names,
                    yticklabels = classes_names,
                    annot = annot,
                    fmt = "",
                    cmap = "Blues",
                    cbar = False)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length = 0)

    plt.xlabel("Acc.: {:.2f}".format(acc) + r'\%', labelpad = 30, fontsize = 60)

    plt.xticks(plt.xticks()[0], [label._text for label in plt.xticks()[1]], fontsize = 60)
    
    plt.yticks(plt.yticks()[0], [label._text for label in plt.yticks()[1]], fontsize = 60, rotation = 90)
    
    plt.savefig(cf_matrix_path, bbox_inches = "tight", pad_inches = 0)

if __name__ == "__main__":
    hps = setup_hyperparameters(sys.argv[1:]) # Import parameters
    
    evaluate(hps)