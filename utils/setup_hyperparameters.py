#%% LIBRARIES

import os

#%% HYPER-PARAMETERS

hps = {
    "model": "", # Which model do you want to train
    "name": "",  # Whatever you want to name your run
    "model_save_dir": "",  # Where will checkpoints be stored (path created automatically using hps["name"])
    "dataset_type": "", # Binary or multi-class
    "dataset": "", # Dataset name
    "datset_path": "", # Path created automatically using hps["dataset_type"] and hps["dataset"]
    "start_epoch": "",
    "num_epochs": 300,
    "num_inputs": "", # Number of attributes of dataset
    "num_outputs": "", # Number of classes of dataset
    "layer_size": 4, # Number of neurons proportional to the number of inputs
    "drop": 0.2, # Drop rate of linear layers
    "restore_model": "", # Restore model for evaluate
    "batch_size": "", # Batchsize
    "num_workers": 0, # Number of workers of DataLoader function
    "learning_rate": 1e-3,
    "weight_decay": 1e-4, # Weight decay of optimizer
    "momentum": .9, # Momentum of optimizer
    "z": "", # Interval z of resampling that each class-based sampling occurs
    "k_fold": 5, # Cross validation
    "init_sampling": "", # Initial sampling ("stratified", "inverse-stratified" or "balanced")
}

possible_models = set(file_name.split(".")[0] for file_name in os.listdir("models"))
possible_dataset_type = set(dir_name for dir_name in os.listdir("data/datasets"))

#%% FUNCTION

def setup_hyperparameters(args):
    for arg in args:
        key, value = arg.split("=")
        
        if key not in hps:
            raise ValueError(key + " is not a valid hyper-parameter")
        else:
            hps[key] = value

    # Invalid model check
    if hps["model"] not in possible_models:
        raise ValueError("Invalid model.\nPossible ones include:\n - " + "\n - ".join(possible_models))
        
    # Invalid dataset_type check
    if hps["dataset_type"] not in possible_dataset_type:
        raise ValueError("Invalid dataset type.\nPossible ones include:\n - " + "\n - ".join(possible_dataset_type))
    else:
        possible_dataset = set(dir_name for dir_name in os.listdir("data/datasets/{}".format(hps["dataset_type"])))
        
    # Invalid parameter check
    try:
        hps["start_epoch"] = int(hps["start_epoch"])
        hps["num_epochs"] = int(hps["num_epochs"])
        hps["num_inputs"] = int(hps["num_inputs"])
        hps["num_outputs"] = int(hps["num_outputs"])
        hps["layer_size"] = int(hps["layer_size"])
        hps["drop"] = float(hps["drop"])
        hps["batch_size"] = int(hps["batch_size"])
        hps["num_workers"] = int(hps["num_workers"])
        hps["learning_rate"] = float(hps["learning_rate"])
        hps["weight_decay"] = float(hps["weight_decay"])
        hps["z"] = int(hps["z"])
        hps["k_fold"] = int(hps["k_fold"])
        
        # Invalid dataset check
        dataset = hps["dataset"] + "-" + str(hps["k_fold"]) + "-fold"
        if dataset not in possible_dataset:
            raise ValueError("Invalid dataset.\nPossible ones include:\n - " + "\n - ".join(possible_dataset))

        # Create checkpoint directory
        hps["model_save_dir"] = os.path.join(os.getcwd(), "checkpoints", hps["dataset_type"], hps["name"])

        if not os.path.exists(hps["model_save_dir"]):
            os.makedirs(hps["model_save_dir"])
            
        # Create dataset path
        hps["dataset_path"] = os.path.join(os.getcwd(), "data", "datasets", hps["dataset_type"], dataset)

    except Exception as e:
        raise ValueError("Invalid input parameters")
        print(e)

    return hps
