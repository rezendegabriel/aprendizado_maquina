#%% LIBRARIES

from data.custom_dataset import CustomDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

#%% SAMPLING DATA

def sample_data(df, train = False):
    attributes = list(df.columns)
    column_class = attributes.pop(len(attributes)-1)
    
    data_array = np.zeros(shape = (len(df), 1, len(attributes)))
    data_label = np.array(list(map(int, df[column_class])))
    data_idx = np.zeros(shape = (len(df)))
    
    for i, row in enumerate(df.index):
        data = df.loc[row, attributes].to_numpy(dtype = float)
        data_array[i] = data
        data_idx[i] = i+1

    if train:
        skf = StratifiedKFold(n_splits = 5)
        for train_i, val_i in skf.split(data_array, data_label, data_idx):
            train_array, train_label, train_idx = data_array[train_i], data_label[train_i], data_idx[train_i]
            val_array, val_label, val_idx = data_array[val_i], data_label[val_i], data_idx[val_i]

        return train_array, train_label, train_idx, val_array, val_label, val_idx
    else:
        return data_array, data_label, data_idx

#%% TENSOR TRANSFORM

mean, std = 0.5, 0.5 # Min-Max normalization

transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean = (mean), std = (std))
                                     ])

#%% DATALOADERS

def get_dataloaders(hps, k, resampling = False, n_sampling = 0, main_diagonal = None, confs = None, inputs_idx = None):
    # Hyper-parameters
    dataset_path = hps["dataset_path"]
    dataset = hps["dataset"]
    k_fold = hps["k_fold"]
    batch_size = hps["batch_size"]
    num_workers = hps["num_workers"]
    init_sampling = hps["init_sampling"]
    adaptive_sampling = hps["adaptive_sampling"]
    val_set = hps["val_set"]
    
    # Load data
    df_train = pd.read_csv("{}/{}-{}-{}tra.csv".format(dataset_path, dataset, k_fold, k))
    df_test = pd.read_csv("{}/{}-{}-{}tst.csv".format(dataset_path, dataset, k_fold, k))
    
    # Sampling data
    if val_set == "train":
        x_train, y_train, id_train = sample_data(df_train)

        # Data tensor transform
        train = CustomDataset(x_train, y_train, id_train, transformation) 
        val = CustomDataset(x_train, y_train, id_train, transformation)
    else: # "val"
        x_train, y_train, id_train, x_val, y_val, id_val = sample_data(df_train, train = True)

        # Data tensor transform
        train = CustomDataset(x_train, y_train, id_train, transformation)
        val = CustomDataset(x_val, y_val, id_val, transformation)

    x_test, y_test, id_test = sample_data(df_test)
    
    test = CustomDataset(x_test, y_test, id_test, transformation) # Data tensor transform

    class_weights = [] # Class sampling distribution
    sampler_weights = [] # Sampler distribution

    if resampling:
        if n_sampling == 0:
            if init_sampling == "stratified":
                train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, shuffle = True, num_workers = num_workers)

                _, counts = np.unique(train.labels, return_counts = True)
                class_weights = [c/sum(counts) for c in counts] # Normalization
                sampler_weights = class_weights
            elif init_sampling == "inverse-stratified":
                _, counts = np.unique(train.labels, return_counts = True)
                class_weights = [sum(counts)/c for c in counts]
                class_weights = [w*w for w in class_weights]

                class_weights = [w/sum(class_weights) for w in class_weights] # Normalization
                    
                samples_weights = [class_weights[e] for e in train.labels]
                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

                # Sampler weights
                sampler_labels = []
                for s in list(sampler):
                    sampler_labels.append(train.labels[s])

                _, counts = np.unique(sampler_labels, return_counts = True)
                sampler_weights = [c/sum(counts) for c in counts] # Normalization

                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
            else: # balanced
                _, counts = np.unique(train.labels, return_counts = True)
                class_weights = [sum(counts)/c for c in counts]

                class_weights = [w/sum(class_weights) for w in class_weights] # Normalization
                    
                samples_weights = [class_weights[e] for e in train.labels]
                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

                # Sampler weights
                sampler_labels = []
                for s in list(sampler):
                    sampler_labels.append(train.labels[s])

                _, counts = np.unique(sampler_labels, return_counts = True)
                sampler_weights = [c/sum(counts) for c in counts] # Normalization

                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
        else: # Adaptive sampling
            if adaptive_sampling == "conf": # Confidence-based
                _, counts = np.unique(train.labels, return_counts = True)
                sampler_ponder = [sum(counts)/c for c in counts]

                _, counts = np.unique(val.labels, return_counts = True)
                counts = [c for c in counts]
                
                samples_weights = []

                # Normalization
                full_conv = 0
                for c in range(len(counts)):
                    w = main_diagonal[c]/counts[c]

                    if w == 1:
                        full_conv+=1
                    
                    class_weights.append(1-w)

                # Full convergence
                if full_conv == len(counts):
                    for i in range(len(class_weights)):
                        class_weights[i] = 1/full_conv # Balanced

                class_weights = [w*sampler_ponder[i] for i, w in enumerate(class_weights)] # Ponderation
                class_weights = [w/sum(class_weights) for w in class_weights] # Normalization

                confs_weights = [1-w for w in confs]

                for i, l in enumerate(train.labels):
                    idx = i+1
                    if idx in inputs_idx:
                        samples_weights.append(confs_weights[inputs_idx.index(idx)]*sampler_ponder[l])
                    else:
                        samples_weights.append(class_weights[l])

                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

                # Sampler weights
                sampler_labels = []
                for s in list(sampler):
                    sampler_labels.append(train.labels[s])

                _, counts = np.unique(sampler_labels, return_counts = True)
                counts = list(counts)

                # Missed target
                if len(counts) != 2:
                    set_sampler_labels = sorted(list(set(sampler_labels)))

                    for i in range(2):
                        if i not in set_sampler_labels:
                            counts.insert(i, 0)
                
                sampler_weights = [c/sum(counts) for c in counts] # Normalization
                
                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
            elif adaptive_sampling == "conf-class": # Confidence-Class-based
                _, counts = np.unique(train.labels, return_counts = True)
                sampler_ponder = [sum(counts)/c for c in counts]

                _, counts = np.unique(val.labels, return_counts = True)
                counts = [c for c in counts]
                
                samples_weights = []

                # Normalization
                full_conv = 0
                for c in range(len(counts)):
                    w = main_diagonal[c]/counts[c]

                    if w == 1:
                        full_conv+=1
                    
                    class_weights.append(1-w)

                # Full convergence
                if full_conv == len(counts):
                    for i in range(len(class_weights)):
                        class_weights[i] = 1/full_conv # Balanced

                class_weights = [w*sampler_ponder[i] for i, w in enumerate(class_weights)] # Ponderation
                class_weights = [w/sum(class_weights) for w in class_weights] # Normalization

                confs_weights = [1-w for w in confs]

                for i, l in enumerate(train.labels):
                    idx = i+1
                    if idx in inputs_idx:
                        samples_weights.append(class_weights[l]*confs_weights[inputs_idx.index(idx)])
                    else:
                        samples_weights.append(class_weights[l])

                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

                # Sampler weights
                sampler_labels = []
                for s in list(sampler):
                    sampler_labels.append(train.labels[s])

                _, counts = np.unique(sampler_labels, return_counts = True)
                counts = list(counts)

                # Missed target
                if len(counts) != 2:
                    set_sampler_labels = sorted(list(set(sampler_labels)))

                    for i in range(2):
                        if i not in set_sampler_labels:
                            counts.insert(i, 0)
                
                sampler_weights = [c/sum(counts) for c in counts] # Normalization
                
                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
            else: # Class-based
                _, counts = np.unique(train.labels, return_counts = True)
                sampler_ponder = [sum(counts)/c for c in counts]

                _, counts = np.unique(val.labels, return_counts = True)
                counts = [c for c in counts]
                
                # Normalization
                full_conv = 0
                for c in range(len(counts)):
                    w = main_diagonal[c]/counts[c]

                    if w == 1:
                        full_conv+=1
                    
                    class_weights.append(1-w)

                # Full convergence
                if full_conv == len(counts):
                    for i in range(len(class_weights)):
                        class_weights[i] = 1/full_conv # Balanced

                class_weights = [w*sampler_ponder[i] for i, w in enumerate(class_weights)] # Ponderation
                class_weights = [w/sum(class_weights) for w in class_weights] # Normalization
                
                samples_weights = [class_weights[e] for e in train.labels]
                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

                # Sampler weights
                sampler_labels = []
                for s in list(sampler):
                    sampler_labels.append(train.labels[s])

                _, counts = np.unique(sampler_labels, return_counts = True)
                counts = list(counts)
                
                # Missed target
                if len(counts) != 2:
                    set_sampler_labels = sorted(list(set(sampler_labels)))

                    for i in range(2):
                        if i not in set_sampler_labels:
                            counts.insert(i, 0)

                sampler_weights = [c/sum(counts) for c in counts] # Normalization

                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

                if val_set == "train":
                    val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
                else: # "val"
                    val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
    else:
        if init_sampling == "stratified":
            train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = num_workers)

            if val_set == "train":
                val_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
            else: # "val"
                val_loader = DataLoader(val, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        elif init_sampling == "inverse-stratified":
            _, counts = np.unique(train.labels, return_counts = True)
            class_weights = [sum(counts)/c for c in counts]
            class_weights = [w*w for w in class_weights]

            class_weights = [w/sum(class_weights) for w in class_weights] # Normalization
                
            samples_weights = [class_weights[e] for e in train.labels]
            sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

            train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

            if val_set == "train":
                val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
            else: # "val"
                val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)
        else: # balanced
            _, counts = np.unique(train.labels, return_counts = True)
            class_weights = [sum(counts)/c for c in counts]

            class_weights = [w/sum(class_weights) for w in class_weights] # Normalization
                
            samples_weights = [class_weights[e] for e in train.labels]
            sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(samples_weights))

            train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

            if val_set == "train":
                val_loader = DataLoader(train, batch_size = batch_size, num_workers = num_workers)
            else: # "val"
                val_loader = DataLoader(val, batch_size = batch_size, num_workers = num_workers)

    return train_loader, val_loader, test_loader, class_weights, sampler_weights