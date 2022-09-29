#%% LIBRARIES

from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

#%% SAMPLING DATA

def sample_data(df):
    attributes = list(df.columns)
    column_class = attributes.pop(len(attributes)-1)
    
    data_array = np.zeros(shape = (len(df), 1, len(attributes)))
    data_label = np.array(list(map(int, df[column_class])))
    data_idx = []
    
    for i, row in enumerate(df.index):
        data = df.loc[row, attributes].to_numpy(dtype = float)
        data_array[i] = data
        data_idx.append(i)

    return data_array, data_label, data_idx

#%% TENSOR TRANSFORM

mean, std = 0.5, 0.5 # Min-Max normalization

transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean = (mean), std = (std))
                                     ])

#%% DATALOADERS

def get_dataloaders(hps, k, resampling = False, n_sampling = 0, cf_matrix = None, missed_targets = None):
    # Hyper-parameters
    dataset_path = hps["dataset_path"]
    dataset = hps["dataset"]
    k_fold = hps["k_fold"]
    batch_size = hps["batch_size"]
    num_workers = hps["num_workers"]
    init_sampling = hps["init_sampling"]
    
    # Load data
    df_train = pd.read_csv("{}/{}-{}-{}tra.csv".format(dataset_path, dataset, k_fold, k))
    df_test = pd.read_csv("{}/{}-{}-{}tst.csv".format(dataset_path, dataset, k_fold, k))
    
    # Sampling data
    x_train, y_train, id_train = sample_data(df_train)
    x_test, y_test, id_test = sample_data(df_test)
    
    # Data tensor transform
    train = CustomDataset(x_train, y_train, id_train, transformation)
    test = CustomDataset(x_test, y_test, id_test, transformation)

    weights = [] # Class sampling distribution

    if resampling:
        if n_sampling == 0:
            if init_sampling == "stratified":
                train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = num_workers)

                _, counts = np.unique(train.labels, return_counts = True)
                weights = [c/sum(counts) for c in counts] # Normalization
            elif init_sampling == "inverse-stratified":
                _, counts = np.unique(train.labels, return_counts = True)
                weights = [sum(counts)/c for c in counts]

                weights = [w/sum(weights) for w in weights] # Normalization
                    
                samples_weights = [weights[e] for e in train.labels]
                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train.labels))

                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)
            else: # balanced
                _, counts = np.unique(train.labels, return_counts = True)
                weights = [1 for c in counts]

                weights = [w/sum(weights) for w in weights] # Normalization
                    
                samples_weights = [weights[e] for e in train.labels]
                sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train.labels))

                train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
                test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)
        else: # Adaptive sampling
            main_diagonal = cf_matrix.diagonal()
            main_diagonal = [c for c in main_diagonal]
            
            if missed_targets:
                for m_tg in missed_targets:
                    main_diagonal.insert(m_tg, 0)
            
            _, counts = np.unique(train.labels, return_counts = True)
            counts = [c for c in counts]
            
            weights = []
            
            # Normalization
            for c in range(len(counts)):
                w = main_diagonal[c]/counts[c]
                
                if w >= 1:
                    weights.append(0.05)
                else:
                    weights.append(1-w)
            
            weights = [w/sum(weights) for w in weights]
            
            samples_weights = [weights[e] for e in train.labels]
            sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train.labels))
            
            train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)
    else:
        if init_sampling == "stratified":
            train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        elif init_sampling == "inverse-stratified":
            _, counts = np.unique(train.labels, return_counts = True)
            weights = [sum(counts)/c for c in counts]

            weights = [w/sum(weights) for w in weights] # Normalization
                
            samples_weights = [weights[e] for e in train.labels]
            sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train.labels))

            train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)
        else: # balanced
            _, counts = np.unique(train.labels, return_counts = True)
            weights = [1 for c in counts]

            weights = [w/sum(weights) for w in weights] # Normalization
                
            samples_weights = [weights[e] for e in train.labels]
            sampler = WeightedRandomSampler(weights = samples_weights, num_samples = len(train.labels))

            train_loader = DataLoader(train, batch_size = batch_size, sampler = sampler, num_workers = num_workers)
            test_loader = DataLoader(test, batch_size = batch_size, num_workers = num_workers)

    return train_loader, test_loader, weights