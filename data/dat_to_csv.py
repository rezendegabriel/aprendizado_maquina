#%% LIBRARIES

import pandas as pd

#%% GLOBAL VARIABLES

# Cross validation
k_fold = 5

# Data path
dataset_type = "binary" # "binary" or "multi-class"
dataset_folder = "ecoli2" # "ecoli2", "vowel0" or "yeast5" (binary), "contraceptive", "glass" or "pageblocks" (multi-class)
dataset_path = "datasets/{}/{}-{}-fold".format(dataset_type, dataset_folder, k_fold)

# .DAT to .CSV
data_csv = False

#%% DAT TO CSV

for k in range(1, k_fold+1):
    train_file = open("{}/{}-{}-{}tra.dat".format(dataset_path, dataset_folder, k_fold, k))
    test_file = open("{}/{}-{}-{}tst.dat".format(dataset_path, dataset_folder, k_fold, k))
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    features = []
    features_type = []
    
    data_flag = False # Initial data flag
    
    # Read train file
    for line in train_file:
        if data_flag:
            line_split = line.split(", ");
            
            for i in range(len(line_split)-1):
                if features_type[i] == "real":
                    line_split[i] = float(line_split[i])
                elif features_type[i] == "integer":
                    line_split[i] = int(line_split[i])
                    
            line_split[len(line_split)-1] = line_split[len(line_split)-1].replace("\n", "")
            
            df_train.loc[len(df_train)] = line_split
        else:
            line_split = line.split(" ");
            
        if line_split[0] == "@attribute":
            df_train.insert(len(features), line_split[1], "")
            df_test.insert(len(features), line_split[1], "")
            
            features.append(line_split[1]) # Feature
            features_type.append(line_split[2]) # Type
        else:
            if line_split[0] == "@data\n":
                data_flag = True
    
    # Class normalization
    column_class = features[len(features)-1]
    classes = sorted(list(df_train[column_class].unique()))
    
    for i, c in enumerate(classes):
        df_train.loc[df_train[column_class] == c, column_class] = i
    
    df_train.to_csv("{}/{}-{}-{}tra.csv".format(dataset_path, dataset_folder, k_fold, k), index = False)
    
    data_flag = False # Initial data flag
    
    # Read test file
    for line in test_file:
        if data_flag:
            line_split = line.split(", ");
            
            for i in range(len(line_split)-1):
                if features_type[i] == "real":
                    line_split[i] = float(line_split[i])
                elif features_type[i] == "integer":
                    line_split[i] = int(line_split[i])
                
            line_split[len(line_split)-1] = line_split[len(line_split)-1].replace("\n", "")
                
            df_test.loc[len(df_test)] = line_split
        else:
            line_split = line.split(" ");
            
        if line_split[0] == "@data\n":
            data_flag = True

    # Class normalization
    for i, c in enumerate(classes):
        df_test.loc[df_test[column_class] == c, column_class] = i
                
    df_test.to_csv("{}/{}-{}-{}tst.csv".format(dataset_path, dataset_folder, k_fold, k), index = False)