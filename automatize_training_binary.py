#%% LIBRARIES

import sys

#%% AUTOMATIZE TRAINING

dataset_type = "binary"
dataset = "yeast5" # "ecoli2", "vowel0" or "yeast5"
qnt_inputs = 8 # 7, 13, 8
qnt_outputs = 2 # 2, 2, 2
init_sampling = "stratified" # "inverse-stratified", "balanced" or "stratified"
val_set = "train" # "train" or "val"
baseline = False
if baseline: 
    z = 0
else:
    adaptive_sampling = "conf-class" # "class", "conf" or "conf-class"
    z = 1

for e in range(1, 6):
    if baseline:
        file_name = open("train.py")
        file = file_name.read()
        sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_{}_{}".format(dataset_type, dataset, init_sampling, val_set, e),
                    "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                    "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                    "start_epoch=0", "restore_model=n", "batch_size=32", "z={}".format(z),
                    "init_sampling={}".format(init_sampling),
                    "val_set={}".format(val_set)]
        exec(file)
        file_name.close()

        file_name = open("evaluate.py")
        file = file_name.read()
        sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_{}_{}".format(dataset_type, dataset, init_sampling, val_set, e),
                    "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                    "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                    "start_epoch=0", "restore_model=y", "batch_size=32", "z={}".format(z),
                    "init_sampling={}".format(init_sampling),
                    "val_set={}".format(val_set)]
        exec(file)
        file_name.close()
    else:
        file_name = open("train_resampling.py")
        file = file_name.read()
        sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_z-{}_{}_{}_{}".format(dataset_type, dataset, init_sampling, z, adaptive_sampling, val_set, e),
                    "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                    "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                    "start_epoch=0", "restore_model=n", "batch_size=32", "z={}".format(z),
                    "init_sampling={}".format(init_sampling),
                    "adaptive_sampling={}".format(adaptive_sampling), "val_set={}".format(val_set)]
        exec(file)
        file_name.close()

        file_name = open("evaluate.py")
        file = file_name.read()
        sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_z-{}_{}_{}_{}".format(dataset_type, dataset, init_sampling, z, adaptive_sampling, val_set, e),
                    "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                    "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                    "start_epoch=0", "restore_model=y", "batch_size=32", "z={}".format(z),
                    "init_sampling={}".format(init_sampling),
                    "adaptive_sampling={}".format(adaptive_sampling), "val_set={}".format(val_set)]
        exec(file)
        file_name.close()