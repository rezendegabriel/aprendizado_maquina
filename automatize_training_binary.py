#%% LIBRARIES

import sys

#%% AUTOMATIZE TRAINING

dataset_type = "binary"
dataset = "yeast5" # "ecoli2", "vowel0" or "yeast5"
qnt_inputs = 8 # 7, 13, 8
qnt_outputs = 2 # 2, 2, 2
init_sampling = "stratified" # "inverse-stratified", "balanced" or "stratified"
z = 1

for e in range(1, 6):
    file_name = open("train_resampling.py")
    file = file_name.read()
    sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_z-{}_{}".format(dataset_type, dataset, init_sampling, z, e),
                "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                "start_epoch=0", "restore_model=n", "batch_size=32", "z={}".format(z),
                "init_sampling={}".format(init_sampling)]
    exec(file)
    file_name.close()
    
    file_name = open("evaluate.py")
    file = file_name.read()
    sys.argv = ["file.py", "model=mlp", "name=mlp_{}_{}_{}_z-{}_{}".format(dataset_type, dataset, init_sampling, z, e),
                "dataset_type={}".format(dataset_type), "dataset={}".format(dataset),
                "num_inputs={}".format(qnt_inputs), "num_outputs={}".format(qnt_outputs),
                "start_epoch=0", "restore_model=y", "batch_size=32", "z={}".format(z),
                "init_sampling={}".format(init_sampling)]
    exec(file)
    file_name.close()