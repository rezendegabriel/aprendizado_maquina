#%% LIBRARIES

from torch.utils.data import Dataset

import numpy as np
import torch

#%% CLASS

class CustomDataset(Dataset):
    def __init__(self, datas, labels, id_data, transform = None):
        self.datas = datas
        self.labels = labels
        self.id_data = id_data
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, id_x):
        if torch.is_tensor(id_x):
            id_x = id_x.tolist()

        data = np.array(self.datas[id_x])

        if self.transform:
            data = self.transform(data)

        label = torch.tensor(self.labels[id_x]).type(torch.long)
        i = torch.tensor(self.id_data[id_x]).type(torch.long)

        sample = (data, label, i)

        return sample