import os
import numpy as np
import torch


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.dataset = [[np.array([0,]), np.array([0,])]]

        self.transform = transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = {'img': self.dataset[idx][0].copy(), 'label': self.dataset[idx][1].copy()}

        if self.transform:
            sample = self.transform(sample)

        return sample
