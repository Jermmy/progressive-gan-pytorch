import os
from os.path import join
import cv2
import numpy as np

import torch
import torch.utils.data as data


class TrainDataset(data.Dataset):

    def __init__(self, celeba_hq_dir, train_file, resolution=1024, transform=None):
        self.resolution = resolution
        self.celeba_hq_dir = celeba_hq_dir
        self.files = []
        with open(train_file, 'r') as f:
            for line in f.readlines():
                self.files += [line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv2.imread(join(self.celeba_hq_dir, self.files[idx]))
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        # noise = np.random.randn(512, 1, 1).astype(np.float)
        noise = torch.randn(512, 1, 1)

        sample = {'image': image, 'noise': noise}

        return sample

