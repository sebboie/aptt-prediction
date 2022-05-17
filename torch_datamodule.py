import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# THESE 3 LOADERS NEED TO BE IMPLEMENTED FOR EACH DATA SET
from preprocessing import train_loader, val_loader, test_loader


class Dataset(Dataset):
    def __init__(self, loader):
        self.data = loader()

    def __getitem__(self, idx):
        x = self.data["x"][idx]
        y = self.data["y"][idx]

        return x, y

    def __len__(self):
        return len(self.data["x"])


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_data = Dataset(train_loader)
        self.val_data = Dataset(val_loader)
        self.test_data = Dataset(test_loader)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, drop_last=True, pin_memory=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, drop_last=True, pin_memory=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, drop_last=False)
