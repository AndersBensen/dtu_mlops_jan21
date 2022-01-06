import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, path):
        print(path)
        self.ids = torch.load(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image, label = self.ids[i]
        return image, label