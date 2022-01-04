import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np 

base_path = "/Users/andersbensen/Documents/github/"
test_path = base_path + "dtu_mlops/data/corruptmnist/test.npz"
train_paths = [base_path + "dtu_mlops/data/corruptmnist/train_"+str(i)+".npz" for i in range(5)]

class MnistDataset(Dataset):
    def __init__(self, test=False):
        self.ids = []
        if (test == True):
            test = np.load(test_path)
            images = [i for i in test['images']]
            labels = [i for i in test['labels']]

            for i in range(len(labels)):
                self.ids.append((images[i], labels[i]))
        else:
            trains = [np.load(i) for i in train_paths]
            images = [i['images'] for i in trains]
            labels = [i['labels'] for i in trains]

            images_concat = np.concatenate(images, axis=0)
            labels_concat = np.concatenate(labels, axis=0)
            for i in range(len(labels_concat)):
                self.ids.append((images_concat[i],labels_concat[i]))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image, label = self.ids[i]

        image = torch.tensor(image)
        label = torch.tensor(label)

        # image = image.view(-1)

        return image, label

def mnist():
    train = MnistDataset()
    test = MnistDataset(test=True)
    return train, test 