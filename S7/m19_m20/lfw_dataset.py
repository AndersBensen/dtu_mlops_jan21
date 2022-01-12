"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os 


def get_files(path_to_folder='/Users/andersbensen/Documents/university/dtu/2sem/mlops/lfw_dataset/lfw/'):
    images = []
    directories = os.listdir(path_to_folder)
    for d in directories: 
        dir_path = path_to_folder + d + "/"
        files = os.listdir(dir_path)
        for f in files:
            file_dir_path = dir_path + f
            images.append(file_dir_path)
    return images

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.dstack(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.images = []
        directories = os.listdir(path_to_folder)
        for d in directories: 
            dir_path = path_to_folder + d + "/"
            files = os.listdir(dir_path)
            for f in files:
                file_dir_path = dir_path + f
                self.images.append(file_dir_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.images[index])
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='/Users/andersbensen/Documents/university/dtu/2sem/mlops/lfw_dataset/lfw/', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.visualize_batch:
        batch = next(iter(dataloader))
        show(batch)
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)

        mean = np.mean(res)
        mean_format = "{:.2f}".format(mean)

        std = np.std(res)
        std_format = "{:.2f}".format(std)

        print(f'Timing: {mean_format}+-{std_format}')
