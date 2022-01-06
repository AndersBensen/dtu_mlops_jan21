import os

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from pytorch_lightning import Trainer
from src.data.dataset import MnistDataset
import hydra 

# Training params
batch_size = 0
learning_rate = 0
epochs = 0
data_dir = ""
test_dir = ""

# Model params
seed = 0

@hydra.main(config_name="training_config.yaml")
def set_training_params(cfg):
    global learning_rate
    learning_rate = cfg.hyperparameters.learning_rate
    global batch_size
    batch_size = cfg.hyperparameters.batch_size
    global epochs
    epochs = cfg.hyperparameters.epochs
    global data_dir
    data_dir = cfg.hyperparameters.data_dir
    global test_dir
    test_dir = cfg.hyperparameters.test_dir

@hydra.main(config_name="model_config.yaml")
def set_model_params(cfg):
    global seed 
    seed = cfg.hyperparameters.seed 

def main():
        print(f'Training data dir: {data_dir}')
        torch.manual_seed(seed)
        model = MyAwesomeModel()

        train_set = MnistDataset(data_dir)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = MnistDataset(test_dir)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

        trainer = Trainer(max_epochs=5,limit_train_batches=0.2)
        trainer.fit(model, train_dataloader=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    set_training_params()
    set_model_params()
    main()