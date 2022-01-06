import os

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.data.dataset import MnistDataset
import hydra 

seed = 0

@click.command()
@click.argument('input_data', type=click.Path(exists=True))
def main(input_data):
        print(f'Training data dir: {input_data}')
        
        #First training parameters
        # seed = cfg.hyperparameters.seed 
        # learning_rate = cfg.hyperparameters.lr
        # batch_size = cfg.hyperparameters.batch_size

        # torch.manual_seed(seed)
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()    
        optimizer = optim.Adam(model.parameters(), lr=float(1e-4))

        train_set = MnistDataset(input_data)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        epochs = 5

        train_losses = []

        for e in range(epochs):
            print(f"##### Epoch {e} ######")
            
            running_loss = 0
            for images, labels in trainloader:
                
                optimizer.zero_grad()
                
                log_ps = model(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            train_loss = running_loss/len(trainloader)
            train_losses.append(train_loss)
            print(f"Training loss: {train_loss}")
            print('-'*35)
            torch.save(model.state_dict(), 'models/running_model.pth')

        plt.plot(train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.savefig("reports/figures/training_plot.png")
    
if __name__ == '__main__':
    main()