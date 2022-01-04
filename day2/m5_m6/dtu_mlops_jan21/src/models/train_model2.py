import argparse
import sys,os

import torch
from torch import nn, optim
from torch.utils import data

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from model import MyAwesomeModel

from src.data.dataset import MnistDataset 

project_dir =  os.path.abspath(os.path.join(__file__ ,"../../.."))+"/"

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):

        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=1e-4)
        parser.add_argument('--e', default=10)
        parser.add_argument('--data', default=project_dir + "data/processed/train_tensor.pt")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(f'Learning Rate used: {args.lr}')
        print(f'EPOCHS: {args.e}')
        print(f'Training data dir: {args.data}')
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()    
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

        train_set = MnistDataset(args.data)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        epochs = int(args.e)

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

            save_dir = project_dir+"/models/"
            torch.save(model.state_dict(), save_dir+'running_model.pth')

        plt.plot(train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.savefig(project_dir+"/reports/figures/training_plot.png")

if __name__ == '__main__':
    TrainOREvaluate()