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
    
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default=project_dir+"models/running_model.pth")
        parser.add_argument('--data', default=project_dir + "data/processed/test_tensor.pt")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        print(f'Using the model: {args.load_model_from}')
        print(f'Training data dir: {args.data}')
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)
        model.eval()
        test_set = MnistDataset(args.data)

        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

        running_acc = 0
        for images, labels in testloader:

            log_ps = model(images.float())

            ## accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_acc += accuracy.item()*100
        
        test_acc = running_acc/len(testloader)
        print(f'Accuracy: {test_acc}%')

if __name__ == '__main__':
    TrainOREvaluate()