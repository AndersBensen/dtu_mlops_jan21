import argparse
import sys

import torch
from torch import nn, optim

import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel


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
        parser.add_argument('--e', default=20)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(f'Learning Rate used: {args.lr}')
        print(f'EPOCHS: {args.e}')
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()    
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

        train_set, _ = mnist()
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
            torch.save(model.state_dict(), 'running_model.pth')

        plt.plot(train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.show()
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="./running_model.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        print(f'Using the model: {args.load_model_from}')
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)
        model.eval()
        _, test_set = mnist()

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
    
    
    
    
    
    
    
    
    