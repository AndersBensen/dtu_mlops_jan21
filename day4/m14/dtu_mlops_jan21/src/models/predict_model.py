import os

import click
import torch
from model import MyAwesomeModel

from src.data.dataset import MnistDataset
import hydra 

def main():
    input_model = 'models/running_model.pth'
    input_data = 'data/processed/test_tensor.pt'

    print("Evaluating until hitting the ceiling")
    print(f'Using the model: {input_model}')
    print(f'Training data dir: {input_data}')

    test_set = MnistDataset(input_data)

    model = MyAwesomeModel()
    print(input_model)
    state_dict = torch.load(input_model)
    model.load_state_dict(state_dict)
    model.eval()

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
    main()