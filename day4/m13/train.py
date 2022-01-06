"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from models import Encoder, Decoder, Model
from torch.optim import Adam, SGD
import argparse
import sys
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=float(1e-3))
    parser.add_argument('--optimizer', default='adam')
    args = parser.parse_args()

    wandb.init(config=args)

    config = wandb.config

    dataset_path = 'datasets'
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = 100
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 20
    lr = float(config.lr)
    print("Learning rate: ", lr)
    optim = config.optimizer
    print("Optimizer: ", optim)
    epochs = 10

    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    # wandb.watch(model, log_freq=100)

    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    if (optim == 'adam'):
        optimizer = Adam(model.parameters(), lr=lr)
    elif (optim == 'sgd'):
        optimizer = SGD(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()
    losses = []
    for epoch in range(epochs):
        print("\tEpoch", epoch + 1, "complete!")   
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        train_loss = overall_loss / (batch_idx*batch_size)
        losses.append(train_loss) 
        wandb.log({"train_loss": train_loss})
        print("train_loss: ", train_loss)

        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_loader):
                x = x.view(batch_size, x_dim)
                x = x.to(DEVICE)      

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)
                
                eval_loss += loss.item()
            val_loss = eval_loss / (batch_idx*batch_size)
            wandb.log({"val_loss": val_loss})
            print("test_loss: ", val_loss)
        
    print("Finish!!")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)      
            x_hat, _, _ = model(x)       
            break

    save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
    save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
        
    save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')

    # columns=["id", "original", "reconstructed"]
    # my_table = wandb.Table(columns=columns)
    # xs = [wandb.Image(im) for im in x.view(batch_size, 1, 28, 28)]
    # x_hats = [wandb.Image(im) for im in x.view(batch_size, 1, 28, 28)]
    # for i in range(len(xs)):
    #     my_table.add_data(i,xs[i],x_hats[i])
    # wandb.log({"vae_mnist": my_table})

    # wandb.log({"VAE generated MNIST images" : [wandb.Image(im) for im in generated_images.view(batch_size, 1, 28, 28)]})