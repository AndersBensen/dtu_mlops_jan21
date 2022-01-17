import logging
import os

import click
import matplotlib.pyplot as plt
import torch
from google.cloud import storage
from model import MyAwesomeModel
from torch import nn, optim

from src.data.dataset import MnistDataset

log = logging.getLogger(__name__)


def save_model(gcloud_dir, local_dir):
    """Saves the model to Google Cloud Storage"""
    log.info(f"Saving model on gcloud at: {gcloud_dir}")
    bucket = storage.Client().bucket(gcloud_dir)
    if (os.path.isdir(local_dir)):
        files = os.listdir(local_dir)
        for f in files:
            blob = bucket.blob(local_dir + f)
            blob.chunk_size = 5 * 1024 * 1024  # Increase upload time to prevent timeout
            blob.upload_from_filename(local_dir + f)
    else:
        blob = bucket.blob(local_dir)
        blob.chunk_size = 5 * 1024 * 1024  # Increase upload time to prevent timeout
        blob.upload_from_filename(local_dir)


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
def main(input_data):
    print(f'Training data dir: {input_data}')

    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(1e-4))

    train_set = MnistDataset(input_data)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    epochs = 10

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
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss}")
        print('-' * 35)
        os.makedirs("models/", exist_ok=True)
        torch.save(model.state_dict(), 'models/running_model.pth')
    save_model()

    plt.plot(train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/training_plot.png")


if __name__ == '__main__':
    main()
