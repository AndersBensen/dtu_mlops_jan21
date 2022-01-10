import hydra
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.data.dataset import MnistDataset

# Training params
batch_size = 0
learning_rate = 0
epochs = 0
data_dir = ""

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


@hydra.main(config_name="model_config.yaml")
def set_model_params(cfg):
    global seed
    seed = cfg.hyperparameters.seed


def main():
    print(f'Training data dir: {data_dir}')
    torch.manual_seed(seed)
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

    train_set = MnistDataset(data_dir)
    trainloader = torch.utils.data.DataLoader(train_set[:3], batch_size=batch_size, shuffle=True)

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
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss}")
        print('-' * 35)
        torch.save(model.state_dict(), 'models/running_model.pth')

    plt.plot(train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.savefig("reports/figures/training_plot.png")


if __name__ == '__main__':
    set_training_params()
    set_model_params()
    main()
