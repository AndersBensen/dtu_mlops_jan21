import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn

from src.data.dataset import MnistDataset
from src.models.model import MyAwesomeModel

project_dir = os.path.abspath(os.path.join(__file__, "../../..")) + "/"


def main():
    """
    This method loads a pretrained MNIST model and defines a new models with all but the last
    layer. The training data is then passed thorugh the data and visualized.

    """
    model = MyAwesomeModel()
    state_dict = torch.load("models/running_model.pth")
    model.load_state_dict(state_dict)
    model.eval()

    train_set = MnistDataset("data/processed/train_tensor.pt")

    class MyAwesomeModelFeature(nn.Module):
        def __init__(self, model, feature_number):
            super(MyAwesomeModelFeature, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-feature_number])

        def forward(self, x):
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
            x = self.features(x)
            return x

    feature_model = MyAwesomeModelFeature(model, -2)
    img, label = train_set[0]
    pred = feature_model(img.float())
    print(pred.shape)

    features = []
    labels = []

    for image, label in train_set:
        pred = feature_model(img.float())
        ps = torch.exp(pred)
        features.append(pred)
        labels.append(int(label))

    features_flat = []
    for f in features:
        features_flat.append(f[0].view(-1).detach().numpy())

    df = pd.DataFrame(features_flat)

    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(df)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
    plt.savefig(project_dir + "/reports/figures/TSNE_plot.png")


if __name__ == "__main__":

    main()
