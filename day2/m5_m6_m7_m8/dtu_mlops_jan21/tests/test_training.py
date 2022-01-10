import os

import pytest
from torch import nn, optim

from src.data.dataset import MnistDataset
from src.models.model import MyAwesomeModel


@pytest.mark.skipif(not os.path.exists('data/processed/train_tensor.pt'), reason="Data not found")
def test_train_labels():
    train_set = MnistDataset('data/processed/train_tensor.pt')

    model = MyAwesomeModel()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(3e-5))

    image, label = train_set[0]

    log_ps = model(image.unsqueeze(0).float())
    loss = criterion(log_ps, label.unsqueeze(0))
    loss.backward()

    optimizer.step()

    weights_pre_zg = model.conv_1.weight.grad.view(-1)

    assert len(set(weights_pre_zg.tolist())) > 1, "The gradients were all zero before zero gradding"

    optimizer.zero_grad()

    weights_after_zg = model.conv_1.weight.grad.view(-1)
    assert len(set(weights_after_zg.tolist())) == 1, "The gradients were NOT zero after zero grad"
