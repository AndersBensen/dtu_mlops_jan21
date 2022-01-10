from src.data.dataset import MnistDataset
import torch 
import numpy as np
import pytest
import os.path

train_set = MnistDataset('data/processed/train_tensor.pt')
test_set = MnistDataset('data/processed/test_tensor.pt')


@pytest.mark.skipif(not os.path.exists('data/processed/train_tensor.pt'), reason="Data files not found")
def test_train_data():
    assert len(train_set) == 25000, "Traning dataset did not have the correct number of samples"

@pytest.mark.skipif(not os.path.exists('data/processed/test_tensor.pt'), reason="Data files not found")
def test_test_data():
    assert len(test_set) == 5000, "Testing datataset did not have the correct number of samples"   


# Test data shapes
rand_tens = torch.rand(28,28)
for i in train_set: assert i[0].shape == rand_tens.shape, "All samples in training dataset not have the right shape" 
for i in test_set: assert i[0].shape == rand_tens.shape , "All samples in training dataset not have the right shape"

# Test labels
train_labels = [int(i[1]) for i in train_set]
test_labels = [int(i[1]) for i in test_set]

numbers = set(range(0,10))
assert set(train_labels) == numbers, "All the labels were not found in the training dataset"
assert set(test_labels) == numbers, "All the labels were not found in the test dataset"