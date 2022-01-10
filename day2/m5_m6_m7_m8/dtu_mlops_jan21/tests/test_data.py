from src.data.dataset import MnistDataset
import torch 
import numpy as np
import pytest
import os.path

# Test data size
@pytest.mark.skipif(not os.path.exists('data/processed/train_tensor.pt'), reason="Data files not found")
def test_train_data():
    train_set = MnistDataset('data/processed/train_tensor.pt')
    assert len(train_set) == 25000, "Traning dataset did not have the correct number of samples"

@pytest.mark.skipif(not os.path.exists('data/processed/test_tensor.pt'), reason="Data files not found")
def test_test_data():
    test_set = MnistDataset('data/processed/test_tensor.pt')
    assert len(test_set) == 5000, "Testing datataset did not have the correct number of samples"   

# Test data shapes
@pytest.mark.skipif(not os.path.exists('data/processed/train_tensor.pt'), reason="Data files not found")
def test_train_shapes():
    rand_tens = torch.rand(28,28)
    train_set = MnistDataset('data/processed/train_tensor.pt')
    for i in train_set: assert i[0].shape == rand_tens.shape, "All samples in training dataset not have the right shape" 

@pytest.mark.skipif(not os.path.exists('data/processed/test_tensor.pt'), reason="Data files not found")
def test_test_shapes():
    rand_tens = torch.rand(28,28)
    test_set = MnistDataset('data/processed/test_tensor.pt')
    for i in test_set: assert i[0].shape == rand_tens.shape , "All samples in testing dataset not have the right shape"

# Test data labels
@pytest.mark.skipif(not os.path.exists('data/processed/train_tensor.pt'), reason="Data files not found")
def test_train_labels():
    train_set = MnistDataset('data/processed/train_tensor.pt')
    train_labels = [int(i[1]) for i in train_set]
    numbers = set(range(0,10))
    assert set(train_labels) == numbers, "All the labels were not found in the training dataset"

@pytest.mark.skipif(not os.path.exists('data/processed/test_tensor.pt'), reason="Data files not found")
def test_test_labels():
    test_set = MnistDataset('data/processed/test_tensor.pt')
    numbers = set(range(0,10))
    test_labels = [int(i[1]) for i in test_set]
    assert set(test_labels) == numbers, "All the labels were not found in the test dataset"