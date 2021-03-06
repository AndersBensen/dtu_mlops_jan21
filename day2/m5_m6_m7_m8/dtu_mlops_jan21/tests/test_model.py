import pytest
import torch

from src.models.model import MyAwesomeModel

model = MyAwesomeModel()

inputs = [(torch.rand(2, 28, 28), 10), (torch.rand(3, 28, 28), 10), (torch.rand(4, 28, 28), 10)]


@pytest.mark.parametrize("test_input, expected", inputs)
def test_eval(test_input, expected):
    y = model(test_input).squeeze(0)
    assert int(y.shape[1]) == expected, "The output shape was not [10] as expected"

# def test_error_on_wrong_shape():
#     with pytest.raises(ValueError, match='Expected shape to be [x,28,28]'):
#         model(torch.randn(28,28))
