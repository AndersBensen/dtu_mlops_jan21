import torch 
import matplotlib.pyplot as plt

rand_tensor = torch.randn(1,256,256)
quant = torch.quantize_per_tensor(rand_tensor, 0.1, 1, torch.qint8)
dequant = quant.dequantize()
