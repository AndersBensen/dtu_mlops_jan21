import torch 
import torchvision.models as models
import time
import torch
import ptflops

resnet152 = models.resnet152(pretrained=True)
resnet152.eval()
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
mobilenet_v3_small.eval()

iterations = 20

resnet_start = time.time()
for i in range(iterations):
    resnet152(torch.rand(8,3,256,256))
resnet_time = time.time() - resnet_start

mobilenet_start = time.time()
for i in range(iterations):
    mobilenet_v3_small(torch.rand(8,3,256,256))
mobilenet_time = time.time() - mobilenet_start

print(f"Resnet152 inference time on {iterations} random samples: {resnet_time}")
print(f"Mobilenet v3 small inference time on {iterations} random samples: {mobilenet_time}")

macs_res, params_res = ptflops.get_model_complexity_info(resnet152, (3,256,256))
macs_mob, params_mob = ptflops.get_model_complexity_info(mobilenet_v3_small, (3,256,256))

print("Macs resnet152: ", macs_res)
print("Macs mobilenetv3 small: ", macs_mob)
