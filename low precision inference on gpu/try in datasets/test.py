import CIFAR10
from CIFAR10 import Dequantization
import torch.nn as nn
import torch

activation_max_val=0

a=torch.Tensor([[0,1,2],[1,1,2],[2,1,2],[3,-1,2],[5,1,2],[5,1,2],[5,1,2],[5,1,2]])

def forward(x):
    global activation_max_val
    activation_max_val = abs(x.numpy()).max()
    x = x * 127 / float(activation_max_val)
    x = x.int()
    return x

print(forward(a))


