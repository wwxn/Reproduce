import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
import cv2
import random


a=torch.Tensor([[1.1,2.2,3.2],[4.4,5.5,6.5],[7.5,8.8,9.7]])
a=(a.numpy()+0.5)*256
a=np.array(a,'uint8')
print(a)


