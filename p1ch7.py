import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms

data_path = '../data/'
cifar10 = datasets.CIFAR10(data_path, 
                           train = True, 
                           download = True,
                           transform = transforms.ToTensor())

cifar10_val = datasets.CIFAR10(data_path, 
                               train=False, 
                               download = True,
                               transform = transforms.ToTensor())

imgs = torch.stack([img_t for img_t, _ in cifar10], dim=3)
print(imgs.shape)