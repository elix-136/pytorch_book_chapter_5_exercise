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

mean = imgs.view(3,-1).mean(dim=1)
std = imgs.view(3,-1).std(dim=1)

t_cifar10 = datasets.CIFAR10(data_path, 
                           train = True, 
                           download = True,
                           transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((imgs.view(3,-1).mean(dim=1)),
                                                    (imgs.view(3,-1).std(dim=1)))    
                            ]
                           ))

img_t, _ = t_cifar10[99]
plt.imshow(img_t.permute(1,2,0))
plt.show()