import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import albumentations as A

from sklearn.model_selection import train_test_split


train_transform = A.Compose([
    A.Normalize(),
])
test_transform = A.Compose([
    A.Normalize(),
])



# CIFA-10数据集 
train_dataset = CIFAR10(root='./data/CIFA', train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root='./data/CIFA', train=False, download=True, transform=test_transform)

