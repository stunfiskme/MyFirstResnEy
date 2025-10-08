import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from resnet_18 import resnet18
from resnet_34 import resnet34
from resnet_50 import resnet50
from DataSet import train_dataloader, validation_dataloader
from EarlyStopping import EarlyStopping
from mixUp import mixup_criterion, mixup_data

if __name__ == '__main__':
    EPOCHS = 50
    #call train.py and have it take any model