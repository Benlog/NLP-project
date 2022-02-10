import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

#pylint: disable=no-member
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class discrimNN(nn.Module):
    '''
        Discriminator in the GAN model
    '''

    def __init__(self, in_size, mem_size):
        super(discrimNN, self).__init__()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))