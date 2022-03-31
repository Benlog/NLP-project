import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.jit import script, trace

#pylint: disable=no-member
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class discrimNN(nn.Module):
    '''
        Discriminator in the GAN model
    '''

    def __init__(self, in_size, mem_size):
        super().__init__()
        self.gru = nn.GRU(in_size, mem_size, batch_first=True)
        self.trilin = nn.Sequential(
            torch.nn.Linear(mem_size, mem_size),
            nn.Hardtanh(),
            torch.nn.Linear(mem_size, mem_size),
            nn.Hardtanh(),
            torch.nn.Linear(mem_size, 1),
            nn.Hardtanh()
        )

    def forward(self, x):
        _, x = self.gru(x)
        x = x.reshape(-1, self.gru.hidden_size)
        x = self.trilin(x)
        return x.reshape(-1)
