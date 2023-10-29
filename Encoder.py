import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model, hidden, nums_head, drop=0.3):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, hidden, nums_head, drop)])

    def forward(self, x):
        return self.layers(x)
