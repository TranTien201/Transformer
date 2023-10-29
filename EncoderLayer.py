import torch
import torch.nn as nn
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention
from LayerNorm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, drop=0.3):
        super().__init__()
        self.mul = MultiHeadAttention(d_model, n_head, drop)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop)
        self.ff = FeedForward(d_model, hidden)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        resudial = x
        x = self.mul(x)
        x = self.drop1(x)
        x = resudial + x
        x = self.norm1(x)
        resudial = x
        x = self.ff(x)
        x = self.drop2(x)
        x = resudial + x
        return self.norm2(x)
