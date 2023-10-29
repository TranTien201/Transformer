import math

import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        """
        x -> (batch_size, seq_length, dim) => (3, 4, 768)
        """
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x, mask=None):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        score = torch.matmul(q, torch.transpose(k, 1, 2)) / math.sqrt(q.shape[2])
        if mask is not None:
            score += mask

        torch.softmax(score, dim=-1)
        return torch.matmul(score, v)


if __name__ == '__main__':
    x = torch.randn(3, 10, 768)
    s = Self_Attention(768)(x)
    mask = torch.full([x.shape[1], x.shape[1]], float('-inf'))
    mask = torch.tril(mask, diagonal=1)
