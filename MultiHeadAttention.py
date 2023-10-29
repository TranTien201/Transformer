import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dr=0.3):
        super().__init__()
        self.wQ = nn.Linear(dim, dim)
        self.wK = nn.Linear(dim, dim)
        self.wV = nn.Linear(dim, dim)
        self.wO = nn.Linear(dim, dim)
        self.d_model = dim // n_head
        self.n_head = n_head
        self.dropout = nn.Dropout(dr)

    def forward(self, x, mask=None):
        batch_size, seq_length, dim = x.size()
        Q = self.wQ(x).view(batch_size, seq_length, self.n_head, self.d_model).permute(0, 2, 1, 3)
        K = self.wK(x).view(batch_size, seq_length, self.n_head, self.d_model).permute(0, 2, 3, 1)
        V = self.wV(x).view(batch_size, seq_length, self.n_head, self.d_model).permute(0, 2, 1, 3)

        score = torch.matmul(Q, K) / math.sqrt(self.d_model)

        if mask is not None:
            score += mask

        score = torch.nn.functional.softmax(score, dim=-1)
        attention = torch.matmul(self.dropout(score), V)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model * self.n_head)

        return self.wO(attention)


if __name__ == '__main__':
    x = torch.rand(3, 4, 768)
    m = MultiHeadAttention(768, 8)(x)

