import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.num_features)
        mean = torch.mean(x, 0, keepdim=True)
        std = torch.std(x, 0, keepdim=True) + 1e-5
        x_norm = self.gamma * (x - mean) / std + self.beta

        return x_norm.reshape(shape)


if __name__ == '__main__':
    x = torch.randn(3, 4, 5)
    b = BatchNorm1d(5)(x)
    print(b.shape)
