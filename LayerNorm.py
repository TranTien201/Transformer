import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, nums_feature: int | list):
        super().__init__()
        if isinstance(nums_feature, int):
            nums_feature = [nums_feature]
        self.gamma = nn.Parameter(torch.ones(nums_feature))
        print(self.gamma.shape)
        self.beta = nn.Parameter(torch.zeros(nums_feature))
        self.nums_feature = nums_feature

    def forward(self, x, ep=1e-5):
        dims = [-(i + 1) for i in range(len(self.nums_feature))]
        mean = torch.mean(x, dim=dims, keepdim=True)
        print(mean.shape)
        std = torch.std(x, dim=dims, keepdim=True) + ep
        print(x.shape)
        return self.gamma * (x - mean) / std + self.beta


if __name__ == '__main__':
    x = torch.randn(3, 4, 5, 6)
    l = LayerNorm([4, 5, 6])(x)
