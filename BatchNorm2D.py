import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    def __init__(self, nums_feature):
        super().__init__()
        self.gama = nn.Parameter(torch.ones(nums_feature))
        self.beta = nn.Parameter(torch.zeros(nums_feature))

    def forward(self, x, ep=1e-5):
        N, C, H, W = x.size()

        x = x.permute(0, 2, 3, 1).reshape(-1, C)

        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(std + ep)
        return (self.gama * x_hat + self.beta).view(N, H, W, C).permute(0, 3, 1, 2)


if __name__ == '__main__':
    x = torch.randn(3, 2, 4, 5)
    b = BatchNorm2d(2)(x)
    print(b.shape)