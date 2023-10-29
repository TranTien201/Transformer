import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        idx = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10000, idx / self.d_model)
        print(denominator)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        print(position)
        print(even_PE)
        odd_PE = torch.cos(position / denominator)
        stack = torch.stack([even_PE, odd_PE], dim=2)
        print(stack.size())
        PE = torch.flatten(stack, start_dim=1, end_dim=2)
        return PE


if __name__ == '__main__':
    pe = PositionalEncoding(6, 10)()
