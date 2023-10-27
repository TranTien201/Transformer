from Vocabulary import Vocab
import torch
import torch.nn as nn


class Embedding:
    def __init__(self, sentences, dimension=768):
        self.__indices = Vocab().__indices__(sentences)
        self.__vocab_size = len(Vocab())
        self.__emb_in_vocab = nn.Parameter(torch.randn(self.__vocab_size, dimension))
        self.__one_hot = torch.eye(self.__vocab_size)[self.__indices]

    def emb_v1(self):
        """
        Embedding verson 1 sẽ tạo ra one-hot vector có chiều (seq_length x vocab) @ (vocab x dim)
        :return: (seq_length x dim)
        """
        return torch.matmul(self.__one_hot, torch.unsqueeze(self.__emb_in_vocab, 0))

    def emb_v2(self):
        return self.__emb_in_vocab[self.__indices]


if __name__ == '__main__':
    sentences = ["Hôm nay tôi buồn ẻ", "Nguyễn Trần Tiến", "Không có biết"]
    e = Embedding(sentences)

