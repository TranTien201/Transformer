import json
import torch

class Vocab:
    def __init__(self, path=None):
        if path is not None:
            self.__path = path
        else:
            self.__path = "./vocab.json"
        with open("vocab.json", "r", encoding="utf-8") as f:
            self.__vocab = json.load(f)

    @property
    def vocal(self):
        return self.__vocab

    def __getitem__(self, item: str) -> int | None:
        try:
            return self.__vocab[item.lower()]
        except:
            return self.__vocab["<UNK>"]

    def __setitem__(self, key, value: dict) -> None:
        with open(self.__path, "w", encoding="utf-8") as f:
            self.__vocab.update(value)
            json.dump(self.__vocab, f, ensure_ascii=False, indent=4)

    def __len__(self):
        return len(self.__vocab)

    @classmethod
    def pad(cls, sentences):
        max_length = max([len(tokens.split()) for tokens in sentences])
        return [sent + str(" <PAD>") * (max_length - len(sent.split())) for sent in sentences], max_length, len(sentences)

    def __indices__(self, sentences: list[str]):
        sentences, max_length, nums_sent = self.pad(sentences)
        indices = [self.__getitem__(token) for sen in sentences for token in sen.split()]
        return torch.tensor(indices, dtype=torch.int32).view(nums_sent, max_length)


if __name__ == '__main__':
    v = Vocab()
    print(v.vocal)
    sentence = "Hôm nay tôi mắc ẻ quá Trần Tiến Không có".split()
    values = {}
    i = 0
    for idx in range(len(sentence)):
        if v[sentence[idx]] == 1 and sentence[idx].upper() not in ["<PAD>", "<UNK>"]:
            values[sentence[idx].lower()] = len(v) + i
            i += 1

    v["_"] = values
