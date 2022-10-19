import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

__all__ = [
    "PalindromeDataset"
]


class PalindromeDataset(Dataset):

    def __init__(
        self,
        vocab_length: int,
        sequence_length: int,
        size: int,
        pos_rate: float = 0.5,
    ):
        super().__init__()
        self.vocab_length = vocab_length
        self.sequence_length = sequence_length
        self.size = size

        data = torch.randint(0, vocab_length, size=(size, sequence_length // 2))
        data = torch.cat([data, data.flip(dims=(-1,))], dim=-1)
        # shuffle sequences so that they are no longer palindromes (with high probability)
        for i in range(int(size * pos_rate), size):
            data[i] = data[i][torch.randperm(self.sequence_length)]
        self.data = data
        # set labels 
        self.labels = torch.zeros(size)
        self.labels[:int(size * pos_rate)] = 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sequence, label = self.data[idx], self.labels[idx]
        # one hot encode input sequence
        sequence = F.one_hot(sequence, num_classes=self.vocab_length)
        return sequence.float(), label.float()
