import numpy as np
import torch


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.batch_size = None
        self.block_size = None
        self.data = None
        self.train_data = None
        self.val_data = None

    def load_data(self, block_size=128, split=0.8, batch_size=64, device='cpu'):
        with open(self.data_path, 'r') as f:
            data = f.read()
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data = data

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        batch = [self.data[i] for i in indexes]
        batch = np.array(batch)
        return batch

    def get_batch(self, split, device='cpu'):
        if self.data is None:
            raise ValueError('Data not loaded')
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


class Encoder:
    def __init__(self, data, type='char'):
        self.data = data
        self.type = type
        self.vocab_size = None
        if type == 'char':
            self.chars = sorted(list(set(data)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
        elif type == 'word':
            self.words = data.split()
            self.stoi = {word: i for i, word in enumerate(self.words)}
            self.itos = {i: word for i, word in enumerate(self.words)}
            self.vocab_size = len(self.words)
        else:
            raise ValueError('Type must be either "char" or "word"')

    def encode(self, string: str):
        if self.type == 'char':
            return torch.tensor([self.stoi[c] for c in string])
        elif self.type == 'word':
            return torch.tensor([self.stoi[w] for w in string.split()])
        else:
            raise ValueError('Type must be either "char" or "word"')

    def decode(self, ids: list):
        if self.type == 'char':
            return ''.join([self.itos[i] for i in ids])
        elif self.type == 'word':
            return ' '.join([self.itos[i] for i in ids])
        else:
            raise ValueError('Type must be either "char" or "word"')
