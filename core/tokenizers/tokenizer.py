import json
import os
from typing import Iterable
import torch

class Tokenizer:
    def __init__(self, data_path: str = None):
        self.config = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        if data_path:
            self.data = self.load_data(data_path)
        else:
            self.data = None
    
    def from_pretrained(self, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        self.config = config
        if 'encode' not in config:
            raise ValueError("Config file must contain an 'encode' key.")
        if 'decode' not in config:
            raise ValueError("Config file must contain a 'decode' key.")
        if 'vocab_size' not in config:
            raise ValueError("Config file must contain a 'vocab_size' key.")
        stoi = config['encode']
        self.stoi = {k: int(v) for k, v in stoi.items()}
        itos = config['decode']
        self.itos = {int(k): v for k, v in itos.items()}
        self.vocab_size = config['vocab_size']
        return self
    
    def load_data(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError("File not found.")
        if not path.endswith('.txt'):
            raise ValueError("File must be a text file.")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(set(chars))
        stoi = {ch: i for i, ch in enumerate(set(chars))}
        itos = {i: ch for i, ch in enumerate(set(chars))}
        self.config = {"vocab_size": vocab_size, "encode": stoi, "decode": itos}
        self.stoi = stoi
        self.itos = itos
        data = torch.tensor(self(text), dtype=torch.long)
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        return text

    def __repr__(self) -> str:
        if self.config:
            return f"Tokenizer(config={self.config})"
        else:
            return f"Tokenizer()"
    
    def __str__(self) -> str:
        if self.config:
            return f"Tokenizer(config_path={self.config})"
        else:
            return f"Tokenizer()"
    
    def __len__(self) -> int:
        return len(self.stoi)
    
    def __getitem__(self, key: str) -> int:
        return self.stoi[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.stoi
    
    def __iter__(self):
        return iter(self.stoi)
    
    def __reversed__(self):
        return reversed(self.stoi)
    
    def keys(self):
        return self.stoi.keys()
    
    def values(self):
        return self.stoi.values()
    
    def items(self):
        return self.stoi.items()
    
    def __call__(self, *args, **kwds) -> list[int]:
        return self.encode(*args, **kwds)

    def encode(self, s: str | list[str]) -> list[int]:
        if isinstance(s, str):
            return [self.stoi[c] for c in s]
        elif isinstance(s, list):
            return [[self.stoi[i] for i in c] for c in s]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def decode(self, l: list[int]) -> str:
        if isinstance(l[0], int):
            return ''.join([self.itos[i] for i in l])
        elif isinstance(l[0], Iterable):
            return [''.join([self.itos[i] for i in c]) for c in l]
        else:
            raise ValueError("Input must be a list of integers or a list of list of integers.")
    
    def save_pretrained(self, path: str) -> str:
        with open(path + 'vocab.json', 'w') as f:
            json.dump(self.config, f)
        return "Tokenizer saved at {}.".format(path)
