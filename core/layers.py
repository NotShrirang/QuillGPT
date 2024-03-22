import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        print(x.shape)
        K = self.key(x)
        Q = self.query(x)
        wei = Q @ K.transpose(-2, -1)
        wei = wei / (C ** 0.5)
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ self.value(x)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_embd, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionLayer(n_embd, head_size, head_size, 0.1) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attns = []
        for h in self.heads:
            attns.append(h(x))

        attns = torch.cat(attns, dim=-1)
        out = self.dropout(self.linear(attns))
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, n_embd, n_head, dropout) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head_attention = MultiHeadAttentionLayer(
            n_embd, n_head, head_size)
        self.feed_forward_network = FeedForwardBlock(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward_network(self.layer_norm2(x))
        return x
