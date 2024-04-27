import torch
import json

# ------------ Hyperparameters ------------
def hyperparameters(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    batch_size = config['batch_size']
    block_size = config['block_size']
    max_iters = config['max_iters']
    eval_interval = config['eval_interval']
    learning_rate = config['learning_rate']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_iters = config['eval_iters']
    n_embd = config['n_embd']
    n_head = config['n_head']
    n_layer = config['n_layer']
    dropout = config['dropout']
    return (batch_size, block_size, max_iters, eval_interval, learning_rate,
            device, eval_iters, n_embd, n_head, n_layer, dropout)
# ----------------------------------------

def load_data(path) -> tuple[torch.Tensor, torch.Tensor, int, callable, callable]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # words = text.split()
    # vocab_size = len(words)
    # stoi = {word: i for i, word in enumerate(words)}
    # itos = {i: word for i, word in enumerate(words)}
    # def encode(s): return [stoi[w] for w in s.split()]
    # def decode(ids): return ' '.join([itos[i] for i in ids])

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    

    return train_data, val_data, vocab_size, encode, decode


def get_batch(split, train_data, val_data, device, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters, train_data, val_data, device, block_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, device, block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

train_data, val_data, vocab_size, encode, decode = load_data('data/input.txt')