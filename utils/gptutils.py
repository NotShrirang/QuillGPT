import torch

# ------------ Hyperparameters ------------
batch_size = 16
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 12
n_layer = 18
dropout = 0.2
# ----------------------------------------


def hyperparameters():
    return (batch_size, block_size, max_iters, eval_interval, learning_rate,
            device, eval_iters, n_embd, n_head, n_layer, dropout, vocab_size)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()
vocab_size = len(words)
stoi = {word: i for i, word in enumerate(words)}
itos = {i: word for i, word in enumerate(words)}


def encode(s): return [stoi[w] for w in s.split()]


def decode(ids): return ' '.join([itos[i] for i in ids])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
