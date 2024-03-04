import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32  # How many independent sequences to process in parallel
block_size = 8  # Maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

# Create a character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(string: str):
    return [stoi[c] for c in string]


def decode(ids: list):
    return ''.join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Set the model to evaluation mode
    for split in ['train', 'val']:
        # Store the loss values for each batch
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Get a batch
            logits, loss = model(X, Y)  # Compute the loss
            losses[k] = loss.item()  # Store the loss
        out[split] = losses.mean()  # Average the loss
    model.train()  # Set the model back to training mode
    return out  # Return the loss values


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        logits = self.lm_head(tok_emb)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
