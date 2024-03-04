from models.bigram import BigramLanguageModel
from utils.preprocessing import Encoder, DataLoader
from torch import nn
import torch
import tqdm


batch_size = 32
block_size = 64
max_iters = 300
eval_interval = 30
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 32

torch.manual_seed(1337)

data_loader = DataLoader('input.txt')
data_loader.load_data(block_size=block_size,
                      batch_size=batch_size, device=device, split=0.9)

encoder = Encoder(data_loader.data, type='word')
vocab_size = encoder.vocab_size

data_loader.data = torch.tensor(
    encoder.encode(data_loader.data), dtype=torch.long, device=device)

n = int(0.8 * len(data_loader.data))
data_loader.train_data = data_loader.data[:n]
data_loader.val_data = data_loader.data[n:]

model = BigramLanguageModel(vocab_size, n_embd, block_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for i in tqdm.tqdm(range(max_iters)):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = data_loader.get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'saved_models/model_context64.pt')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoder.decode(model.generate(context, 500)[0].tolist()))
