import torch
from core.bigram import BigramLanguageModel
from utils.preprocessing import Encoder, DataLoader

batch_size = 32
block_size = 64
max_iters = 300
eval_interval = 30
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 32

data_loader = DataLoader('input.txt')
data_loader.load_data(block_size=block_size,
                      batch_size=batch_size, device=device, split=0.9)

encoder = Encoder(data_loader.data, type='word')
vocab_size = encoder.vocab_size

model = BigramLanguageModel(vocab_size, n_embd, block_size).to(device)
state_dict = torch.load(
    './weights/model_context64.pt', map_location=device)

model.load_state_dict(state_dict)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoder.decode(model.generate(context, 500)[0].tolist()))
