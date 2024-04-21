import json
import os
import sys
import torch
import tqdm

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

import utils.gptutils as gptutils
from core.models.gpt import GPTLanguageModel

config_path = os.path.join(parent_dir, 'config/config.json')
with open(config_path) as f:
    config = json.load(f)
data_path = os.path.join(parent_dir, config['data_path'])
name = "GPT"

print("Loading data...")
train_data, val_data, vocab_size, encode, decode = gptutils.load_data(data_path)
print(f"\nData loaded from `{data_path}`. Vocab size: {vocab_size}.")

(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout, vocab_size) = gptutils.hyperparameters(config_path=config_path, data_path=data_path)


print("Loading model...")
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, name)
model = model.to(device)
print("Loaded model on device `{}`".format(device))
print("""Hyperparameters:
    batch_size: {}
    block_size: {}
    max_iters: {}
    eval_interval: {}
    learning_rate: {}
    eval_iters: {}
    n_embd: {}
    n_head: {}
    n_layer: {}
    dropout: {}
    vocab_size: {}
""".format(batch_size, block_size, max_iters, eval_interval, learning_rate,
           eval_iters, n_embd, n_head, n_layer, dropout, vocab_size))
print("\nModel is of ", sum(p.numel()
      for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nStarting training...")
for iter in tqdm.tqdm(range(max_iters)):
    if not iter == 0:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = gptutils.estimate_loss(model, gptutils.get_batch, eval_iters)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss \
                    {losses['val']:.4f}")

    xb, yb = gptutils.get_batch('train', train_data, val_data, device, block_size, batch_size)


    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete.")

print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(gptutils.decode(model.generate(context, max_new_tokens=100)[0].tolist()))

# save the model
os.makedirs('./trained_models', exist_ok=True)
torch.save(model.state_dict(), './trained_models/GPT_model_letter.pt')

print("Saved model to `/trained_models/GPT_model_letter.pt`")