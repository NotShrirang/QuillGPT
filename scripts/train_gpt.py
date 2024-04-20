import tqdm
import torch
import sys
import os

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

import utils.gptutils as gptutils
from core.models.gpt import GPTLanguageModel

torch.manual_seed(0)

hyperparameters = gptutils.hyperparameters()
(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
 eval_iters, n_embd, n_head, n_layer, dropout, vocab_size) = hyperparameters

print("Loading model...")
model = GPTLanguageModel()
m = model.to(device)
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
      for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("\nStarting training...")
for iter in tqdm.tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if not iter == 0:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = gptutils.estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss \
                    {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = gptutils.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(gptutils.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), './weights/GPT_model_word.pt')
