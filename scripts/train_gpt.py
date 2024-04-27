import json
import os
import sys
import torch
import tqdm
import argparse

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

import utils.gptutils as gptutils
from core.models.gpt import GPTLanguageModel
from core.tokenizer.tokenizer import Tokenizer

args = argparse.ArgumentParser()

args.add_argument('--config_path', type=str, default='config/config.json',
                    help='Path to the config file')

args.add_argument('--data_path', type=str, default='data/corpus.txt',
                    help='Path to the data file')

args.add_argument('--name', type=str, default='GPT',
                    help='Name of the model')

args.add_argument('--output_dir', type=str, default='trained_models',
                    help='Path to save the model')

args = args.parse_args()

config_path = args.config_path
data_path = args.data_path
name = args.name
save_path = args.output_dir

tokenizer = Tokenizer(data_path=data_path)
with open(config_path) as f:
    config = json.load(f)

print("\nLoading data...")
train_data, val_data, vocab_size, encode, decode = tokenizer.train_data, tokenizer.val_data, tokenizer.vocab_size, tokenizer.encode, tokenizer.decode
print(f"Data loaded from `{data_path}`. Vocab size: {tokenizer.vocab_size}.")

(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout) = gptutils.hyperparameters(config_path=config_path)


print("\nLoading model...")
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, name)
model = model.to(device)
print("\nLoaded model on device `{}`".format(device))
print("""\nHyperparameters:
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
history = {}
history['train'] = []
history['val'] = []
for iter in tqdm.tqdm(range(max_iters)):
    if not iter == 0:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = gptutils.estimate_loss(model, gptutils.get_batch, eval_iters, tokenizer.train_data, tokenizer.val_data, device, block_size, batch_size)
            history['train'].append(losses['train'])
            history['val'].append(losses['val'])
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss \
                    {losses['val']:.4f}")

    xb, yb = gptutils.get_batch('train', tokenizer.train_data, tokenizer.val_data, device, block_size, batch_size)


    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.history = history
print("\nTraining complete.")


def inference(input, model: GPTLanguageModel, max_tokens, temperature):
    for idx in model.generate(idx=input, max_new_tokens=max_tokens, max_seq_length=50, temperature=temperature):
        text = tokenizer.decode(idx[0].tolist())[-1]
        print(text, end='')

print("\nGenerating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)

inference(input=context, model=model, max_tokens=10, temperature=0.1)

os.makedirs('./trained_models', exist_ok=True)

save_model_path = os.path.join(parent_dir, save_path, name + ".pt")
torch.save(model.state_dict(), save_model_path)

config['vocab_size'] = tokenizer.vocab_size
config.update({"encode": tokenizer.stoi, "decode": tokenizer.itos})
save_config_path = os.path.join(parent_dir, save_path, 'config.json')
with open(save_config_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f"\n\nSaved model config file at {save_config_path}")
print("Saved model to `{}`".format(save_model_path))