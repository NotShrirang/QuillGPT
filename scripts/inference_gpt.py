import torch
import sys
import os
import argparse

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

from core.models.gpt import GPTLanguageModel
from utils.gptutils import hyperparameters, load_data
from core.tokenizer.tokenizer import Tokenizer

args = argparse.ArgumentParser()

args.add_argument('--config', type=str, default='config/shakespearean_config.json',
                    help='Path to the config file')

args.add_argument('--weights', type=str, default='../weights/GPT_model_char.pt',
                    help='Path to the weights file')

args.add_argument('--name', type=str, default='Shakespearean GPT',
                    help='Name of the model')

args = args.parse_args()

config_path = args.config
weights_path = args.weights
name = args.name

config_path = os.path.join(parent_dir, config_path)
tokenizer: Tokenizer = Tokenizer()
print("Loading tokenizer...")
tokenizer.from_pretrained(config_path)
print("Loaded tokenizer from file {}".format(config_path))

print("Loading hyperparameters...")
(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout, vocab_size) = hyperparameters(config_path=config_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

print("\nLoading model...")
model = GPTLanguageModel(
    vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, name).to(device)
state_dict = torch.load(
    os.path.join(parent_dir, weights_path),
    map_location=device)

model.load_state_dict(state_dict)

print("\nModel is of ", sum(p.numel()
      for p in model.parameters())/1e6, 'M parameters')
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt = """Write a scene about ROMEO arguing with JULIET.
ROMEO:"""
input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
# print(decode(model.generate(input, max_new_tokens=1000)[0].tolist()))

print("Generating text...")
generated_text = []
for idx in model.generate(input, 100):
    print(tokenizer.decode(idx[0].tolist())[-1], end='')
    generated_text.append(tokenizer.decode(idx[0].tolist()))

print("\nInference completed.")