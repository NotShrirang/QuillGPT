import torch
import sys
import os
import argparse
import colorama

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from core.models.gpt import GPTLanguageModel
from core.utils.gptutils import hyperparameters, load_data
from core.tokenizers.tokenizer import Tokenizer

args = argparse.ArgumentParser()

args.add_argument('--config_path', type=str, default='config/shakespearean_config.json',
                    help='Path to the config file')

args.add_argument('--weights_path', type=str, default='weights/GPT_model_char.pt',
                    help='Path to the weights file')

args.add_argument('--prompt', type=str, default="""Write a scene about ROMEO arguing with JULIET.
ROMEO:""")

args.add_argument('--max_length', type=int, default=500, help='Maximum length of the generated text')

args.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')

args = args.parse_args()

config_path = args.config_path
weights_path = args.weights_path
prompt = args.prompt
max_length = args.max_length
temperature = args.temperature
if temperature < 0.0001:
    temperature = 0.0001
    print("Temperature must be greater than 0.0001. Setting temperature to 0.0001.")

config_path = os.path.join(parent_dir, config_path)
tokenizer: Tokenizer = Tokenizer()
print("Loading tokenizer...")
tokenizer.from_pretrained(config_path)
vocab_size = tokenizer.vocab_size
print("Loaded tokenizer from file {}".format(config_path))

print("Loading hyperparameters...")
(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout) = hyperparameters(config_path=config_path)
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
    vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, "Shakespearean GPT").to(device)
state_dict = torch.load(
    os.path.join(parent_dir, weights_path),
    map_location=device)

model.load_state_dict(state_dict)

print("\nModel is of ", sum(p.numel()
      for p in model.parameters())/1e6, 'M parameters')

input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

print(f"\nGenerating text with {temperature=}")

print("\nPrompt:", colorama.Fore.GREEN, prompt, colorama.Fore.BLUE, end='')
generated_text = []
for idx in model.generate(input, max_length, temperature=temperature):
    print(tokenizer.decode(idx[0].tolist())[-1], end='')
    generated_text.append(tokenizer.decode(idx[0].tolist()))

print(colorama.Fore.RESET, "\n\n\nInference completed.")