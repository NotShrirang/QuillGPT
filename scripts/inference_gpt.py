import torch
import sys
import os

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

from core.models.gpt import GPTLanguageModel
from utils.gptutils import hyperparameters, load_data
config_path = os.path.join(parent_dir, 'config/shakespearean_config.json')
data_path = os.path.join(parent_dir, 'data/input.txt')
name = "Shakespearean GPT"
train_data, val_data, vocab_size, encode, decode = load_data(data_path)
(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout, vocab_size) = hyperparameters(config_path=config_path, data_path=data_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel(
    vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, name).to(device)
state_dict = torch.load(
    os.path.join(parent_dir, 'weights/GPT_model_char.pt'),
    map_location=device)

model.load_state_dict(state_dict)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt = """Write a scene about ROMEO arguing with JULIET.
ROMEO:"""
input = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
# print(decode(model.generate(input, max_new_tokens=1000)[0].tolist()))

generated_text = []
for idx in model.generate(input, 500):
    print(decode(idx[0].tolist())[-1], end='')
    generated_text.append(decode(idx[0].tolist()))
