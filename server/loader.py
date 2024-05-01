import torch
import sys
import os

from core.models import gpt
from core.tokenizers.tokenizer import Tokenizer
from core.utils.gptutils import hyperparameters, load_data

def load_model(config_path: str = 'config/shakespearean_config.json', weights_path: str = 'weights/GPT_model_char.pt') -> tuple[gpt.GPTLanguageModel, Tokenizer]:
    """
    Load the model
    """
    tokenizer = Tokenizer()
    tokenizer.from_pretrained(config_path)
    vocab_size = tokenizer.vocab_size

    (batch_size, block_size, max_iters, eval_interval, learning_rate, device,
        eval_iters, n_embd, n_head, n_layer, dropout) = hyperparameters(config_path=config_path)
    
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = gpt.GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        device=device
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"\nLoaded weights in model.")
    print("\nModel is of ", sum(p.numel()
      for p in model.parameters())/1e6, 'M parameters')
    return model, tokenizer
