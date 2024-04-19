# GPT from Scratch

This repository contains a custom implementation of the GPT (Generative Pre-trained Transformer) model from scratch. The GPT model is a powerful architecture for natural language processing tasks, including text generation, language translation, and more.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
    - [Training the GPT Model](#training-the-gpt-model)
    - [Using the Trained Model for Inference](#using-the-trained-model-for-inference)

## Overview
The GPT model implemented in this repository is designed for text generation tasks. It uses the Transformer architecture with self-attention mechanisms to generate coherent and contextually relevant text. The GPT architecture provided in this repository is an implementation of decoder block from [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et. al.

## Installation:
To run the training and inference scripts, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/NotShrirang/GPT-Model-from-Scratch.git
```
2. Install the required packages:
```sh
cd GPT-Model-from-Scratch
pip install -r requirements.txt
```
3. For running streamlit interface (Optional):
```sh
streamlit run app.py
```

## Usage
### Training the GPT Model:
To train the GPT model, follow these steps:

1. Set up hyperparameters in [gptutils.py](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/utils/gptutils.py) file and initialize the model.
2. Define an optimizer and train the model using the provided training script [train_gpt.py](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/scripts/train_gpt.py).
3. Save the trained model weights.

### Using the Trained Model for Inference:
After training, you can use the trained GPT model for text generation. Here's an example of using the trained model for inference:

```python
import torch
from core.models.gpt import GPTLanguageModel
from utils.gptutils import encode, decode

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel().to(device)
state_dict = torch.load(
    './weights/GPT_model_word.pt', map_location=device)

model.load_state_dict(state_dict)

prompt = "Ted said, '"
input = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

generated_text = []
for idx in model.generate(input, 500):
    print(decode(idx[0].tolist()).split()[-1], end=' ')
    generated_text.append(decode(idx[0].tolist()))
```
