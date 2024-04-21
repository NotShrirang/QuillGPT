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

![Streamlit Demo](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/fa888670-2c44-4f97-a07d-c58473d847d0)

## Installation:

To run the training and inference scripts, follow these steps:

1. Clone the repository:

```sh
git clone https://github.com/NotShrirang/GPT-From-Scratch.git
```

2. Install the required packages:

```sh
cd GPT-From-Scratch
pip install -r requirements.txt
```

3. For running streamlit interface (Optional):

```sh
streamlit run app.py
```

## Usage

### Training the GPT Model:

To train the GPT model, follow these steps:

1. Set up hyperparameters in [config.json](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/config.json) file and initialize the model.
2. Define an optimizer and train the model using the provided training script [train_gpt.py](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/scripts/train_gpt.py).
3. Save the trained model weights.

### For Inference:

After training, you can use the trained GPT model for text generation. Here's an example of using the trained model for inference:

```bash
python scripts/inference_gpt.py
```
