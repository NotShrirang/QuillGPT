![QuillGPT-cropped-removebg-preview](https://github.com/NotShrirang/QuillGPT/assets/85283622/2e63d8ce-24f8-4bf0-835a-0c621f1d7400)

# QuillGPT

![GitHub stars](https://img.shields.io/github/stars/NotShrirang/GPT-From-Scratch?style=social)
![GitHub forks](https://img.shields.io/github/forks/NotShrirang/GPT-From-Scratch?style=social)
![GitHub issues](https://img.shields.io/github/issues/NotShrirang/GPT-From-Scratch)
![GitHub pull requests](https://img.shields.io/github/issues-pr/NotShrirang/GPT-From-Scratch)
![GitHub](https://img.shields.io/github/license/NotShrirang/GPT-From-Scratch)
![GitHub last commit](https://img.shields.io/github/last-commit/NotShrirang/GPT-From-Scratch)
![GitHub repo size](https://img.shields.io/github/repo-size/NotShrirang/GPT-From-Scratch)

QuillGPT is an implementation of the GPT decoder block based on the architecture from [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et. al. implemented in PyTorch. Additionally, this repository contains two pre-trained models—Shakespearean GPT and Harpoon GPT—along with their trained weights. For ease of experimentation and deployment, a Streamlit Playground is provided for interactive exploration of these models and FastAPI microservice implemented with Docker containerization for scalable deployment. You'll also find Python scripts for training new GPT models and performing inference on them, along with notebooks showcasing trained models. To facilitate text encoding and decoding, a simple tokenizer is implemented. Explore QuillGPT to utilize these tools and enhance your natural language processing projects!

## Table of Contents

- [Overview](#overview)
  - [Decoder Block](#the-decoder-block)
  - [Input Embeddings](#input-embeddings)
  - [Positional Embeddings](#positional-embeddings)
  - [Self-Attention](#self-attention)
- [Models](#models)
  - [Shakespearean GPT](#shakespearean-gpt)
  - [Harpoon GPT](#harpoon-gpt)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Streamlit Playground](#streamlit-playground)
  - [FastAPI Microservice](#for-running-fastapi-microservice)
  - [Running Docker Container](#for-using-containerized-version)
- [Usage](#usage)
  - [Training the GPT Model](#training-the-gpt-model)
  - [Using the Trained Model for Inference](#for-inference)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Overview

### The Decoder Block:

<img src="https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/397049a3-10cc-49b5-8696-f19806b2668e" width=350 alt="Decoder Architecture"/>

The decoder block is a crucial component of the GPT (Generative Pre-trained Transformer) model, it is where GPT actually generates the text. It leverages the self-attention mechanism to process input sequences and generate coherent outputs. Each decoder block consists of multiple layers, including self-attention layers, feed-forward neural networks, and layer normalization. The self-attention layers allow the model to weigh the importance of different words in a sequence, capturing context and dependencies regardless of their positions. This enables the GPT model to generate contextually relevant text.

### Input Embeddings:

![vector embeddings](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/29b4c375-c9f0-47b9-9d34-2a21dfdf0be8)

Input embeddings play a crucial role in transformer-based models like GPT by transforming input tokens into meaningful numerical representations. These embeddings serve as the initial input for the model, capturing semantic information about the words in the sequence. The process involves mapping each token in the input sequence to a high-dimensional vector space, where similar tokens are positioned closer together. This enables the model to understand the relationships between different words and effectively learn from the input data. The input embeddings are then fed into the subsequent layers of the model for further processing.

### Positional Embeddings:

![positional_encoding](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/90293fb0-8f20-4dc0-adba-8c31a54ef4f4)

In addition to input embeddings, positional embeddings are another vital component of transformer architectures such as GPT. Since transformers lack inherent information about the order of tokens in a sequence, positional embeddings are introduced to provide the model with positional information. These embeddings encode the position of each token within the sequence, allowing the model to distinguish between tokens based on their positions. By incorporating positional embeddings, transformers like GPT can effectively capture the sequential nature of data and generate coherent outputs that maintain the correct order of words in the generated text.

### Self-Attention:

![self attention](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/a6d785e4-ab00-4da0-a072-791f680d2bb8)

Self-attention, a fundamental mechanism in transformer-based models like GPT, operates by assigning importance scores to different words in a sequence. This process involves three key steps: calculating attention scores, applying softmax to obtain attention weights, and finally combining these weights with the input embeddings to generate contextually informed representations. At its core, self-attention allows the model to focus more on relevant words while de-emphasizing less important ones, facilitating effective learning of contextual dependencies within the input data. This mechanism is pivotal in capturing long-range dependencies and contextual nuances, enabling transformer models to generate long sequences of text.

## Models:

There are two pre-trained models and weights included in this repository.

### Shakespearean GPT
   - Parameters - 10.7 M
   - [Weights](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/weights/GPT_model_char.pt)
   - [Model Config](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/config/shakespearean_config.json)
   - Training Data contains text from Shakespearean plays. Data - [input.txt](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/data/input.txt)
   - Trained on character embeddings.
   - [Training Notebook](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/notebooks/GPT_From_Scratch_CharEmbeddings.ipynb)
   - Model trained on NVIDIA T4.
     <br> ![Training and Validation oss over training steps](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/133c5064-db26-4b3b-b5f6-95c040a7ff66)

### Harpoon GPT
   - Parameters - 226 M
   - [Weights](https://www.dropbox.com/scl/fi/vi5z3s17otn0jf7sr40po/Harpoon_Corpus_GPT_model.pt?rlkey=r7oppeslusv736fzmi908le95&st=wak0uf2t&dl=0)
   - [Model Config](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/config/config.json)
   - Trained on random text from books. Data - [corpus.txt](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/data/corpus.txt)
   - Trained on character embeddings.
   - [Training Notebook](https://github.com/NotShrirang/GPT-From-Scratch/blob/main/notebooks/GPT_From_Scratch_with_1024_char_embd.ipynb)
   - Model trained on NVIDIA A100.

## Getting Started:

### Installation:

To run the training and inference scripts, follow these steps:

1. Clone the repository:

```sh
git clone https://github.com/NotShrirang/GPT-From-Scratch.git
cd GPT-From-Scratch
```

2. Install the required packages:

```sh
pip install -r requirements.txt
```

Make sure you download the weights for Harpoon GPT from [here](https://www.dropbox.com/scl/fi/vi5z3s17otn0jf7sr40po/Harpoon_Corpus_GPT_model.pt?rlkey=r7oppeslusv736fzmi908le95&st=wak0uf2t&dl=0) before proceeding!

### Streamlit Playground:

It is hosted on Streamlit Cloud Service. You can visit it through the link [here](https://quillgpt.streamlit.app/).

[![Streamlit Demo](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/fa888670-2c44-4f97-a07d-c58473d847d0)](https://quillgpt.streamlit.app/)

```sh
streamlit run app.py
```

### For running FastAPI Microservice:
```sh
python main.py
```

### For using Containerized Version:

#### Build and Run the Docker Container with bash:
```sh
./run.sh start-dev
```

#### To stop the Docker Container, run the following command:
```sh
./run.sh stop-dev
```

## Usage

### Training the GPT Model:

To train the GPT model, follow these steps:

1. Prepare data. Put the whole text data into single .txt file and save it.
2. Write the configurations for transformer and save the file. 
<br>For example: 
    ```json
    {
      "data_path": "data/corpus.txt",
      "vocab_size": 135,
      "batch_size": 32,
      "block_size": 256,
      "max_iters": 3000,
      "eval_interval": 300,
      "learning_rate": 3e-5,
      "eval_iters": 50,
      "n_embd": 1024,
      "n_head": 12,
      "n_layer": 18,
      "dropout": 0.3,
    }
    ```

3. Train model using script `scripts/train_gpt.py`
```bash
python scripts/train_gpt.py \
        --config_path config/config.json \
        --data_path data/corpus.txt \
        --output_dir trained_models
```
(You can change the `config_path`, `data_path` and `output_dir` as per your requirements.)

4. The trained model will be saved in the `output_dir` specified in the command.

### For Inference:

After training, you can use the trained GPT model for text generation. Here's an example of using the trained model for inference:

```bash
python scripts/inference_gpt.py \
        --config_path config/shakespearean_config.json \
        --weights_path weights/GPT_model_char.pt \
        --max_length 500 \
        --prompt "Once upon a time"
```

## License
MIT © [Shrirang Mahajan](https://github.com/NotShrirang)


## Contributing
Feel free to submit pull requests, create issues, or spread the word!

## Support
Support me by simply starring this repository! ⭐
