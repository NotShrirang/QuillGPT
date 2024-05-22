# GPT from Scratch

This repository contains a custom implementation of the GPT (Generative Pre-trained Transformer) model from scratch. The GPT model is a powerful architecture for natural language processing tasks, including text generation, language translation, and more.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
  - [Shakespearean GPT](#shakespearean-gpt)
  - [Harpoon GPT](#harpoon-gpt)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the GPT Model](#training-the-gpt-model)
  - [Using the Trained Model for Inference](#for-inference)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Overview

The GPT model implemented in this repository is designed for text generation tasks. It uses the Transformer architecture with self-attention mechanisms to generate text. The GPT architecture provided in this repository is an implementation of decoder block from [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et. al.

![Streamlit Demo](https://github.com/NotShrirang/GPT-From-Scratch/assets/85283622/fa888670-2c44-4f97-a07d-c58473d847d0)

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

### For running Streamlit interface:

```sh
streamlit run app.py
```

### For running FastAPI Microservice:
```sh
python main.py
```

### Build and Run the Docker Container with bash:
```sh
./run.sh start-dev
```

### To stop the Docker Container, run the following command:
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
