import torch
import streamlit as st
from colorama import Fore
from core.models.gpt import GPTLanguageModel
from core.tokenizers.tokenizer import Tokenizer
from core.utils.gptutils import hyperparameters, load_data

st.set_page_config(layout='wide',
                   page_title='QuillGPT',
                   page_icon='🪶',
                   initial_sidebar_state='expanded'
                   )

def decode_text(input, model: GPTLanguageModel, max_tokens, temperature):
    for idx in model.generate(idx=input, max_new_tokens=max_tokens, max_seq_length=50, temperature=temperature):
        text = tokenizer.decode(idx[0].tolist())[-1]
        yield text

models = {
    "Shakespearean GPT": './weights/GPT_model_char.pt',
    "GPT": './weights/Harpoon_Corpus_GPT_model_word2.pt',
}

st.sidebar.header('QuillGPT')

st.sidebar.write("This app generates text using a GPT model trained on either the Harpoon corpus or Shakespearean plays.")

# Select one of the two model
model_name = st.sidebar.selectbox('Select a model:', list(models.keys()))
if model_name == "GPT":
    st.title('GPT From Scratch')
    st.write("This model was trained on the Harpoon corpus.")
else:
    st.title('Shakespearean GPT')
    st.write("This model was trained on Shakespearean plays.")

path = models[model_name]

if model_name == "GPT":
    config_path = './config/harpoon_config.json'
    data_path = './data/corpus.txt'
    name = "Harpoon GPT"
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.from_pretrained(config_path)
    vocab_size = tokenizer.vocab_size
    (batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout) = hyperparameters(config_path=config_path)

elif model_name == "Shakespearean GPT":
    config_path = './config/shakespearean_config.json'
    data_path = './data/input.txt'
    name = "Shakespearean GPT"
    tokenizer: Tokenizer = Tokenizer()
    tokenizer.from_pretrained(config_path)
    vocab_size = tokenizer.vocab_size
    (batch_size, block_size, max_iters, eval_interval, learning_rate, device,
    eval_iters, n_embd, n_head, n_layer, dropout) = hyperparameters(config_path=config_path)
    

if model_name == "GPT":
    input_text = st.text_area(
        'Enter a prompt:', 'And then Ted said, "'
    )
else:
    input_text = st.text_area(
        'Enter a prompt:', 'Write a scene about ROMEO arguing with JULIET. \nROMEO:'
    )

temperature = st.sidebar.slider('Temperature:', 0.1, 1.0, 0.5, 0.1)
max_tokens = st.sidebar.slider('Max Tokens:', 250, 1000, 500, 50)

@st.cache_resource
def load_model(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = GPTLanguageModel(
            vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, name=name
        ).to(device)
        state_dict = torch.load(
            path, map_location=device)

        model.load_state_dict(state_dict)
        return model, device
    except FileNotFoundError as e:
        st.error(f"Don't forget to download the model weights from the link in the README.md file.")
        return None, None


model, device = load_model(path)


if model:
    if st.button('Generate Text'):
        prompt = input_text
        st.subheader(model.name)
        input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated_text = []
        st.write(f":green[{prompt}]")
        st.write_stream(decode_text(input, model, max_tokens, temperature))
