import torch
import streamlit as st
from core.models.gpt import GPTLanguageModel
from utils.gptutils import encode, decode

torch.manual_seed(0)

st.set_page_config(layout='wide',
                   page_title='Shakespearean GPT',
                   page_icon='📜',
                   initial_sidebar_state='expanded'
                   )

st.title('📜 Shakespearean GPT')

st.subheader('Generate text in the style of Shakespeare')


def decode_text(input, model):
    for idx in model.generate(input, 500):
        text = decode(idx[0].tolist()).split()[-1] + ' '
        yield text


@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPTLanguageModel().to(device)
    state_dict = torch.load(
        './weights/GPT_model_word.pt', map_location=device)

    model.load_state_dict(state_dict)
    return model, device


model, device = load_model()

input_text = st.text_area(
    'Enter a prompt:', 'Write a scene about Romeo arguing with Juliet.\nROMEO:'
)

if st.button('Generate Text'):
    prompt = input_text
    input = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated_text = []
    st.write_stream(decode_text(input, model))
