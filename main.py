from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from fastapi.responses import JSONResponse, StreamingResponse
from functools import cache
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
    return model, tokenizer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", status_code=status.HTTP_200_OK, tags=["Root"])
def root():
    """
    Root route

    Returns:
    dict: The message
    """
    return {"message": "GPT FastAPI Service is running."}


@app.on_event("startup")
@cache
def on_startup():
    """
    Load the model on startup
    """
    load_model()


@app.post("/generate", status_code=status.HTTP_200_OK, tags=["Generate"])
def generate(prompt: str, max_length: int = 500, temperature: float = 0.7):
    """
    Generate text from the prompt

    Args:
    prompt (str): The prompt to generate text from
    max_length (int): The maximum length of the generated text
    temperature (float): The temperature for sampling

    Returns:
    dict: The generated text
    """
    model, tokenizer = load_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generated_text = ""
    input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for idx in model.generate(input, max_length, temperature=temperature):
        generated_text += tokenizer.decode(idx[0].tolist())[-1]
    
    return JSONResponse(content={"prompt": prompt, "response": generated_text}, status_code=status.HTTP_200_OK)
    

@app.post("/generate_stream", status_code=status.HTTP_200_OK, tags=["Generate"])
def streaming_response(prompt: str, max_length: int = 500, temperature: float = 0.7):
    """
    Generate text from the prompt

    Args:
    prompt (str): The prompt to generate text from
    max_length (int): The maximum length of the generated text
    temperature (float): The temperature for sampling

    Returns:
    StreamingResponse: The generated text
    """
    model, tokenizer = load_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    def generate_text():
        for idx in model.generate(input, max_length, temperature=temperature):
            text = tokenizer.decode(idx[0].tolist())[-1]
            yield text
    return StreamingResponse(generate_text(), media_type="text/plain", status_code=status.HTTP_200_OK)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)