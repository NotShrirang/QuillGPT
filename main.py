from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from fastapi.responses import JSONResponse, StreamingResponse
from functools import cache

from server import load_model, inference

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


@cache
def load_inference_model() -> tuple:
    model, tokenizer = load_model()
    return model, tokenizer


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
    model, tokenizer = load_inference_model()
    inference_generator = inference(model, tokenizer, prompt, max_length, temperature)
    generated_text = ""
    for generated_letter in inference_generator:
        generated_text += generated_letter
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
    inference_generator = inference(model, tokenizer, prompt, max_length, temperature)
    print("Inference complete.")
    return StreamingResponse(inference_generator, media_type="text/plain", status_code=status.HTTP_200_OK)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)