import torch

def inference(model, tokenizer, prompt, max_length=500, temperature=0.7):
    print("\nRunning inference...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for idx in model.generate(input, max_length, temperature=temperature):
        generated_text = tokenizer.decode(idx[0].tolist())[-1]
        yield generated_text