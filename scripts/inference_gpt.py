import torch
from core.models.gpt import GPTLanguageModel
from utils.gptutils import encode, decode

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel().to(device)
state_dict = torch.load(
    './weights/GPT_model_word.pt', map_location=device)

model.load_state_dict(state_dict)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt = """Write a scene about Romeo arguing with Juliet.
ROMEO:"""
input = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
# print(decode(model.generate(input, max_new_tokens=1000)[0].tolist()))

generated_text = []
for idx in model.generate(input, 500):
    print(decode(idx[0].tolist()).split()[-1], end=' ')
    generated_text.append(decode(idx[0].tolist()))
