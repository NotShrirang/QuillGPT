import torch
from core.models.gpt import GPTLanguageModel
import utils.gptutils as gptutils
import tqdm

torch.manual_seed(0)

hyperparameters = gptutils.hyperparameters()
(batch_size, block_size, max_iters, eval_interval, learning_rate, device,
 eval_iters, n_embd, n_head, n_layer, dropout, vocab_size) = hyperparameters

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = gptutils.estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss \
                {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = gptutils.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(gptutils.decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), './weights/GPT_model_word.pt')
