import torch
import torch.optim as optim
from model import BigramLanguageModel, device, block_size, batch_size, max_iters, eval_interval, learning_rate, eval_iters
from data import load_data, Tokenizer, get_batch
import os

# 1. Load your dataset
DATA_PATH = 'data_instruct.txt'
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit()

text = load_data(DATA_PATH)
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# 2. Train/Val splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # First 90% is train, rest val
train_data = data[:n]
val_data = data[n:]

# 3. Model & Optimizer
model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data if split == 'train' else val_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 4. Training Loop
print(f"Starting training on {device}...")
for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch(train_data, block_size, batch_size, device)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 5. Save Model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'chars': tokenizer.chars,
    'vocab_size': vocab_size
}
torch.save(checkpoint, 'model_ckpt.pt')
print("Model saved to model_ckpt.pt")

# Test generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nSample Generation:")
print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))
