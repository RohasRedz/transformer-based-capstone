import math
import torch
from torch.nn import CrossEntropyLoss

@torch.no_grad()
def evaluate(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)

    loss_fn = CrossEntropyLoss()
    total_loss = 0
    count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        if torch.isnan(loss):
            print("⚠️ Skipping batch due to NaN loss")
            print("Logits contained NaNs:", torch.isnan(logits).any().item())
            continue

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    print(f"📊 Evaluation — Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity