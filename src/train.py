import torch
from tqdm import tqdm
import torch.nn.functional as F

def train(model, train_loader, optimizer, device, scheduler=None, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        # 🔥 Unfreeze pretrained embeddings after 3 warm-up epochs
        if epoch == 5:
            model.token_emb.weight.requires_grad = True
            model.pos_emb.weight.requires_grad = True
            print("🔥 Unfroze GPT-2 embeddings")
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
            print("🔽 Lowered LR for fine-tuning")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, targets = model(input_ids, targets=labels)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)

            if torch.isnan(loss):
                print("⚠️ Skipping batch due to NaN loss")
                print("Logits contained NaNs:", torch.isnan(logits).any().item())
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")