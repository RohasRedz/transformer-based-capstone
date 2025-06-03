import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cpu', temperature=0.9, top_p=0.9):
    model.eval()
    model.to(device)

    input_ids = tokenizer(prompt, return_tensors='pt')["input_ids"].to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.block_size:
            input_ids = input_ids[:, -model.block_size:]

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature

         # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        probs = F.softmax(sorted_logits, dim=-1)

        # ✅ Safe sampling fallback
        if torch.sum(probs).item() == 0 or torch.isnan(probs).any():
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, sorted_indices.gather(-1, next_token)], dim=1)

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output