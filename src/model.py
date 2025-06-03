import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_hidden_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.ln2(x + self.dropout(ff_output))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, block_size, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            return logits.view(B * T, C), targets.view(B * T)

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_p=0.9):
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.block_size:
                input_ids = input_ids[:, -self.block_size:]
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for i in range(logits.size(0)):
                logits[i][sorted_indices[i][sorted_indices_to_remove[i]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

    def load_pretrained_embeddings(self, tokenizer_name='gpt2'):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        gpt2_model = GPT2Model.from_pretrained(tokenizer_name)

        with torch.no_grad():
            self.token_emb.weight[:gpt2_model.wte.weight.size(0)] = gpt2_model.wte.weight
            self.pos_emb.weight[:gpt2_model.wpe.weight.size(0)] = gpt2_model.wpe.weight
            # Optionally freeze pretrained embeddings
            self.token_emb.weight.requires_grad = False
            self.pos_emb.weight.requires_grad = False
            print("🧊 GPT-2 embeddings frozen. They can be unfrozen after a few epochs if desired.")
        print(f"✅ Loaded pretrained GPT-2 embeddings from '{tokenizer_name}'")