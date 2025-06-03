import sys
import torch
from transformers import AutoTokenizer

# 📂 Allow local src/ folder imports
sys.path.append('src')

# 🧠 Import custom modules
from inference import generate_text
from evaluation import evaluate
from train import train
from model import TransformerLM
from dataset import WikiTextDataset
from torch.utils.data import DataLoader

# ✅ Runtime Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer_name = 'gpt2'
block_size = 1024       # Must match GPT-2's positional embedding length
embed_dim = 768         # Must match GPT-2 hidden size
batch_size = 16
epochs = 20
learning_rate = 5e-5    # Small LR for transfer learning

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load WikiText-2 dataset using custom Dataset class
train_dataset = WikiTextDataset(split='train', tokenizer_name=tokenizer_name, block_size=block_size)
val_dataset = WikiTextDataset(split='validation', tokenizer_name=tokenizer_name, block_size=block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ✅ Build Model
model = TransformerLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=6,
    ff_hidden_dim=512,
    num_layers=4,
    block_size=block_size
).to(device)

# 🧠 Load GPT-2 pretrained embeddings and freeze them initially
model.load_pretrained_embeddings()

# 🔧 Optimizer + Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# 🎯 Train the model
train(model, train_loader, optimizer, device, scheduler, epochs=epochs)

# 📉 Evaluate final performance
evaluate(model, val_loader, device=device)

# 📝 Inference with prompt
prompt = "The scientist opened the ancient journal and found"
output = generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_p=0.92, device=device)
print("\n📘 Generated Text:")
print(output)
