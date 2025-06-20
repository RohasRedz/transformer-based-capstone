import streamlit as st
import torch
import sys

sys.path.append("src")

from model import TransformerLM
from inference import generate_text
from train import train
from evaluation import evaluate
from dataset import WikiTextDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

st.set_page_config(page_title="Capstone LLM - Q19", layout="centered")

st.title("🧠 Transformer-based LLM (Q19 - Capstone)")
st.markdown("Built from scratch in PyTorch with optional GPT-2 transfer learning")

mode = st.radio("Choose mode:", ["🔍 Inference", "🛠️ Train Model"])

# Load tokenizer and model
tokenizer_name = 'gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 1024
embed_dim = 1280 

@st.cache_resource
def load_model():
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=20,
        ff_hidden_dim=512,
        num_layers=4,
        block_size=block_size
    ).to(device)
    model.load_pretrained_embeddings()
    return model

model = load_model()

if mode == "🔍 Inference":
    st.subheader("Enter your prompt:")
    prompt = st.text_area("Prompt", value="The scientist opened the ancient journal and found", height=100)

    max_tokens = st.slider("Max New Tokens", 10, 100, 50)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.92)

    if st.button("🔮 Generate Text"):
        with st.spinner("Generating..."):
            output = generate_text(model, tokenizer, prompt, max_new_tokens=max_tokens,
                                   temperature=temperature, top_p=top_p, device=device)
            st.markdown("### 📘 Generated Output")
            st.text_area("📘 Generated Output", value=output, height=None, key="output_box")
            st.markdown("""
                <style>
                textarea[readonly] {
                    overflow-x: hidden !important;
                    resize: none;
                }
                </style>
                """, unsafe_allow_html=True)

elif mode == "🛠️ Train Model":
    st.subheader("Trigger training from Streamlit (demo)")

    epochs = st.slider("Epochs", 2, 20, 5)
    batch_size = st.slider("Batch size", 4, 64, 16)

    if st.button("🚀 Start Training"):
        with st.spinner("Loading dataset..."):
            train_data = WikiTextDataset(split='train', tokenizer_name=tokenizer_name, block_size=block_size)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        progress = st.progress(0)
        status = st.empty()

        for epoch in range(1, epochs + 1):
            status.text(f"Training Epoch {epoch}/{epochs}")
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                logits, targets = model(input_ids, targets=labels)
                loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            st.write(f"✅ Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
            progress.progress(epoch / epochs)

        st.success("Training finished!")
        st.balloons()