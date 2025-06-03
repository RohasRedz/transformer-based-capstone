# 🧠 Capstone Project Q19 – Transformer Model with Transfer Learning (PyTorch)

This project demonstrates building a transformer-based LLM from scratch in PyTorch and fine-tuning it with transfer learning using GPT-2 embeddings.

---

## ✅ Problem Statement

Build a transformer-based language model in PyTorch. Integrate pretrained GPT-2 embeddings for transfer learning. Train on WikiText-2 dataset. Demonstrate model inference and performance evaluation.

---

## 🧩 Architecture Overview

1. **Preprocessing**
   - Dataset: WikiText-2 (raw)
   - Tokenized using GPT2TokenizerFast
   - Data chunked into `block_size` sequences
   - `WikiTextDataset` used for custom batching

2. **Model (TransformerLM)**
   - Positional and token embeddings
   - Multi-head attention layers
   - Residual + LayerNorm
   - Feed-forward layer
   - Custom transformer stack

3. **Transfer Learning**
   - Load GPT-2 embeddings (token and position)
   - Freeze embeddings for first 4 epochs
   - Unfreeze with lower LR for fine-tuning

4. **Training Logic**
   - Optimizer: AdamW
   - Learning rate scheduling
   - Gradient clipping
   - NaN-loss skipping
   - Epoch-level logging

5. **Evaluation**
   - Metrics:
     - Avg Loss
     - Perplexity
   - Skips invalid batches

6. **Inference**
   - Supports temperature and top-p sampling
   - Generates text from seed prompt

7. **UI**
   - Built with Streamlit
   - Features:
     - Training button
     - Prompt box for inference
     - Parameter sliders

---

## 🔧 Tech Stack

| Component         | Tool/Library            |
|------------------|-------------------------|
| Model Framework  | PyTorch                 |
| Pretrained Model | HuggingFace GPT-2       |
| UI               | Streamlit               |
| Dataset          | WikiText-2              |
| Tokenizer        | HuggingFace Transformers|
| Optimizer        | AdamW                   |

---

## 🎓 Viva Questions

**Q1. What is a Transformer model?**  
A deep learning model based on self-attention, used for NLP tasks like translation, QA, and generation.

**Q2. Why GPT-2 embeddings?**  
To leverage pretrained linguistic knowledge for faster convergence and improved results.

**Q3. What is block_size?**  
Defines how many tokens are processed per forward pass.

**Q4. Why use nucleus sampling?**  
To generate diverse text outputs by sampling from top-p probable tokens.

**Q5. What are frozen embeddings?**  
Embeddings with gradients disabled during early training.

**Q6. What is perplexity?**  
A measure of model's confidence. Lower values mean better performance.

**Q7. What does fine-tuning mean here?**  
Enabling full training after pretrained layers have stabilized base knowledge.

**Q8. How is training stabilized?**  
- Gradient clipping  
- Skipping NaN loss batches  
- LR scheduling  
- Transfer learning

---

## ✅ Outcomes

- Built and trained a custom transformer model
- Applied transfer learning from GPT-2
- Generated coherent, structured language
- Delivered inference via CLI and Streamlit UI