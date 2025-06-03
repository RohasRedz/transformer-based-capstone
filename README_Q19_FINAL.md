
# 🧠 Capstone Project - Q19: Transformer-based LLM in PyTorch (with Transfer Learning)

This project implements a transformer-based Language Model from scratch using PyTorch, fine-tuned using transfer learning with GPT-2 embeddings, and integrated with a user-friendly **Streamlit UI** for training and inference.

---

## 🎯 Problem Statement

> **Train a Transformer-based LLM in PyTorch, use transfer learning with pretrained GPT-2, and perform inference via prompt generation.**

---

## ✅ Features

### 🧪 Model Training
- Built using custom Transformer architecture
- Uses **WikiText-2** dataset for training
- Loads **pretrained GPT-2 token and positional embeddings**
- Embeddings are **frozen initially**, and **unfrozen after epoch 4**
- Supports learning rate scheduling and gradient clipping

### 🧠 Inference
- Uses top-p (nucleus sampling) and temperature control
- Generates completions based on user prompts
- Example prompt: `The scientist opened the ancient journal and found...`

### 🖼️ Streamlit Interface
- 📥 Prompt input + parameter sliders
- 📊 Epoch + batch size training controls
- 📈 Training progress bar (live)
- 📘 Output display with auto-expanding, scroll-free box

---

## 🗂️ Folder Structure

```
q19_transformer_llm/
├── streamlit_app.py
└── src/
    ├── main.py
    ├── model.py
    ├── train.py
    ├── evaluation.py
    ├── inference.py
    └── dataset.py
```

---

## 📦 Setup Instructions

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🧪 Training Settings

| Parameter         | Value                       |
|------------------|-----------------------------|
| Epochs           | 8 (recommended)             |
| Batch Size       | 16                          |
| Learning Rate    | 5e-5 → 1e-5 (after unfreeze)|
| Optimizer        | AdamW                       |
| Scheduler        | StepLR                      |
| Dataset          | WikiText-2                  |
| Tokenizer        | GPT-2                       |

---

## 📘 Sample Output (after training)
```
The scientist opened the ancient journal and found, theerver nationally merry of ofAppleomers...
```

> Coherence and fluency improves with better tuning, epoch adjustments, and dataset extensions.

---

## 💡 Potential Improvements
- Add support for saving/loading trained model
- Extend training with larger corpora (e.g., OpenWebText)
- Use mixed precision for faster training on GPU

---

## 🏁 Status
✅ Completed end-to-end training, inference, and UI integration for a minimal Transformer LLM using PyTorch and LangChain-compatible interfaces.
