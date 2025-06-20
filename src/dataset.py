
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast

class WikiTextDataset(Dataset):
    def __init__(self, split='train', tokenizer_name='gpt2-large', block_size=128):
        # Load the dataset split (train/validation/test)
        raw_dataset = load_dataset("openwebtext", split=split, trust_remote_code=True)

        # Load tokenizer (you can later switch to a custom one)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad_token is set

        # Tokenize all text into one long string
        tokens = self.tokenizer("\n\n".join(raw_dataset['text']), return_tensors=None, return_attention_mask=False, add_special_tokens=True, truncation=True, max_length=block_size * 100)["input_ids"]
        tokens = tokens[:200_000]  # Limit to 200k tokens for faster training
        
        # Chunk the token stream into blocks of fixed length
        self.block_size = block_size
        self.examples = [
            tokens[i : i + block_size]
            for i in range(0, len(tokens) - block_size, block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {
            "input_ids": torch.tensor(x[:-1], dtype=torch.long),
            "labels": torch.tensor(x[1:], dtype=torch.long)
        }
