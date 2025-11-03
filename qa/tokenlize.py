import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load prompts
prompts = np.load("./5options/qa_tensors.prompts.npy", allow_pickle=True)

# Model + tokenizer
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Batching parameters
batch_size = 30  # adjust based on your GPU memory
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(prompts), batch_size):
        print(i)
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize this batch
        inputs = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        # Forward pass to get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]   # [B, seq_len, hidden_dim]

        # Mean-pool over valid tokens
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
        emb = (hidden_states * mask).sum(1) / mask.sum(1)  # [B, hidden_dim]

        all_embeddings.append(emb.cpu())  # move to CPU to save GPU memory

# Stack all batches into one tensor
embeddings = torch.cat(all_embeddings, dim=0)  # [N, hidden_dim]
print("Final embeddings shape:", embeddings.shape)
torch.save(embeddings, f'./5options/qa_embeddings_results.pt')
