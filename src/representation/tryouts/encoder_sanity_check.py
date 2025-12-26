from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel

dataset = load_dataset("ailsntua/QEvasion")
sample = dataset["train"][0]
text = f"Question: {sample['question']}\nAnswer: {sample['interview_answer']}"

# Load the tokenizer and encoder model.
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Tokenize the text
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512,
)

inputs = {k: v.to(device) for k, v in inputs.items()}

for k, v in inputs.items():
    print(k, v.shape)

with torch.no_grad():
    outputs = model(**inputs)

print(type(outputs))
print(outputs)

last_hidden_state = outputs.last_hidden_state

print("Encoder output shape:", last_hidden_state.shape)

# POOLING
attention_mask = inputs["attention_mask"]

mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
masked_embeddings = last_hidden_state * mask

sum_embeddings = masked_embeddings.sum(dim=1)
token_count = mask.sum(dim=1)

sentence_embedding = sum_embeddings / token_count

print("Final sentence embedding shape:", sentence_embedding.shape)


