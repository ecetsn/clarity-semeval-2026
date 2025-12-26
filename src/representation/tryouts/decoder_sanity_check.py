import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load dataset & sample
dataset = load_dataset("ailsntua/QEvasion")
sample = dataset["train"][0]

text = f"Question: {sample['question']}\nAnswer: {sample['interview_answer']}"

print("=== RAW TEXT (truncated) ===")
print(text[:300])
print()

# 2. Load decoder
MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # required for batching

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Using device: {device}")
print()

# 3. Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512,
)

inputs = {k: v.to(device) for k, v in inputs.items()}

print("=== TOKENIZED INPUT SHAPES ===")
for k, v in inputs.items():
    print(f"{k}: {v.shape}")
print()

# 4. Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
    )

hidden_states = outputs.hidden_states[-1]

print("=== DECODER OUTPUT ===")
print("last_hidden_state shape:", hidden_states.shape)
print()

# 5. Extract representation
# Use the last token representation
last_token_embedding = hidden_states[:, -1, :]

print("=== FINAL DECODER EMBEDDING ===")
print("decoder_embedding shape:", last_token_embedding.shape)
