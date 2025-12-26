from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion")

print(dataset)

train = dataset["train"]
print(train.column_names)


example = train[0]

for k, v in example.items():
    print(f"{k}:\n{v}\n")