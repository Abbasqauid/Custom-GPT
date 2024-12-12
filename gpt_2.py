import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Paths and configurations
output_dir = "./fine_tuned_models"
model_name = "gpt2"

# Load the dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")

# Check available splits
print(dataset)

# Use only 'train' and 'test', create 'validation' if not available
if "validation" not in dataset:
    dataset_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset = {
        "train": dataset_split["train"].shuffle(seed=42).select(range(min(1000, len(dataset_split["train"])))) ,
        "validation": dataset_split["test"].shuffle(seed=42).select(range(min(500, len(dataset_split["test"])))) ,
        "test": dataset_split["train"].shuffle(seed=42).select(range(min(500, len(dataset_split["train"])))) ,
    }
else:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(500))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(500))

# Preprocessing and filtering
def preprocess_function(examples):
    question = examples["question"] if isinstance(examples["question"], str) else examples["question"].get("text", "")
    context = examples["context"] if isinstance(examples["context"], str) else examples["context"].get("text", "")
    long_answer = examples["long_answer"] if isinstance(examples["long_answer"], str) else examples["long_answer"].get("text", "")
    return {"text": question + " " + context + " " + long_answer}

# Apply preprocessing
dataset = {split: data.map(preprocess_function, remove_columns=["question", "context", "long_answer"]) for split, data in dataset.items()}

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Fine-tune the model
print(f"Fine-tuning {model_name}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Freeze most layers for efficient training
for param in model.base_model.parameters():
    param.requires_grad = False

# Tokenize dataset
tokenized_datasets = {split: data.map(tokenize_function, batched=True) for split, data in dataset.items()}

# Training configuration
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, model_name.split("/")[-1]),
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train and save model
trainer.train()
model.save_pretrained(os.path.join(output_dir, model_name.split("/")[-1]))
tokenizer.save_pretrained(os.path.join(output_dir, model_name.split("/")[-1]))

print("Fine-tuning complete for GPT-2.")
