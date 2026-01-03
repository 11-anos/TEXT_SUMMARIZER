


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Dataset URL - Replace with your actual GitHub raw CSV file URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/11-anos/TEXT_SUMMARIZER/refs/heads/main/maintenance_dataset.csv"

# Load and preprocess the dataset
df = pd.read_csv(GITHUB_RAW_URL)
df = df[["long_description", "abstract"]]
df.columns = ["input_text", "target_text"]

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Initialize the model and tokenizer
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(batch):
    """Preprocess the data for training"""
    inputs = tokenizer(
        batch["input_text"],
        truncation=True,
        max_length=384
    )

    targets = tokenizer(
        batch["target_text"],
        truncation=True,
        max_length=128
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize the dataset
tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Data collator for batching
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./maintenance_distilbart",
    eval_steps=1500,
    save_steps=1500,
    logging_steps=300,
    num_train_epochs=3,
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Clear GPU cache and start training
torch.cuda.empty_cache()
trainer.train()

# Save the model and tokenizer
trainer.save_model("./maintenance_distilbart")
tokenizer.save_pretrained("./maintenance_distilbart")