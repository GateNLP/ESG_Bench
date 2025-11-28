import os
import gc
import torch
import transformers
import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    PeftModel
)
from trl import SFTTrainer  
import os
import yaml

# Set seed for reproducibility
np.random.seed(42)

with open("../conf/sft.yaml", "r") as f:
    cfg = yaml.safe_load(f)

base_model = cfg["base_model"]
new_model = cfg["new_model"]
train_path = cfg["train_path"]

# BitsAndBytes quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg["quant"]["load_in_4bit"],
    bnb_4bit_quant_type=cfg["quant"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, cfg["quant"]["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=cfg["quant"]["bnb_4bit_use_double_quant"]
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA Configuration (Optional for Parameter Efficient Fine-Tuning)
peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["target_modules"],
    task_type="CAUSAL_LM",
    bias="none"
)


with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)


# Convert dataset format to SFT format
def dataset_format(example):
    message = [
        {"role": "system", "content": "You are going to read a report, the report starts with [S_REPORT] and ends with [E_REPORT]. Then answer question followed by the report, the answer should be brief and do not provide any explanations. If you can't find the answer from the report, please answer not provided."},
        {"role": "user", "content": f"{example['prompt']}"},
        {"role": "assistant", "content": f"Answer:{example['chosen']}"},
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, max_length = 14000)
    return {"prompt": prompt}

def dataset_format_gemma(example):
    message = [
        {"role": "user", "content": f"system: You are going to read a report, the report starts with [S_REPORT] and ends with [E_REPORT]. Then answer question followed by the report, the answer should be brief and do not provide any explanations. If you can't find the answer from the report, please answer not provided. {example['prompt']}"},
        {"role": "assistant", "content": f"Answer:{example['chosen']}"},
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, max_length = 14000)
    return {"prompt": prompt}

# Load dataset
train_dataset = Dataset.from_list(train_data)
split_ratio = 0.1

train_valid_split = train_dataset.train_test_split(test_size=split_ratio, seed=42, shuffle=True)

if "gemma" in base_model.lower():
    train_dataset = train_valid_split["train"].map(dataset_format_gemma, num_proc=2)
    eval_dataset = train_valid_split["test"].map(dataset_format_gemma, num_proc=2)
else:
    train_dataset = train_valid_split["train"].map(dataset_format, num_proc=2)
    eval_dataset = train_valid_split["test"].map(dataset_format, num_proc=2)



print(f"Train Size: {len(train_dataset)}, Test Size: {len(eval_dataset)}")


# Training Arguments for SFT
training_args = TrainingArguments(
    output_dir= new_model + "checkpoints",  
    per_device_train_batch_size=48,  
    gradient_accumulation_steps=4,  
    optim="paged_adamw_32bit",
    num_train_epochs=10,  
    save_strategy="epoch",  
    eval_strategy="epoch",  
    learning_rate=2e-5,  
    lr_scheduler_type="cosine",
    logging_steps=10,  
    warmup_steps=100,  
    warmup_ratio=0.1,
    group_by_length=True,
    bf16=True,  
    gradient_checkpointing=True, 
    report_to="wandb"
)


# Initialize SFTTrainer (Supervised Fine-Tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=peft_config,
    max_seq_length=1500,
    tokenizer=tokenizer,
    dataset_text_field = "prompt"
)

# Train the Model
trainer.train()

# Save Fine-Tuned Model and Tokenizer
trainer.model.save_pretrained("final_sft_ckpt_QKQ")
tokenizer.save_pretrained("final_sft_ckpt_QKQ")

# Reload Model in FP16
base_model = AutoModelForCausalLM.from_pretrained(
    # "meta-llama/Llama-3.2-3B-Instruct",
    base_model,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Merge LoRA Adapters (if used)
model = PeftModel.from_pretrained(base_model, "final_sft_ckpt_QKQ")
model = model.merge_and_unload()

# Save Final Model
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

print("SFT Training Complete. Model saved successfully!")
