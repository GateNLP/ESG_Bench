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
import yaml
from functools import partial
np.random.seed(42)


with open("../conf/cot.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load Model and Tokenizer

base_model_name = cfg["model"]["name"]
dataset_name = cfg["data"]["name"]
raw_cot_path = cfg["data"]["path"]

# Output dirs
checkpoint_dir   = cfg["training"]["checkpoint_dir"]
final_save_file  = cfg["saving"]["final_dir"]
new_model        = cfg["saving"]["merged_model_dir"]

# BitsAndBytes quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA Configuration (Optional for Parameter Efficient Fine-Tuning)
peft_config = LoraConfig(
    r=16,  
    lora_alpha=16,  
    lora_dropout=0.2,  
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

with open(raw_cot_path, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]
with open("converted_data.json", "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=2, ensure_ascii=False)


# Convert dataset format to SFT format
def dataset_format(example, base_model):
    if "gemma" in base_model.lower():
        message = [m for m in example["messages"] if m["role"] != "system"]
    else:    
        message = example["messages"]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, return_tensors="pt", max_length = 14000)
    return {"prompt": prompt}


# Load dataset
train_dataset = Dataset.from_list(raw_data)
split_ratio = 0.1

train_valid_split = train_dataset.train_test_split(test_size=split_ratio, seed=42, shuffle=True)

format_func = partial(dataset_format, base_model=base_model_name)
train_dataset = train_valid_split["train"].map(format_func, num_proc=2)
eval_dataset = train_valid_split["test"].map(format_func, num_proc=2)



print(f"Train Size: {len(train_dataset)}, Test Size: {len(eval_dataset)}")


# Training Arguments for SFT
training_args = TrainingArguments(
    output_dir= new_model + "checkpoints",  
    per_device_train_batch_size=12,  
    gradient_accumulation_steps=4,  
    optim="paged_adamw_32bit",
    num_train_epochs=10,  
    save_strategy="epoch",  
    # evaluation_strategy="epoch",  
    eval_strategy="epoch", 
    learning_rate=2e-5,  
    lr_scheduler_type="cosine",
    logging_steps=10,  
    warmup_steps=100,  
    warmup_ratio=0.1,
    group_by_length=True,
    # bf16=True,  
    fp16=True,
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
trainer.model.save_pretrained(final_save_file)
tokenizer.save_pretrained(final_save_file)

# Cleanup Memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload Model in FP16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Merge LoRA Adapters (if used)
model = PeftModel.from_pretrained(base_model, final_save_file)
model = model.merge_and_unload()

# Save Final Model
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

print(" SFT Training Complete. Model saved successfully!")
