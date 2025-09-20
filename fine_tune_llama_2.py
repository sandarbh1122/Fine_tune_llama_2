#!/usr/bin/env python3
"""
Fine-tune Llama 2 using QLoRA

This script demonstrates how to fine-tune a Llama 2 model using QLoRA (Quantized LoRA)
for parameter-efficient fine-tuning on limited GPU resources.

Original notebook: Fine_tune_Llama_2.ipynb
"""

# Step 1: Install All the Required Packages
# Run this command in terminal or uncomment the line below:
# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

# Step 2: Import All the Required Libraries
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Step 3: Configuration Parameters
# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetune"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


def main():
    """
    Main function to execute the fine-tuning process
    """
    print("Starting Llama 2 fine-tuning with QLoRA...")
    
    # Step 4: Load everything and start the fine-tuning process
    # 1. Load dataset (you can process it here)
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    
    # 2. Configure bitsandbytes for 4-bit quantization
    print("Configuring quantization...")
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    # 3. Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load LLaMA tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    
    # Load LoRA configuration
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Set training parameters
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )
    
    # Set supervised fine-tuning parameters
    print("Setting up SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Step 5: Save trained model
    print("Saving trained model...")
    trainer.model.save_pretrained(new_model)
    
    # Step 6: Test the model
    print("Testing the fine-tuned model...")
    test_model(model, tokenizer)
    
    # Step 7: Merge and save final model
    print("Merging LoRA weights with base model...")
    merge_and_save_model(model_name, new_model, device_map)
    
    print("Fine-tuning completed successfully!")


def test_model(model, tokenizer):
    """
    Test the fine-tuned model with a sample question
    """
    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)
    
    # Run text generation pipeline with our next model
    prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print("Generated response:")
    print(result[0]['generated_text'])
    print("-" * 80)


def merge_and_save_model(model_name, new_model, device_map):
    """
    Merge LoRA weights with base model and save the final model
    """
    # Reload model in FP16 and merge it with LoRA weights
    print("Reloading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    print("Merging LoRA weights...")
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    
    # Reload tokenizer to save it
    print("Reloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Save the merged model
    print("Saving merged model...")
    model.save_pretrained(f"{new_model}_merged")
    tokenizer.save_pretrained(f"{new_model}_merged")
    
    print(f"Merged model saved as: {new_model}_merged")


def push_to_huggingface_hub(model, tokenizer, hub_name):
    """
    Push the model to Hugging Face Hub
    Note: You need to be logged in to Hugging Face CLI first
    """
    print(f"Pushing model to Hugging Face Hub as: {hub_name}")
    
    # Set locale for proper encoding
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"
    
    # Push model and tokenizer to hub
    model.push_to_hub(hub_name, check_pr=True)
    tokenizer.push_to_hub(hub_name, check_pr=True)
    
    print(f"Model successfully pushed to: https://huggingface.co/{hub_name}")


if __name__ == "__main__":
    main()
