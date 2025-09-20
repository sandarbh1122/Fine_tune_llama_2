#!/usr/bin/env python3
"""
Data Preprocessing Script for Llama 2 Fine-tuning

This script helps preprocess and format datasets for fine-tuning Llama 2 models.
It supports various data formats and can convert them to the required format.
"""

import json
import pandas as pd
from datasets import Dataset, load_dataset
from typing import List, Dict, Any, Optional
import argparse
import os
from pathlib import Path


def format_llama2_prompt(instruction: str, response: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt according to Llama 2 chat template
    
    Args:
        instruction (str): User instruction
        response (str): Model response
        system_prompt (str, optional): System prompt
    
    Returns:
        str: Formatted prompt
    """
    if system_prompt:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] {response} </s>"
    else:
        return f"<s>[INST] {instruction} [/INST] {response} </s>"


def process_jsonl_file(input_file: str, output_file: str, instruction_key: str = "instruction", 
                      response_key: str = "response", system_key: str = "system"):
    """
    Process a JSONL file and format it for Llama 2 training
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        instruction_key (str): Key for instruction in JSON
        response_key (str): Key for response in JSON
        system_key (str): Key for system prompt in JSON
    """
    print(f"Processing {input_file}...")
    
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                instruction = data.get(instruction_key, "")
                response = data.get(response_key, "")
                system_prompt = data.get(system_key, None)
                
                if not instruction or not response:
                    print(f"Warning: Skipping line {line_num} - missing instruction or response")
                    continue
                
                formatted_text = format_llama2_prompt(instruction, response, system_prompt)
                processed_data.append({"text": formatted_text})
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(processed_data)} examples and saved to {output_file}")


def process_csv_file(input_file: str, output_file: str, instruction_col: str = "instruction", 
                    response_col: str = "response", system_col: str = "system"):
    """
    Process a CSV file and format it for Llama 2 training
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output JSONL file
        instruction_col (str): Column name for instruction
        response_col (str): Column name for response
        system_col (str): Column name for system prompt
    """
    print(f"Processing {input_file}...")
    
    df = pd.read_csv(input_file)
    
    processed_data = []
    
    for idx, row in df.iterrows():
        instruction = str(row.get(instruction_col, "")).strip()
        response = str(row.get(response_col, "")).strip()
        system_prompt = str(row.get(system_col, "")).strip() if system_col in df.columns else None
        
        if not instruction or not response or instruction == "nan" or response == "nan":
            print(f"Warning: Skipping row {idx} - missing instruction or response")
            continue
        
        if system_prompt == "nan":
            system_prompt = None
        
        formatted_text = format_llama2_prompt(instruction, response, system_prompt)
        processed_data.append({"text": formatted_text})
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(processed_data)} examples and saved to {output_file}")


def process_huggingface_dataset(dataset_name: str, output_file: str, 
                               instruction_key: str = "instruction", 
                               response_key: str = "response", 
                               system_key: str = "system",
                               split: str = "train",
                               max_samples: Optional[int] = None):
    """
    Process a Hugging Face dataset and format it for Llama 2 training
    
    Args:
        dataset_name (str): Name of the Hugging Face dataset
        output_file (str): Path to output JSONL file
        instruction_key (str): Key for instruction in dataset
        response_key (str): Key for response in dataset
        system_key (str): Key for system prompt in dataset
        split (str): Dataset split to use
        max_samples (int, optional): Maximum number of samples to process
    """
    print(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        processed_data = []
        
        for idx, item in enumerate(dataset):
            instruction = str(item.get(instruction_key, "")).strip()
            response = str(item.get(response_key, "")).strip()
            system_prompt = str(item.get(system_key, "")).strip() if system_key in item else None
            
            if not instruction or not response or instruction == "nan" or response == "nan":
                print(f"Warning: Skipping item {idx} - missing instruction or response")
                continue
            
            if system_prompt == "nan":
                system_prompt = None
            
            formatted_text = format_llama2_prompt(instruction, response, system_prompt)
            processed_data.append({"text": formatted_text})
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Processed {len(processed_data)} examples and saved to {output_file}")
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")


def create_sample_dataset(output_file: str, num_samples: int = 100):
    """
    Create a sample dataset for testing purposes
    
    Args:
        output_file (str): Path to output JSONL file
        num_samples (int): Number of sample examples to create
    """
    print(f"Creating sample dataset with {num_samples} examples...")
    
    sample_data = [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "system": "You are a helpful AI assistant that explains technical concepts clearly."
        },
        {
            "instruction": "How do neural networks work?",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information by passing signals between layers, learning patterns through training on data.",
            "system": "You are a helpful AI assistant that explains technical concepts clearly."
        },
        {
            "instruction": "What is the difference between supervised and unsupervised learning?",
            "response": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples. Supervised learning is used for classification and regression, while unsupervised learning is used for clustering and dimensionality reduction.",
            "system": "You are a helpful AI assistant that explains technical concepts clearly."
        }
    ]
    
    # Extend the sample data
    extended_data = []
    for i in range(num_samples):
        base_sample = sample_data[i % len(sample_data)]
        extended_data.append({
            "instruction": f"{base_sample['instruction']} (Example {i+1})",
            "response": f"{base_sample['response']} This is example {i+1} of the dataset.",
            "system": base_sample['system']
        })
    
    # Format and save
    processed_data = []
    for item in extended_data:
        formatted_text = format_llama2_prompt(
            item["instruction"], 
            item["response"], 
            item["system"]
        )
        processed_data.append({"text": formatted_text})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created sample dataset with {len(processed_data)} examples and saved to {output_file}")


def main():
    """
    Main function for the preprocessing script
    """
    parser = argparse.ArgumentParser(description="Preprocess data for Llama 2 fine-tuning")
    parser.add_argument("--input_file", type=str, help="Path to input file (JSONL or CSV)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--dataset_name", type=str, help="Hugging Face dataset name")
    parser.add_argument("--file_type", type=str, choices=["jsonl", "csv"], help="Type of input file")
    parser.add_argument("--instruction_key", type=str, default="instruction", 
                       help="Key/column name for instruction")
    parser.add_argument("--response_key", type=str, default="response", 
                       help="Key/column name for response")
    parser.add_argument("--system_key", type=str, default="system", 
                       help="Key/column name for system prompt")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create a sample dataset for testing")
    parser.add_argument("--num_samples", type=int, default=100, 
                       help="Number of samples for sample dataset")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        create_sample_dataset(args.output_file, args.num_samples)
    elif args.dataset_name:
        process_huggingface_dataset(
            dataset_name=args.dataset_name,
            output_file=args.output_file,
            instruction_key=args.instruction_key,
            response_key=args.response_key,
            system_key=args.system_key,
            max_samples=args.max_samples
        )
    elif args.input_file and args.file_type:
        if args.file_type == "jsonl":
            process_jsonl_file(
                input_file=args.input_file,
                output_file=args.output_file,
                instruction_key=args.instruction_key,
                response_key=args.response_key,
                system_key=args.system_key
            )
        elif args.file_type == "csv":
            process_csv_file(
                input_file=args.input_file,
                output_file=args.output_file,
                instruction_col=args.instruction_key,
                response_col=args.response_key,
                system_col=args.system_key
            )
    else:
        print("Please specify either --create_sample, --dataset_name, or --input_file with --file_type")
        return 1
    
    print("✅ Data preprocessing completed!")
    return 0


if __name__ == "__main__":
    exit(main())
