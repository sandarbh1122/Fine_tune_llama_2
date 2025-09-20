#!/usr/bin/env python3
"""
Inference Script for Fine-tuned Llama 2 Model

This script provides an easy way to run inference with your fine-tuned Llama 2 model.
It can load either the LoRA weights or the merged model for text generation.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging
)
from peft import PeftModel
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.set_verbosity(logging.CRITICAL)


def load_model_and_tokenizer(model_path, use_4bit=True, use_lora=True, base_model_name=None):
    """
    Load the fine-tuned model and tokenizer
    
    Args:
        model_path (str): Path to the fine-tuned model
        use_4bit (bool): Whether to use 4-bit quantization
        use_lora (bool): Whether to load LoRA weights or merged model
        base_model_name (str): Base model name for LoRA loading
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {'LoRA' if use_lora else 'merged'} model from: {model_path}")
    
    # Configure quantization if needed
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None
    
    # Load base model
    if use_lora and base_model_name:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if use_4bit else torch.float32
        )
        
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if use_4bit else torch.float32
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name if use_lora and base_model_name else model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def format_prompt(user_prompt, system_prompt=None):
    """
    Format the prompt according to Llama 2 chat template
    
    Args:
        user_prompt (str): User's question or prompt
        system_prompt (str, optional): System prompt to guide the model
    
    Returns:
        str: Formatted prompt
    """
    if system_prompt:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    else:
        return f"<s>[INST] {user_prompt} [/INST]"


def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """
    Generate a response using the fine-tuned model
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt (str): Input prompt
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
    
    Returns:
        str: Generated response
    """
    # Create text generation pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Generate response
    result = pipe(prompt)
    return result[0]['generated_text']


def interactive_chat(model, tokenizer, system_prompt=None):
    """
    Start an interactive chat session with the model
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        system_prompt (str, optional): System prompt to guide the model
    """
    print("\n" + "="*60)
    print("🤖 Fine-tuned Llama 2 Chat")
    print("="*60)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    if system_prompt:
        print(f"System: {system_prompt}")
    print("="*60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared!")
                continue
            elif not user_input:
                continue
            
            # Format the prompt
            formatted_prompt = format_prompt(user_input, system_prompt)
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Generate response
            response = generate_response(model, tokenizer, formatted_prompt)
            
            # Extract only the assistant's response (remove the prompt)
            if "[/INST]" in response:
                assistant_response = response.split("[/INST]")[-1].strip()
            else:
                assistant_response = response[len(formatted_prompt):].strip()
            
            print(assistant_response)
            
            # Store in conversation history
            conversation_history.append({"user": user_input, "assistant": assistant_response})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


def main():
    """
    Main function to run the inference script
    """
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama 2 model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-chat-hf",
                       help="Base model name (required for LoRA models)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA weights instead of merged model")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt to guide the model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt to process (non-interactive mode)")
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            use_4bit=args.use_4bit,
            use_lora=args.use_lora,
            base_model_name=args.base_model
        )
        
        print("✅ Model loaded successfully!")
        
        if args.interactive:
            # Interactive chat mode
            interactive_chat(model, tokenizer, args.system_prompt)
        elif args.prompt:
            # Single prompt mode
            formatted_prompt = format_prompt(args.prompt, args.system_prompt)
            response = generate_response(
                model, tokenizer, formatted_prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            print("\n" + "="*60)
            print("🤖 Generated Response:")
            print("="*60)
            if "[/INST]" in response:
                assistant_response = response.split("[/INST]")[-1].strip()
            else:
                assistant_response = response[len(formatted_prompt):].strip()
            print(assistant_response)
        else:
            print("Please specify either --interactive or --prompt")
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
