# Fine-tune Llama 2 with QLoRA

This repository contains a Python script for fine-tuning Llama 2 models using QLoRA (Quantized LoRA) for parameter-efficient fine-tuning on limited GPU resources.

## Overview

The script demonstrates how to:
- Fine-tune a Llama 2 model using QLoRA
- Use 4-bit quantization to reduce memory usage
- Train on the Guanaco dataset with Llama 2 prompt template
- Merge LoRA weights with the base model
- Push the final model to Hugging Face Hub

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 15GB GPU memory for Llama 2-7B

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd fine-tuned-llama-2
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the fine-tuning script:
```bash
python fine_tune_llama_2.py
```

### Configuration

You can modify the following parameters in the script:

- `model_name`: Base model to fine-tune (default: "NousResearch/Llama-2-7b-chat-hf")
- `dataset_name`: Dataset to use for training (default: "mlabonne/guanaco-llama2-1k")
- `new_model`: Name for the fine-tuned model
- `lora_r`: LoRA rank (default: 64)
- `lora_alpha`: LoRA alpha parameter (default: 16)
- `num_train_epochs`: Number of training epochs (default: 1)

### Pushing to Hugging Face Hub

To push your fine-tuned model to Hugging Face Hub:

1. Login to Hugging Face CLI:
```bash
huggingface-cli login
```

2. Modify the `push_to_huggingface_hub` function call in the script with your desired hub name.

## Dataset Information

The script uses the `mlabonne/guanaco-llama2-1k` dataset, which contains 1,000 samples formatted for Llama 2. For the complete dataset, you can use `mlabonne/guanaco-llama2`.

## Model Architecture

- **Base Model**: Llama 2-7B-Chat
- **Quantization**: 4-bit (NF4)
- **LoRA**: Rank 64, Alpha 16
- **Training**: 1 epoch with cosine learning rate schedule

## Output

The script will generate:
- Fine-tuned LoRA weights in the `./results` directory
- Merged model in `{new_model}_merged` directory
- Training logs and metrics via TensorBoard

## Memory Requirements

- **GPU Memory**: ~15GB for Llama 2-7B with 4-bit quantization
- **RAM**: ~8GB for dataset loading and processing
- **Storage**: ~15GB for model weights and checkpoints

## Notes

- The script is optimized for Google Colab with limited resources
- For full fine-tuning, you would need significantly more GPU memory
- The model uses the Llama 2 chat template for proper formatting
- Training progress can be monitored via TensorBoard

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size` or `gradient_accumulation_steps`
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Hugging Face Hub Issues**: Check your authentication token and internet connection

## License

This project follows the same license as the base Llama 2 model. Please refer to the original model's license for details.
