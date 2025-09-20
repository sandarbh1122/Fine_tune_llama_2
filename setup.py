#!/usr/bin/env python3
"""
Setup script for Llama 2 Fine-tuning Project

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """
    Run a command and handle errors
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """
    Check if Python version is compatible
    """
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True


def check_cuda():
    """
    Check if CUDA is available
    """
    print("🖥️  Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available with {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA is not available. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. Will check after installation.")
        return None


def install_dependencies():
    """
    Install required dependencies
    """
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def create_directories():
    """
    Create necessary directories
    """
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "results",
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    return True


def check_huggingface_login():
    """
    Check if user is logged in to Hugging Face
    """
    print("🤗 Checking Hugging Face authentication...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return True
    except Exception:
        print("⚠️  Not logged in to Hugging Face Hub")
        print("   To push models to the hub, run: huggingface-cli login")
        return False


def main():
    """
    Main setup function
    """
    print("🚀 Setting up Llama 2 Fine-tuning Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Check CUDA after installation
    cuda_available = check_cuda()
    
    # Check Hugging Face login
    hf_logged_in = check_huggingface_login()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("=" * 50)
    
    print("\n📋 Next steps:")
    print("1. If you want to push models to Hugging Face Hub:")
    print("   huggingface-cli login")
    print("\n2. Start fine-tuning:")
    print("   python fine_tune_llama_2.py")
    print("\n3. Monitor training (in another terminal):")
    print("   python monitor_training.py --tensorboard")
    print("\n4. Run inference after training:")
    print("   python inference_script.py --model_path ./results --use_lora --interactive")
    
    if not cuda_available:
        print("\n⚠️  Note: CUDA not available. Training will be slower on CPU.")
    
    if not hf_logged_in:
        print("\n⚠️  Note: Not logged in to Hugging Face. You won't be able to push models to the hub.")
    
    return 0


if __name__ == "__main__":
    exit(main())
