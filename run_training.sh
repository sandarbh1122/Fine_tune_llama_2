#!/bin/bash
# Shell script for Linux/Mac to run Llama 2 fine-tuning
# This script provides an easy way to start the training process

echo "========================================"
echo "    Llama 2 Fine-tuning Launcher"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python $python_version is not supported. Please use Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# Run setup script
echo "Running setup..."
python setup.py
if [ $? -ne 0 ]; then
    echo "ERROR: Setup failed"
    exit 1
fi

echo
echo "========================================"
echo "    Starting Fine-tuning Process"
echo "========================================"
echo

# Start the fine-tuning
python fine_tune_llama_2.py

echo
echo "Training completed!"
