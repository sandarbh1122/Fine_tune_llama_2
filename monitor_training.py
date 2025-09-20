#!/usr/bin/env python3
"""
Training Monitoring Script for Llama 2 Fine-tuning

This script helps monitor the training progress by reading TensorBoard logs
and providing real-time updates on training metrics.
"""

import os
import time
import argparse
from pathlib import Path
import subprocess
import webbrowser
from datetime import datetime


def check_tensorboard_logs(log_dir):
    """
    Check if TensorBoard logs exist in the specified directory
    
    Args:
        log_dir (str): Directory containing TensorBoard logs
    
    Returns:
        bool: True if logs exist, False otherwise
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return False
    
    # Look for TensorBoard event files
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    return len(event_files) > 0


def start_tensorboard(log_dir, port=6006):
    """
    Start TensorBoard server
    
    Args:
        log_dir (str): Directory containing TensorBoard logs
        port (int): Port to run TensorBoard on
    """
    print(f"Starting TensorBoard on port {port}...")
    print(f"Log directory: {log_dir}")
    print(f"Access TensorBoard at: http://localhost:{port}")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        # Start TensorBoard
        subprocess.run([
            "tensorboard", 
            "--logdir", log_dir, 
            "--port", str(port),
            "--host", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("❌ TensorBoard not found. Please install it with: pip install tensorboard")
    except Exception as e:
        print(f"❌ Error starting TensorBoard: {str(e)}")


def monitor_training_files(results_dir):
    """
    Monitor training files and provide status updates
    
    Args:
        results_dir (str): Directory containing training results
    """
    results_path = Path(results_dir)
    
    print(f"Monitoring training files in: {results_dir}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    try:
        while True:
            # Check for checkpoint directories
            checkpoint_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
            
            # Check for log files
            log_files = list(results_path.glob("**/*.log"))
            
            # Check for model files
            model_files = list(results_path.glob("**/*.bin")) + list(results_path.glob("**/*.safetensors"))
            
            # Display status
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Training Status:")
            print(f"  📁 Checkpoints: {len(checkpoint_dirs)}")
            print(f"  📄 Log files: {len(log_files)}")
            print(f"  🧠 Model files: {len(model_files)}")
            
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
                print(f"  🕒 Latest checkpoint: {latest_checkpoint.name}")
            
            # Check if training is still active (look for recent file modifications)
            recent_files = []
            for file_path in results_path.rglob("*"):
                if file_path.is_file():
                    # Check if file was modified in the last 5 minutes
                    if (time.time() - file_path.stat().st_mtime) < 300:
                        recent_files.append(file_path)
            
            if recent_files:
                print(f"  🔄 Active training detected ({len(recent_files)} recent files)")
            else:
                print("  ⏸️  No recent activity")
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def check_gpu_usage():
    """
    Check GPU usage and memory
    """
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                               "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("\n🖥️  GPU Status:")
            for i, line in enumerate(lines):
                gpu_util, mem_used, mem_total = line.split(', ')
                mem_percent = (int(mem_used) / int(mem_total)) * 100
                print(f"  GPU {i}: {gpu_util}% utilization, {mem_used}/{mem_total}MB ({mem_percent:.1f}%)")
        else:
            print("❌ Could not get GPU information (nvidia-smi not found)")
    except Exception as e:
        print(f"❌ Error checking GPU: {str(e)}")


def main():
    """
    Main function for the monitoring script
    """
    parser = argparse.ArgumentParser(description="Monitor Llama 2 fine-tuning progress")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory containing training results")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Start TensorBoard server")
    parser.add_argument("--port", type=int, default=6006,
                       help="Port for TensorBoard server")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor training files")
    parser.add_argument("--gpu", action="store_true",
                       help="Check GPU usage")
    parser.add_argument("--open_browser", action="store_true",
                       help="Open TensorBoard in browser automatically")
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        print("Make sure you're running this from the correct directory and training has started.")
        return 1
    
    # Check GPU usage if requested
    if args.gpu:
        check_gpu_usage()
    
    # Start TensorBoard if requested
    if args.tensorboard:
        if check_tensorboard_logs(args.results_dir):
            if args.open_browser:
                # Open browser after a short delay
                def open_browser():
                    time.sleep(2)
                    webbrowser.open(f"http://localhost:{args.port}")
                
                import threading
                threading.Thread(target=open_browser, daemon=True).start()
            
            start_tensorboard(args.results_dir, args.port)
        else:
            print(f"❌ No TensorBoard logs found in {args.results_dir}")
            print("Make sure training has started and is generating logs.")
            return 1
    
    # Monitor training files if requested
    if args.monitor:
        monitor_training_files(args.results_dir)
    
    # If no specific action requested, show help
    if not any([args.tensorboard, args.monitor, args.gpu]):
        print("Please specify an action:")
        print("  --tensorboard    Start TensorBoard server")
        print("  --monitor        Monitor training files")
        print("  --gpu            Check GPU usage")
        print("  --help           Show this help message")
    
    return 0


if __name__ == "__main__":
    exit(main())
