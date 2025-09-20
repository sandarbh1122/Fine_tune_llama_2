@echo off
REM Batch script for Windows to run Llama 2 fine-tuning
REM This script provides an easy way to start the training process

echo ========================================
echo    Llama 2 Fine-tuning Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Run setup script
echo Running setup...
python setup.py
if errorlevel 1 (
    echo ERROR: Setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Starting Fine-tuning Process
echo ========================================
echo.

REM Start the fine-tuning
python fine_tune_llama_2.py

echo.
echo Training completed! Press any key to exit...
pause >nul
