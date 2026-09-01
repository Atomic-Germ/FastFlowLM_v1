#!/bin/bash

# Setup script for Linux/Mac
# This script sets up the Python virtual environment and installs dependencies for FLM testing

set -e  # Exit on any error

echo "Setting up FLM test environment for Linux/Mac..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version (require 3.8+)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.8" | sort -V | head -n1)" != "3.8" ]]; then
    echo "Error: Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing Python packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping package installation."
fi

echo "Setup complete!"
echo ""
echo "To activate the virtual environment in future sessions, run:"
echo "source venv/bin/activate"
echo ""
echo "To run tests, make sure the FLM server is running and use:"
echo "python main.py --help"