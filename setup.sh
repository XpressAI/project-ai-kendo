#!/bin/bash

# Setup Script for EVF-SAM Project

echo "🔧 Setting up the EVF-SAM Project..."

# Step 1: Update Submodules (EVF-SAM)
echo "📥 Checking out the 'evf-sam' submodule..."
git submodule update --init --recursive

# Step 2: Create and Activate Virtual Environment
echo "🐍 Setting up a Python virtual environment..."
python -m venv venv

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  echo "⚙️ Activating virtual environment for Windows..."
  source venv/Scripts/activate
else
  echo "⚙️ Activating virtual environment for Linux/Mac..."
  source venv/bin/activate
fi

# Step 3: Install Requirements
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Download Model Checkpoints
echo "📥 Fetching model checkpoints from Hugging Face Hub..."
python backend/download_evf_model.py

echo "✅ Project setup complete!"
echo "Run 'source venv/bin/activate' to activate the virtual environment."
