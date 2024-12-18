#!/bin/bash

# Setup Script for EVF-SAM Project
set -e  # Exit on error

echo "🔧 Setting up the EVF-SAM Project..."

# Step 1: Update Submodules (EVF-SAM)
echo "📥 Checking out the 'evf-sam' submodule..."
git submodule update --init --recursive

# Step 2: Install Requirements
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Step 3: Download Model Checkpoints
echo "📥 Fetching model checkpoints from Hugging Face Hub..."
python backend/download_model.py

echo "✅ Project setup complete!"
echo "Run 'source venv/bin/activate' to activate the virtual environment."
