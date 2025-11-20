#!/bin/bash

# Sales RAG System Setup Script
# This script sets up your development environment

set -e  # Exit on any error

echo "🚀 Sales RAG System Setup"
echo "=========================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt --break-system-packages

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your GEMINI_API_KEY"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p chroma_db
mkdir -p logs
mkdir -p outputs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GEMINI_API_KEY"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python app.py"
echo ""
