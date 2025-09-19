#!/bin/bash

# Spend Visibility Dashboard - Setup & Run Script
echo "======================================"
echo "💰 Spend Visibility Dashboard Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Installing pip..."
    python3 -m ensurepip --default-pip
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/sample
mkdir -p config
mkdir -p logs
mkdir -p exports

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your credentials"
fi

# Check if categories.json exists
if [ ! -f "config/categories.json" ]; then
    echo "📝 Categories config not found. Please ensure config/categories.json exists."
fi

# Check CUDA availability
echo "🎮 Checking GPU availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Download FinBERT model
echo "🤖 Downloading FinBERT model (first run only)..."
python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); print('✅ Model downloaded successfully')"

# Launch the dashboard
echo ""
echo "======================================"
echo "🚀 Launching Spend Visibility Dashboard"
echo "======================================"
echo "Dashboard URL: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost