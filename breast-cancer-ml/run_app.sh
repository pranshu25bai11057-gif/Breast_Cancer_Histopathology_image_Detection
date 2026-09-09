#!/bin/bash
# One-click startup script for macOS / Linux

# Ensure the working directory is always the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  Starting Breast Cancer Histopathology Classifier..."
echo "  Working Directory: $SCRIPT_DIR"
echo "=========================================================="

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Checking dependencies..."
pip install -r requirements.txt

# Run Streamlit Application
echo "Launching Streamlit..."
streamlit run app.py
