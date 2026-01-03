#!/bin/bash

echo "Starting Simple Maintenance Text Summarizer..."
echo ""
echo "This will launch a clean and simple Streamlit web application."
echo ""
echo "To stop the application, press Ctrl+C in this terminal."
echo ""
read -p "Press Enter to continue..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo ""
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "Starting Simple Streamlit UI..."
streamlit run simple_ui.py --server.port 8501