#  Maintenance Text Summarizer

A web-based application for summarizing maintenance descriptions and technical documents using a fine-tuned DistilBART model.

##  Quick Start

### Option 1: Simple UI (Recommended for First Time)

**Windows:**
```bash
# Double-click run_simple_ui.bat or run in Command Prompt
run_simple_ui.bat
```

**Linux/Mac:**
```bash
# Make executable and run
chmod +x run_simple_ui.sh
./run_simple_ui.sh
```

### Option 2: Enhanced UI (Advanced Features)

**Windows:**
```bash
# Double-click run_ui.bat or run in Command Prompt
run_ui.bat
```

**Linux/Mac:**
```bash
# Make executable and run
chmod +x run_ui.sh
./run_ui.sh
```

### Option 3: Manual Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (if not already done):**
   ```bash
   python maintanace__summarizer.py
   ```

3. **Launch the UI:**
   ```bash
   # For Simple UI:
   streamlit run simple_ui.py --server.port 8501
   
   # For Enhanced UI:
   streamlit run summarizer_ui.py --server.port 8501
   ```

4. **Open your browser and go to:** `http://localhost:8501`

##  Features

### Simple UI (`simple_ui.py`)
- **Clean Interface**: Straightforward, easy-to-use design
- **Fast Loading**: Minimal dependencies, quick startup
- **Core Functionality**: Text input, summarization, download
- **Model Management**: Simple model loading with status indicators
- **Basic Statistics**: Word count and compression metrics
- **Sample Texts**: Pre-loaded examples for quick testing

### Enhanced UI (`summarizer_ui.py`)
- **Advanced AI**: Multiple generation parameters and controls
- **Real-time Monitoring**: System resource tracking and performance analytics
- **Session Management**: Processing history and performance statistics
- **Multiple Export Formats**: Text files, JSON session data
- **Batch Processing Ready**: Infrastructure for bulk text processing
- **System Integration**: GPU/CPU detection and comprehensive monitoring

### Common Features (Both UIs)
- **AI-Powered Summarization**: Fine-tuned DistilBART model
- **Responsive Design**: Works across different screen sizes
- **Optimized Performance**: Cached model loading and efficient inference
- **Robust Error Handling**: Comprehensive error management and user feedback

## Project Structure

```
maintenance_summarizer/
├── maintanace__summarizer.py    # Model training script
├── simple_ui.py                 # Simple, clean UI (recommended)
├── summarizer_ui.py             # Enhanced UI with advanced features
├── requirements.txt             # Python dependencies
├── run_simple_ui.bat            # Windows launcher for simple UI
├── run_simple_ui.sh             # Linux/Mac launcher for simple UI
├── run_ui.bat                   # Windows launcher for enhanced UI
├── run_ui.sh                    # Linux/Mac launcher for enhanced UI
└── README.md                    # This documentation file
```

## Which UI Should I Use?

### Choose Simple UI (`simple_ui.py`) if you:
- Want a quick, straightforward experience
- Prefer clean, minimal interface
- Don't need advanced monitoring features
- Want faster startup and fewer dependencies
- Are new to the application

### Choose Enhanced UI (`summarizer_ui.py`) if you:
- Want advanced generation controls
- Need system monitoring and analytics
- Plan to process many documents
- Want detailed performance tracking
- Need multiple export formats

##  Usage Guide

### Simple UI Usage
1. **Load Model**: Click "Load Model" in the sidebar
2. **Select Sample**: Choose from pre-written examples or enter custom text
3. **Adjust Settings**: Use sliders for summary length
4. **Generate**: Click "Generate Summary"
5. **Download**: Save the summary as text file

### Enhanced UI Usage
1. **Load Model**: Click "Load Model" in the sidebar
2. **Choose Tab**: Single Text, Batch Processing, or Analytics
3. **Advanced Settings**: Configure temperature, sampling, etc.
4. **Monitor Performance**: View system resources and statistics
5. **Export Options**: Multiple format choices available

### 1. Model Training
Before using the UI, you need to train the model:

1. Prepare your maintenance dataset (CSV format with "long_description" and "abstract" columns)
2. Update the `GITHUB_RAW_URL` in `maintanace__summarizer.py`
3. Run the training script:
   ```bash
   python maintanace__summarizer.py
   ```

### 2. Using the Enhanced UI

#### **Single Text Processing (Main Tab)**
1. **Load Model**: Click "Load Model" in the sidebar - you'll see a green status indicator when ready
2. **Quick Testing**: Use the "Load Sample Text" dropdown to test with pre-written examples
3. **Input Text**: Paste maintenance descriptions in the large text area
4. **Adjust Settings**: 
   - **Basic**: Summary length sliders (30-200 tokens)
   - **Advanced**: Temperature (0.1-2.0), sampling options
5. **Generate Summary**: Click the prominent "Generate Summary" button
6. **View Results**: See the AI-generated summary with detailed analytics
7. **Export Options**: 
   - Download as text file
   - Export complete session data as JSON
   - Copy to clipboard (simulated)

#### **Performance Monitoring (Analytics Tab)**
- **Model Information**: Device, model type, vocabulary size
- **Performance Metrics**: Processing times, success rates
- **System Resources**: CPU usage, memory, GPU status
- **Real-time Updates**: Refresh system info on demand

#### **Batch Processing (Coming Soon)**
- Upload multiple maintenance reports
- Process in bulk with progress tracking
- Export all summaries at once

### 3. Settings

- **Maximum Summary Length**: Control the maximum tokens in summaries (30-200)
- **Minimum Summary Length**: Set the minimum tokens for summaries (10-100)
- **Model Path**: Specify custom model directory path

## 🔧 Configuration

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- 4GB+ RAM
- 2GB+ disk space for model

### Dependencies
```
streamlit>=1.28.0      # Web UI framework
torch>=2.0.0           # Deep learning framework
transformers>=4.30.0   # Hugging Face transformers
datasets>=2.12.0       # Dataset utilities
pandas>=1.5.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
psutil>=5.9.0          # System monitoring
GPUtil>=1.4.0          # GPU monitoring
```

### Environment Variables
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

##  Model Details

- **Base Model**: sshleifer/distilbart-cnn-12-6
- **Input Length**: 384 tokens
- **Output Length**: 128 tokens (configurable)
- **Training**: Fine-tuned on maintenance data
- **Device**: Auto-detects GPU/CPU

## 🐛 Enhanced Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure you've trained the model first
   - Check that the model path is correct
   - Verify the `maintenance_distilbart` directory exists
   - Use the model information panel in the sidebar to debug

2. **CUDA errors**
   - Install PyTorch with CUDA support
   - Set `PYTORCH_CUDA_ALLOC_CONF` environment variable
   - Try running on CPU if GPU issues persist
   - Check GPU status in the Analytics tab

3. **Memory issues**
   - Monitor memory usage in the Analytics tab
   - Reduce batch size in training arguments
   - Use gradient accumulation during training
   - Enable mixed precision training (fp16)
   - Close other applications to free memory

4. **Streamlit connection issues**
   - Check firewall settings
   - Try different port: `streamlit run summarizer_ui.py --server.port 8502`
   - Clear browser cache
   - Restart the Streamlit application

5. **Performance issues**
   - Check system resources in the Analytics tab
   - Monitor processing times in performance statistics
   - Ensure adequate RAM and GPU memory
   - Adjust generation parameters for faster processing

### Enhanced Performance Tips

- **GPU Acceleration**: Use CUDA-enabled GPU for faster inference (monitor in Analytics tab)
- **Model Caching**: The UI caches the model for faster subsequent loads
- **Batch Processing**: Process multiple texts efficiently (infrastructure ready)
- **Memory Management**: Clear GPU cache and monitor system resources
- **Parameter Tuning**: Adjust temperature and sampling for desired output quality vs speed
- **Session Tracking**: Monitor your usage patterns and performance over time

## Example Usage

```
Input Text:
"The HVAC system in building 5 has been experiencing intermittent cooling issues. 
During inspection, we discovered that the refrigerant levels are below optimal, 
and the condenser coils show signs of dust accumulation. Additionally, the air 
filters are clogged and require immediate replacement. The system's performance 
has degraded by approximately 15% over the past month. Recommended actions include: 
refrigerant recharge, deep cleaning of condenser coils, filter replacement, and 
scheduling a comprehensive system tune-up."

Generated Summary:
"The HVAC system in building 5 shows cooling issues due to low refrigerant levels, 
dusty condenser coils, and clogged air filters. Performance has dropped 15%. 
Recommended actions: refrigerant recharge, clean coils, replace filters, and 
schedule comprehensive tune-up."
```

## Development

### Adding Features

1. **Custom Models**: Modify `MaintenanceSummarizer` class to use different models
2. **UI Enhancements**: Add new components in the Streamlit interface
3. **Batch Processing**: Implement multi-text summarization
4. **Export Options**: Add PDF, DOCX export capabilities

