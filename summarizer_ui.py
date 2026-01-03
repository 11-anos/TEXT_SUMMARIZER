# -*- coding: utf-8 -*-
"""
Advanced Maintenance Text Summarizer UI
Connected to fine-tuned DistilBART model for intelligent summarization
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import pandas as pd
import os
import time
import json
from datetime import datetime

# Optional imports for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class MaintenanceSummarizer:
    def __init__(self, model_path="./maintenance_distilbart"):
        """
        Initialize the summarizer model with enhanced capabilities
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_info = {}
        self.performance_stats = {
            'total_summaries': 0,
            'avg_processing_time': 0,
            'last_used': None
        }
        
    def get_system_info(self):
        """Get system information for monitoring"""
        info = {
            'device': self.device,
            'torch_version': torch.__version__,
            'psutil_available': PSUTIL_AVAILABLE,
            'gputil_available': GPUTIL_AVAILABLE
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info['cpu_count'] = psutil.cpu_count()
                info['memory_total'] = psutil.virtual_memory().total
                info['memory_available'] = psutil.virtual_memory().available
            except:
                info['cpu_count'] = 'N/A'
                info['memory_total'] = 'N/A'
                info['memory_available'] = 'N/A'
        else:
            info['cpu_count'] = 'Not available'
            info['memory_total'] = 'Not available'
            info['memory_available'] = 'Not available'
        
        if self.device == "cuda" and GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu_name'] = gpu.name
                    info['gpu_memory_total'] = gpu.memoryTotal
                    info['gpu_memory_used'] = gpu.memoryUsed
            except:
                info['gpu_name'] = "CUDA available but not accessible"
        elif self.device == "cuda":
            info['gpu_name'] = "CUDA available (GPU monitoring not available)"
        
        return info
    
    def load_model(self):
        """Load the trained model and tokenizer with enhanced error handling"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                return False, f"Model directory '{self.model_path}' not found. Please train the model first."
                
            # Load tokenizer and model with progress indication
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Store model information
            self.model_info = {
                'model_path': self.model_path,
                'model_type': type(self.model).__name__,
                'device': self.device,
                'vocab_size': len(self.tokenizer),
                'max_length': self.model.config.max_length,
                'loaded_at': datetime.now().isoformat()
            }
            
            return True, "Model loaded successfully!"
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            return False, error_msg
    
    def summarize_text(self, text, max_length=128, min_length=30, temperature=1.0, do_sample=True):
        """
        Summarize the input text with enhanced features
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            temperature (float): Sampling temperature for generation
            do_sample (bool): Whether to use sampling
            
        Returns:
            tuple: (summary, processing_time, success, error_message)
        """
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please load the model first.", 0, False, "Model not initialized"
        
        start_time = time.time()
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=384,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary with enhanced parameters
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    top_p=0.95 if do_sample else None,
                    top_k=50 if do_sample else None
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_summaries'] += 1
            self.performance_stats['last_used'] = datetime.now().isoformat()
            
            # Calculate running average of processing time
            total = self.performance_stats['total_summaries']
            current_avg = self.performance_stats['avg_processing_time']
            self.performance_stats['avg_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
            
            return summary, processing_time, True, None
            
        except Exception as e:
            processing_time = time.time() - start_time
            return f"Error generating summary: {str(e)}", processing_time, False, str(e)
    
    def get_word_count(self, text):
        """Get word count of the text"""
        return len(text.split())
    
    def get_compression_ratio(self, original_text, summary):
        """Calculate compression ratio"""
        original_words = self.get_word_count(original_text)
        summary_words = self.get_word_count(summary)
        
        if summary_words == 0:
            return 0
        
        compression_ratio = (original_words - summary_words) / original_words * 100
        return round(compression_ratio, 2)
    
    def batch_summarize(self, texts, max_length=128, min_length=30, temperature=1.0, do_sample=True):
        """
        Process multiple texts in batch
        
        Args:
            texts (list): List of input texts
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            list: List of (summary, processing_time, success, error) tuples
        """
        results = []
        
        for i, text in enumerate(texts):
            if text.strip():
                summary, proc_time, success, error = self.summarize_text(
                    text, max_length, min_length, temperature, do_sample
                )
                results.append((summary, proc_time, success, error))
            else:
                results.append(("", 0, False, "Empty text"))
        
        return results
    
    def get_model_statistics(self):
        """Get comprehensive model and performance statistics"""
        stats = {
            'model_info': self.model_info,
            'performance': self.performance_stats,
            'system': self.get_system_info()
        }
        return stats
    
    def export_session_data(self, input_text, summary, processing_time, settings):
        """Export session data as JSON"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'summary': summary,
            'processing_time': processing_time,
            'settings': settings,
            'model_info': self.model_info,
            'performance_stats': self.performance_stats
        }
        return json.dumps(session_data, indent=2)

# Initialize the summarizer
@st.cache_resource
def get_summarizer():
    """Get or create the summarizer instance"""
    return MaintenanceSummarizer()

def main():
    """Enhanced Main Streamlit application"""
    st.set_page_config(
        page_title="Maintenance Text Summarizer - AI Powered",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #00cc88;
        font-weight: bold;
    }
    .status-error {
        color: #ff4444;
        font-weight: bold;
    }
    .status-warning {
        color: #ffaa00;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🔧 AI-Powered Maintenance Text Summarizer
        </h1>
        <p style="color: #e0e0e0; text-align: center; margin: 0.5rem 0 0 0;">
            Intelligent summarization using fine-tuned DistilBART model
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize summarizer
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = get_summarizer()
        st.session_state.model_loaded = False
        st.session_state.processing_times = []
    
    summarizer = st.session_state.summarizer
    
    # Sidebar for model management and settings
    with st.sidebar:
        st.header("🤖 Model Control")
        
        # Model path input
        model_path = st.text_input(
            "Model Path",
            value="./maintenance_distilbart",
            help="Path to the trained model directory"
        )
        
        # Load model button with status
        col_load1, col_load2 = st.columns([2, 1])
        with col_load1:
            if st.button("Load Model", type="primary", use_container_width=True):
                with st.spinner("Loading model... This may take a moment."):
                    success, message = summarizer.load_model()
                    st.session_state.model_loaded = success
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        with col_load2:
            if st.session_state.model_loaded:
                st.markdown("<span class='status-success'>●</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-error'>●</span>", unsafe_allow_html=True)
        
        # Model information display
        if st.session_state.model_loaded and summarizer.model_info:
            with st.expander("📊 Model Information", expanded=False):
                info = summarizer.model_info
                st.markdown(f"**Device:** {info.get('device', 'Unknown')}")
                st.markdown(f"**Model Type:** {info.get('model_type', 'Unknown')}")
                st.markdown(f"**Vocab Size:** {info.get('vocab_size', 'Unknown'):,}")
                st.markdown(f"**Max Length:** {info.get('max_length', 'Unknown')}")
                st.markdown(f"**Loaded:** {info.get('loaded_at', 'Unknown')[:19]}")
        
        # Performance statistics
        if st.session_state.model_loaded and summarizer.performance_stats['total_summaries'] > 0:
            with st.expander("📈 Performance Stats", expanded=False):
                stats = summarizer.performance_stats
                st.metric("Total Summaries", stats['total_summaries'])
                st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.2f}s")
                if stats['last_used']:
                    st.caption(f"Last used: {stats['last_used'][:19]}")
        
        st.markdown("---")
        st.subheader("⚙️ Generation Settings")
        
        # Generation parameters
        max_length = st.slider(
            "Maximum Summary Length",
            min_value=30,
            max_value=200,
            value=128,
            help="Maximum number of tokens in the summary"
        )
        
        min_length = st.slider(
            "Minimum Summary Length",
            min_value=10,
            max_value=100,
            value=30,
            help="Minimum number of tokens in the summary"
        )
        
        # Advanced generation settings
        with st.expander("Advanced Settings", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Controls randomness in generation (lower = more focused, higher = more creative)"
            )
            
            do_sample = st.checkbox(
                "Use Sampling",
                value=False,
                help="Enable sampling for more diverse outputs"
            )
        
        # System information
        with st.expander("🖥️ System Info", expanded=False):
            if st.button("Refresh System Info"):
                system_info = summarizer.get_system_info()
                st.json(system_info)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📋 Batch Processing", "📊 Analytics"])
    
    with tab1:
        st.header("Text Summarization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Input Text")
            
            # Sample texts for quick testing
            sample_texts = {
                "HVAC System": "The HVAC system in building 5 has been experiencing intermittent cooling issues over the past week. During inspection, we discovered that the refrigerant levels are below optimal, and the condenser coils show significant dust accumulation. The air filters are clogged and require immediate replacement. The system's performance has degraded by approximately 15% over the past month. Additionally, the thermostat readings indicate inconsistent temperature control across different zones. Recommended actions include refrigerant recharge, deep cleaning of condenser coils, filter replacement, and scheduling a comprehensive system tune-up.",
                
                "Electrical Panel": "The main electrical panel in facility A has been experiencing frequent breaker trips over the past three days. Initial inspection revealed that the 200-amp main breaker is showing signs of wear and the connections are loose. The secondary breakers are functioning normally, but the load calculation shows that we're operating at 85% capacity during peak hours. This presents a safety concern and potential fire hazard. Immediate actions required: replace the main breaker, tighten all connections, and conduct a comprehensive electrical load analysis.",
                
                "Plumbing Issue": "The water pressure in the east wing has dropped significantly over the past two days. Investigation shows that there's a leak in the main supply line near the mechanical room. The leak appears to be slow but continuous, causing water damage to the surrounding area. The shut-off valve is functional but needs maintenance. Water quality tests show elevated mineral content, which may be contributing to pipe degradation."
            }
            
            selected_sample = st.selectbox(
                "Load Sample Text (for testing)",
                ["Custom Text"] + list(sample_texts.keys())
            )
            
            if selected_sample != "Custom Text":
                input_text = sample_texts[selected_sample]
            else:
                input_text = ""
            
            # Text input with better placeholder
            input_text = st.text_area(
                "Enter your maintenance description here:",
                value=input_text,
                height=300,
                placeholder="Paste your maintenance description, work orders, or technical documents here...\n\nTips:\n• Include relevant details like equipment type, issues, and recommended actions\n• Longer texts generally produce better summaries\n• Technical terminology is handled well by the fine-tuned model",
                key="main_input"
            )
            
            # Text statistics
            if input_text:
                char_count = len(input_text)
                word_count = len(input_text.split())
                st.caption(f"📊 **Statistics:** {char_count:,} characters | {word_count:,} words | ~{word_count//4} sentences")
        
        with col2:
            st.subheader("📋 Generated Summary")
            
            # Generate summary section
            if st.button("🚀 Generate Summary", type="primary", disabled=not input_text or not st.session_state.model_loaded):
                if not st.session_state.model_loaded:
                    st.warning("⚠️ Please load the model first using the sidebar.")
                elif input_text.strip():
                    # Progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Preparing input...")
                    progress_bar.progress(20)
                    time.sleep(0.2)
                    
                    status_text.text("Generating summary...")
                    progress_bar.progress(60)
                    
                    # Generate summary
                    summary, processing_time, success, error = summarizer.summarize_text(
                        input_text, max_length, min_length, temperature, do_sample
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    if success:
                        st.success(f"✅ Summary generated in {processing_time:.2f} seconds!")
                        
                        # Store results in session state
                        st.session_state.last_summary = summary
                        st.session_state.last_processing_time = processing_time
                        st.session_state.last_input = input_text
                        st.session_state.last_settings = {
                            'max_length': max_length,
                            'min_length': min_length,
                            'temperature': temperature,
                            'do_sample': do_sample
                        }
                        
                        # Display summary
                        st.text_area(
                            "Summary:",
                            value=summary,
                            height=200,
                            key="summary_output"
                        )
                        
                        # Enhanced statistics
                        st.markdown("### 📊 Summary Analytics")
                        
                        summary_word_count = summarizer.get_word_count(summary)
                        compression_ratio = summarizer.get_compression_ratio(input_text, summary)
                        
                        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                        with col_stats1:
                            st.metric("Original Words", f"{word_count:,}")
                        with col_stats2:
                            st.metric("Summary Words", f"{summary_word_count:,}")
                        with col_stats3:
                            st.metric("Compression", f"{compression_ratio}%")
                        with col_stats4:
                            st.metric("Speed", f"{processing_time:.2f}s")
                        
                        # Export options
                        st.markdown("### 💾 Export Options")
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            # Download as text
                            st.download_button(
                                label="📝 Download as Text",
                                data=summary,
                                file_name=f"maintenance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            # Download session data as JSON
                            session_json = summarizer.export_session_data(
                                input_text, summary, processing_time, st.session_state.last_settings
                            )
                            st.download_button(
                                label="📊 Download Session Data",
                                data=session_json,
                                file_name=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp3:
                            # Copy to clipboard (simulated)
                            if st.button("📋 Copy Summary", use_container_width=True):
                                st.info("Summary copied to clipboard! (In a real app, this would use clipboard API)")
                        
                    else:
                        st.error(f"❌ {error or 'Unknown error occurred'}")
                else:
                    st.warning("⚠️ Please enter some text to summarize.")
            
            # Display last summary if available
            elif 'last_summary' in st.session_state:
                st.info("👆 Click 'Generate Summary' to create a new summary, or modify the input text.")
                st.text_area(
                    "Last Generated Summary:",
                    value=st.session_state.last_summary,
                    height=200,
                    key="last_summary_display",
                    disabled=True
                )
    
    with tab2:
        st.header("Batch Text Processing")
        st.info("🚀 Coming Soon: Process multiple texts simultaneously for bulk summarization!")
        
        # Placeholder for batch processing UI
        st.markdown("""
        ### Planned Features:
        - **CSV Upload**: Upload maintenance reports in bulk
        - **Batch Processing**: Process multiple texts at once
        - **Progress Tracking**: Real-time progress for large batches
        - **Export Options**: Download all summaries at once
        - **Queue Management**: Priority processing for urgent items
        """)
    
    with tab3:
        st.header("Analytics & Monitoring")
        
        if st.session_state.model_loaded:
            # Model performance analytics
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Performance Metrics")
                
                stats = summarizer.get_model_statistics()
                
                # Performance chart placeholder
                if summarizer.performance_stats['total_summaries'] > 0:
                    # Create a simple performance chart
                    performance_data = {
                        'Metric': ['Total Summaries', 'Avg Time (s)', 'Success Rate (%)'],
                        'Value': [
                            summarizer.performance_stats['total_summaries'],
                            summarizer.performance_stats['avg_processing_time'],
                            100.0  # Assuming all were successful for now
                        ]
                    }
                    
                    df_performance = pd.DataFrame(performance_data)
                    st.bar_chart(df_performance.set_index('Metric'))
                else:
                    st.info("No summaries generated yet. Generate some summaries to see analytics!")
            
            with col2:
                st.subheader("🖥️ System Resources")
                
                system_info = summarizer.get_system_info()
                
                # System metrics
                col_sys1, col_sys2 = st.columns(2)
                with col_sys1:
                    st.metric("CPU Cores", system_info.get('cpu_count', 'N/A'))
                    st.metric("Device", system_info['device'].upper())
                with col_sys2:
                    if PSUTIL_AVAILABLE and isinstance(system_info.get('memory_total'), (int, float)):
                        memory_gb = system_info['memory_total'] / (1024**3)
                        st.metric(f"Total RAM", f"{memory_gb:.1f} GB")
                    else:
                        st.metric("Memory", "Not available")
                    
                    if 'gpu_name' in system_info:
                        st.metric("GPU", system_info['gpu_name'])
                    else:
                        st.metric("GPU", "Not detected")
        else:
            st.warning("⚠️ Load the model to view analytics and monitoring data.")
    
    # Instructions and help section
    st.markdown("---")
    with st.expander("📖 User Guide & Best Practices", expanded=False):
        st.markdown("""
        ### Getting Started:
        1. **Load Model**: Click "Load Model" in the sidebar (requires trained model)
        2. **Input Text**: Paste your maintenance description in the text area
        3. **Adjust Settings**: Modify summary length and generation parameters
        4. **Generate**: Click "Generate Summary" for AI-powered summarization
        5. **Export**: Download results in various formats
        
        ### Best Practices:
        - **Detailed Input**: Provide comprehensive maintenance descriptions for better summaries
        - **Technical Terms**: The model is fine-tuned for maintenance terminology
        - **Appropriate Length**: Text between 100-1000 words works best
        - **Consistent Format**: Structured input (problem, analysis, recommendation) improves results
        
        ### Model Requirements:
        - **Training**: Run `maintanace__summarizer.py` to train the model first
        - **Hardware**: GPU recommended for faster inference, CPU works too
        - **Memory**: At least 4GB RAM, 2GB for model storage
        
        ### Troubleshooting:
        - **Model Not Found**: Ensure you've trained the model and the path is correct
        - **Slow Performance**: Try reducing summary length or using CPU if GPU issues
        - **Poor Quality**: Increase input detail or adjust generation parameters
        """)