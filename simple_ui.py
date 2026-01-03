#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Maintenance Text Summarizer UI
Clean and straightforward interface for text summarization
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time

# Page configuration
st.set_page_config(
    page_title="Maintenance Text Summarizer",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>🔧 Maintenance Text Summarizer</h1>
    <p style="color: #e0e0e0; margin: 0;">AI-Powered Text Summarization for Maintenance Reports</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    model_path = "./maintenance_distilbart"
    
    if not os.path.exists(model_path):
        return None, f"Model directory '{model_path}' not found. Please train the model first."
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return (model, tokenizer, device), "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def summarize_text(model_data, text, max_length=128, min_length=30):
    """Generate summary from text"""
    if model_data is None:
        return "Model not loaded"
    
    model, tokenizer, device = model_data
    
    try:
        # Prepare input
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=384,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode summary
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_word_count(text):
    """Get word count of text"""
    return len(text.split())

def get_compression_ratio(original_text, summary):
    """Calculate compression ratio"""
    original_words = get_word_count(original_text)
    summary_words = get_word_count(summary)
    
    if summary_words == 0:
        return 0
    
    compression_ratio = (original_words - summary_words) / original_words * 100
    return round(compression_ratio, 2)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Model loading
    st.subheader("🤖 Model")
    
    if st.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            model_data, message = load_model()
            if model_data:
                st.session_state.model_data = model_data
                st.session_state.model_loaded = True
                st.markdown(f'<div class="status-box success-box">✅ {message}</div>', unsafe_allow_html=True)
            else:
                st.session_state.model_loaded = False
                st.markdown(f'<div class="status-box error-box">❌ {message}</div>', unsafe_allow_html=True)
    
    # Model status
    if st.session_state.get('model_loaded', False):
        st.markdown('<div class="status-box success-box">● Model Loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box error-box">● Model Not Loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generation settings
    st.subheader("📝 Summary Settings")
    
    max_length = st.slider(
        "Maximum Length",
        min_value=30,
        max_value=200,
        value=128,
        help="Maximum number of tokens in summary"
    )
    
    min_length = st.slider(
        "Minimum Length", 
        min_value=10,
        max_value=100,
        value=30,
        help="Minimum number of tokens in summary"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Input Text")
    
    # Sample texts for testing
    sample_texts = {
        "HVAC System": """The HVAC system in building 5 has been experiencing intermittent cooling issues over the past week. During inspection, we discovered that the refrigerant levels are below optimal, and the condenser coils show significant dust accumulation. The air filters are clogged and require immediate replacement. The system's performance has degraded by approximately 15% over the past month. Additionally, the thermostat readings indicate inconsistent temperature control across different zones. Recommended actions include refrigerant recharge, deep cleaning of condenser coils, filter replacement, and scheduling a comprehensive system tune-up.""",
        
        "Electrical Panel": """The main electrical panel in facility A has been experiencing frequent breaker trips over the past three days. Initial inspection revealed that the 200-amp main breaker is showing signs of wear and the connections are loose. The secondary breakers are functioning normally, but the load calculation shows that we're operating at 85% capacity during peak hours. This presents a safety concern and potential fire hazard. Immediate actions required: replace the main breaker, tighten all connections, and conduct a comprehensive electrical load analysis.""",
        
        "Plumbing Issue": """The water pressure in the east wing has dropped significantly over the past two days. Investigation shows that there's a leak in the main supply line near the mechanical room. The leak appears to be slow but continuous, causing water damage to the surrounding area. The shut-off valve is functional but needs maintenance. Water quality tests show elevated mineral content, which may be contributing to pipe degradation. Recommended immediate actions: repair the main supply line leak, replace the shut-off valve, and implement a water treatment system to prevent future issues."""
    }
    
    # Sample selector
    selected_sample = st.selectbox(
        "Load Sample Text:",
        [""] + list(sample_texts.keys())
    )
    
    if selected_sample:
        input_text = sample_texts[selected_sample]
    else:
        input_text = ""
    
    # Text input
    input_text = st.text_area(
        "Enter maintenance description:",
        value=input_text,
        height=300,
        placeholder="Paste your maintenance description, work orders, or technical documents here..."
    )
    
    # Text statistics
    if input_text:
        char_count = len(input_text)
        word_count = len(input_text.split())
        st.caption(f"📊 {char_count:,} characters | {word_count:,} words")

with col2:
    st.header("📋 Summary")
    
    # Generate button
    if st.button("Generate Summary", type="primary", disabled=not input_text):
        if not st.session_state.get('model_loaded', False):
            st.markdown('<div class="status-box error-box">⚠️ Please load the model first</div>', unsafe_allow_html=True)
        elif input_text.strip():
            with st.spinner("Generating summary..."):
                summary = summarize_text(
                    st.session_state.get('model_data'),
                    input_text,
                    max_length,
                    min_length
                )
            
            if "Error" not in summary:
                st.markdown('<div class="status-box success-box">✅ Summary generated successfully!</div>', unsafe_allow_html=True)
                
                # Display summary
                st.text_area(
                    "Generated Summary:",
                    value=summary,
                    height=200
                )
                
                # Statistics
                st.subheader("📊 Statistics")
                summary_word_count = get_word_count(summary)
                compression_ratio = get_compression_ratio(input_text, summary)
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Original Words", word_count)
                with col_stat2:
                    st.metric("Summary Words", summary_word_count)
                with col_stat3:
                    st.metric("Compression", f"{compression_ratio}%")
                
                # Download button
                st.download_button(
                    "📥 Download Summary",
                    data=summary,
                    file_name="maintenance_summary.txt",
                    mime="text/plain"
                )
                
            else:
                st.markdown(f'<div class="status-box error-box">{summary}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box info-box">⚠️ Please enter some text</div>', unsafe_allow_html=True)

# Instructions
st.markdown("---")
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    ### Getting Started:
    1. **Load Model**: Click "Load Model" in the sidebar
    2. **Enter Text**: Paste your maintenance description or select a sample
    3. **Adjust Settings**: Modify summary length if needed
    4. **Generate**: Click "Generate Summary"
    5. **Download**: Save the summary using the download button
    
    ### Requirements:
    - Train the model first: `python maintanace__summarizer.py`
    - Model will be saved to `./maintenance_distilbart/`
    
    ### Tips:
    - Use detailed maintenance descriptions for better summaries
    - The model works well with technical maintenance text
    - Adjust length settings based on your needs
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔧 Maintenance Text Summarizer | Powered by AI"
    "</div>",
    unsafe_allow_html=True
)