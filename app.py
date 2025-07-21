import streamlit as st
from PIL import Image
from model_utils import OpenAICLIPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="OpenAI CLIP Medical Image Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .analysis-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def plot_confidence_chart(results):
    """Create a confidence visualization"""
    if not results:
        return None
    
    labels, probs = zip(*results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), probs, color=['#1f77b4' if i == np.argmax(probs) else '#cccccc' for i in range(len(probs))])
    ax.set_xlabel('Labels')
    ax.set_ylabel('Confidence')
    ax.set_title('Classification Confidence Scores')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_attention_heatmap(attention_data, image):
    """Create attention heatmap visualization"""
    if not attention_data.get("has_attention", False):
        return None
    
    try:
        attention_map = attention_data["attention_map"]
        most_likely_label = attention_data.get("most_likely_label", "Unknown")
        similarity_score = attention_data.get("similarity_score", 0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Attention heatmap
        im = ax2.imshow(attention_map, cmap='hot', interpolation='nearest')
        ax2.set_title(f'Attention Map\nMost likely: {most_likely_label}\nSimilarity: {similarity_score:.3f}')
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating attention visualization: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🔬 OpenAI CLIP Medical Image Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced zero-shot classification with detailed analysis</p>', unsafe_allow_html=True)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = OpenAICLIPClassifier()

    with st.sidebar:
        st.header("⚙️ Settings")
        device = "GPU" if st.session_state.classifier.device == "cuda" else "CPU"
        st.info(f"Running on: {device}")
        st.divider()
        
        st.subheader("🏷️ Classification Labels")
        label_option = st.selectbox(
            "Choose label set:",
            ["Default Labels", "Enhanced Medical Labels", "Custom Labels"]
        )
        
        if label_option == "Default Labels":
            default_labels = st.session_state.classifier.get_default_labels()
        elif label_option == "Enhanced Medical Labels":
            default_labels = st.session_state.classifier.get_enhanced_labels()
        else:
            default_labels = []
        
        labels_text = st.text_area(
            "Enter labels (one per line):",
            value="\n".join(default_labels),
            height=120
        )
        labels = [l.strip() for l in labels_text.splitlines() if l.strip()]
        
        st.divider()
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Upload a medical image (X-ray, CT, MRI, Ultrasound)
        2. Choose or enter classification labels
        3. Click 'Analyze Image' for detailed results
        4. Explore different analysis tabs
        """)
        st.divider()
        st.subheader("⚠️ Disclaimer")
        st.markdown("""
        This is a prototype for educational purposes.\n**Not for clinical use.**
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload X-ray, CT scan, MRI, or Ultrasound images"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Performing detailed analysis with OpenAI CLIP..."):
                    analysis_results = st.session_state.classifier.get_detailed_analysis(image, labels)
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analyzed_image = image
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("📊 Quick Results")
        analysis_results = st.session_state.get('analysis_results', None)
        
        if analysis_results and 'error' not in analysis_results:
            basic_results = analysis_results.get('basic_classification', [])
            confidence_analysis = analysis_results.get('confidence_analysis', {})
            
            if basic_results:
                st.markdown("### Top Predictions:")
                for i, (label, prob) in enumerate(basic_results[:3]):
                    confidence_class = "confidence-high" if prob > 0.8 else "confidence-medium" if prob > 0.6 else "confidence-low"
                    st.markdown(f'<span class="{confidence_class}">{label}: {prob:.1%}</span>', unsafe_allow_html=True)
                
                if confidence_analysis:
                    st.markdown(f"**Confidence Level:** {confidence_analysis.get('confidence_level', 'Unknown')}")
                    st.markdown(f"**Entropy:** {confidence_analysis.get('entropy', 0):.3f}")
        elif analysis_results and 'error' in analysis_results:
            st.error(analysis_results['error'])
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Analysis Tabs
    if st.session_state.get('analysis_results') and 'error' not in st.session_state.analysis_results:
        st.divider()
        st.subheader("🔬 Detailed Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Confidence Analysis", "🎯 Attention Visualization", "📊 Quality Assessment", "🔍 Feature Analysis", "❓ Uncertainty Analysis"])
        
        with tab1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("Confidence Analysis")
            
            confidence_analysis = st.session_state.analysis_results.get('confidence_analysis', {})
            basic_results = st.session_state.analysis_results.get('basic_classification', [])
            
            if confidence_analysis and basic_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Confidence", f"{confidence_analysis.get('max_confidence', 0):.1%}")
                with col2:
                    st.metric("Avg Confidence", f"{confidence_analysis.get('avg_confidence', 0):.1%}")
                with col3:
                    st.metric("Confidence Level", confidence_analysis.get('confidence_level', 'Unknown'))
                
                # Confidence chart
                fig = plot_confidence_chart(basic_results)
                if fig:
                    st.pyplot(fig)
                    plt.close()
                
                # Top predictions table
                st.subheader("Top 3 Predictions")
                top_3 = confidence_analysis.get('top_3_predictions', [])
                for i, (label, prob) in enumerate(top_3):
                    st.markdown(f"{i+1}. **{label}**: {prob:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("Attention Visualization")
            
            attention_analysis = st.session_state.analysis_results.get('attention_analysis', {})
            if attention_analysis.get('has_attention', False):
                fig = plot_attention_heatmap(attention_analysis, st.session_state.analyzed_image)
                if fig:
                    st.pyplot(fig)
                    plt.close()
                st.info("The heatmap shows which regions of the image the model focuses on when making predictions.")
            else:
                st.warning("Attention visualization not available for this model configuration.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("Image Quality Assessment")
            
            quality_analysis = st.session_state.analysis_results.get('image_quality', {})
            if quality_analysis and 'error' not in quality_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Resolution", quality_analysis.get('resolution', 'Unknown'))
                    st.metric("Aspect Ratio", f"{quality_analysis.get('aspect_ratio', 0):.2f}")
                    st.metric("Brightness", f"{quality_analysis.get('brightness', 0):.1f}")
                with col2:
                    st.metric("Contrast", f"{quality_analysis.get('contrast', 0):.1f}")
                    st.metric("Quality Score", quality_analysis.get('quality_score', 0))
                
                quality_notes = quality_analysis.get('quality_notes', [])
                if quality_notes:
                    st.subheader("Quality Notes:")
                    for note in quality_notes:
                        st.markdown(f"• {note}")
                else:
                    st.success("Image quality appears good!")
            else:
                st.error("Quality analysis failed or not available.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("Feature Analysis")
            
            feature_analysis = st.session_state.analysis_results.get('feature_analysis', {})
            if feature_analysis and 'error' not in feature_analysis:
                st.metric("Most Similar Label", feature_analysis.get('most_similar_label', 'Unknown'))
                
                similarities = feature_analysis.get('cosine_similarities', [])
                if similarities:
                    st.subheader("Cosine Similarities")
                    for i, (label, sim) in enumerate(zip(labels, similarities)):
                        st.markdown(f"**{label}**: {sim:.3f}")
                
                similarity_range = feature_analysis.get('similarity_range', (0, 0))
                st.metric("Similarity Range", f"{similarity_range[0]:.3f} - {similarity_range[1]:.3f}")
            else:
                st.error("Feature analysis failed or not available.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.subheader("Uncertainty Analysis")
            
            uncertainty_analysis = st.session_state.analysis_results.get('uncertainty_analysis', {})
            if uncertainty_analysis:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entropy", f"{uncertainty_analysis.get('entropy', 0):.3f}")
                with col2:
                    st.metric("Max Probability", f"{uncertainty_analysis.get('max_probability', 0):.1%}")
                with col3:
                    st.metric("Uncertainty Level", uncertainty_analysis.get('uncertainty_level', 'Unknown'))
                
                recommendation = uncertainty_analysis.get('recommendation', '')
                if recommendation:
                    st.subheader("Recommendation:")
                    st.info(recommendation)
                
                prob_gap = uncertainty_analysis.get('probability_gap', 0)
                if prob_gap > 0:
                    st.metric("Probability Gap", f"{prob_gap:.1%}")
            else:
                st.error("Uncertainty analysis failed or not available.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using OpenAI CLIP and Streamlit</p>
        <p>For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 