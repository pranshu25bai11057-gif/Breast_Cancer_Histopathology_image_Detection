import streamlit as st
from PIL import Image
import os
import predict

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Histopathology Classifier",
    page_icon="🔬",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.8rem;
        text-align: center;
    }
    .result-card-benign {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-card-malignant {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-card-invalid {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .disclaimer-box {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<div class='main-header'>🔬 Breast Histopathology Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI/ML Exhibition Prototype — Project Exhibition 01</div>", unsafe_allow_html=True)

# Sidebar with Project Details and Viva Explanations
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("""
    **Modality:** Breast Histopathology (H&E)  
    **Dataset:** BreaKHis  
    **Model:** 3-Stage Convolutional Neural Network  
    **Classes:** Invalid / Benign / Malignant  
    """)
    
    st.divider()
    
    st.subheader("⚙️ System Pipeline")
    st.markdown("""
    1. **Upload / Select:** Provide slide image (PNG/JPG).
    2. **Gate:** Image segregation & validity check.
    3. **Preprocess:** RGB conversion, $128\\times128$ resize, normalization $[0, 1]$.
    4. **Inference:** Single CNN classification.
    5. **Output:** Benign/Malignant prediction + confidence score.
    """)
    
    st.divider()
    st.info("💡 **Zero Leakage:** Split strictly at patient level to ensure genuine unseen-patient evaluation.")

# Input Mode: Upload or Built-in Demo Samples
input_mode = st.radio(
    "Choose Input Method:",
    ["Upload My Own Image", "Use Built-in Demo Sample"],
    horizontal=True
)

target_image = None
image_caption = ""

if input_mode == "Upload My Own Image":
    uploaded_file = st.file_uploader(
        "Upload a Breast Histopathology Slide Image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        target_image = uploaded_file
        image_caption = uploaded_file.name
else:
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    sample_options = {
        "Sample 1: Benign Slide (Fibroadenoma / Adenoma)": os.path.join(sample_dir, "sample_benign.png"),
        "Sample 2: Malignant Slide (Carcinoma)": os.path.join(sample_dir, "sample_malignant.png"),
        "Sample 3: Invalid Non-Histopathology Image": os.path.join(sample_dir, "sample_invalid.png"),
    }
    selected_sample = st.selectbox("Select a demo sample to test:", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]
    if os.path.exists(sample_path):
        target_image = sample_path
        image_caption = selected_sample

# Main Prediction Section
if target_image is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("🖼️ Selected Image")
        try:
            display_img = Image.open(target_image)
            st.image(display_img, use_container_width=True, caption=image_caption)
        except Exception as e:
            st.error(f"Error opening image file: {str(e)}")
            st.stop()
            
    with col2:
        st.subheader("📊 Analysis & Prediction")
        
        with st.spinner("Analyzing image..."):
            model_path = predict.DEFAULT_MODEL_PATH
            if not os.path.exists(model_path):
                st.warning("⚠️ Model weights not found. Please verify `model/breast_cancer_model.keras` exists.")
            else:
                # Run prediction
                result = predict.predict_cancer(target_image, model_path=model_path)
                
                if result["status"] == "error":
                    st.error(f"⚠️ {result['message']}")
                    
                elif result["prediction"] == "Invalid":
                    st.markdown(f"""
                    <div class='result-card-invalid'>
                        <h4 style='color: #B45309; margin:0;'>⚠️ Invalid Image</h4>
                        <p style='margin: 0.5rem 0 0 0; color: #78350F;'>
                            {result['message']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"Rejection Confidence: **{result['confidence']}%**")
                    
                else:
                    # Valid Benign or Malignant result
                    pred_label = result["prediction"]
                    confidence = result["confidence"]
                    
                    if pred_label == "Benign":
                        st.markdown(f"""
                        <div class='result-card-benign'>
                            <h3 style='color: #065F46; margin:0;'>✅ Prediction: Benign</h3>
                            <p style='margin: 0.4rem 0 0 0; color: #047857;'>
                                Non-cancerous tissue characteristics detected.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-card-malignant'>
                            <h3 style='color: #991B1B; margin:0;'>🚨 Prediction: Malignant</h3>
                            <p style='margin: 0.4rem 0 0 0; color: #B91C1C;'>
                                Cancerous tissue characteristics detected.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown(f"**Model Confidence:** `{confidence:.2f}%`")
                    st.progress(min(max(confidence / 100.0, 0.0), 1.0))
                    
                    # Detailed Probabilities
                    with st.expander("🔍 View Probability Distribution"):
                        st.write(result["probabilities"])
else:
    st.info("👆 Upload an image or select a built-in demo sample above to test the classifier.")

# Mandatory Medical Disclaimer
st.markdown("""
<div class='disclaimer-box'>
    <strong>⚠️ Educational & Research Disclaimer:</strong><br>
    This application is an educational prototype developed for second-year student exhibition. 
    It is <strong>NOT</strong> a medical device or diagnostic system and must not be used for clinical decision-making or treatment recommendations.
</div>
""", unsafe_allow_html=True)
