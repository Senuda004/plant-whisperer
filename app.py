"""
Plant Whisperer - Streamlit App (Optimized for Cloud)
AI-powered plant health diagnosis with background removal
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
try:
    import google.genai as genai
    GENAI_NEW = True
except ImportError:
    try:
        import google.generativeai as genai
        GENAI_NEW = False
    except ImportError:
        genai = None
from datetime import datetime
import os
from rembg import remove
from huggingface_hub import hf_hub_download
import json
import sys

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="🌱 Plant Whisperer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Print Python and TensorFlow info for debugging
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

CLASS_NAMES = ["healthy", "high_light", "low_light", "under_water"]
IMG_SIZE = (224, 224)
MODEL_REPO = "Senuda2004/plant-whisperer-keras"
MODEL_FILENAME = "plant_whisperer_final.keras"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except:
        pass

# ==================== TREATMENT DATA ====================

TREATMENTS = {
    "under_water": {
        "title": "🚰 Under-watered Plant",
        "description": "Your plant needs water. The soil is too dry.",
        "immediate_action": "Water thoroughly until water drains from the bottom of the pot",
        "ongoing_care": "Check soil moisture before watering - top 2 inches should be dry",
        "frequency": "Every 5-7 days (varies by species and environment)",
        "recovery_time": "24-48 hours - leaves should perk up",
        "prevention": [
            "Set a watering reminder on your phone",
            "Check soil moisture regularly with your finger",
            "Use a moisture meter for accuracy"
        ],
        "warning_signs": "Wilting, drooping leaves, dry soil, leaf tips turning brown"
    },
    "low_light": {
        "title": "💡 Insufficient Light",
        "description": "Your plant isn't getting enough light for healthy growth.",
        "immediate_action": "Move plant to a brighter location (avoid direct harsh sunlight)",
        "ongoing_care": "Place 3-5 feet from a window with indirect light",
        "frequency": "Monitor growth - adjust placement if needed",
        "recovery_time": "1-2 weeks - new growth should be darker and healthier",
        "prevention": [
            "Rotate plant weekly for even light exposure",
            "Consider grow lights if natural light is limited",
            "Research your plant's specific light requirements"
        ],
        "warning_signs": "Leggy growth, pale/yellow leaves, slow growth, leaning toward light"
    },
    "high_light": {
        "title": "☀️ Too Much Light",
        "description": "Your plant is getting too much direct sunlight.",
        "immediate_action": "Move plant away from direct sun or add sheer curtain",
        "ongoing_care": "Provide bright, indirect light - not direct harsh sun",
        "frequency": "Monitor for sunburn - adjust location as seasons change",
        "recovery_time": "2-3 weeks - damaged leaves won't recover but new growth will be healthy",
        "prevention": [
            "Use sheer curtains to filter intense sunlight",
            "Move plant back from window during peak sun hours",
            "Research your plant's light tolerance"
        ],
        "warning_signs": "Brown/bleached spots on leaves, scorched leaf edges, faded colors"
    },
    "healthy": {
        "title": "✅ Healthy Plant",
        "description": "Your plant looks healthy! Keep up the good care.",
        "immediate_action": "No action needed - continue current care routine",
        "ongoing_care": "Maintain consistent watering, light, and temperature",
        "frequency": "Check weekly for any changes",
        "recovery_time": "N/A - plant is healthy",
        "prevention": [
            "Keep doing what you're doing!",
            "Monitor for any changes in appearance",
            "Maintain consistent care schedule"
        ],
        "warning_signs": "N/A - No issues detected"
    }
}

# ==================== MODEL FUNCTIONS ====================

@st.cache_resource(show_spinner=False)
def download_and_load_model():
    """Download and load model in one step"""
    try:
        print(f"\n{'='*60}")
        print("🚀 MODEL LOADING PROCESS STARTED")
        print(f"{'='*60}")
        
        # Check if model already exists locally
        if os.path.exists(MODEL_FILENAME):
            print(f"✓ Found local model: {MODEL_FILENAME}")
            print("📂 Loading model from local file...")
            model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
            print("✓ Model loaded successfully!")
            return model
        
        # Download from Hugging Face
        print(f"📥 Downloading model from Hugging Face...")
        print(f"   Repository: {MODEL_REPO}")
        print(f"   File: {MODEL_FILENAME}")
        print(f"   Size: ~85MB (this may take 1-3 minutes)...")
        
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            cache_dir="./model_cache",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✓ Download complete: {model_path}")
        print("📂 Loading model into memory...")
        
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        print(f"✓ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"{'='*60}\n")
        
        return model
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ MODEL LOADING FAILED")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")
        return None


# ==================== HELPER FUNCTIONS ====================

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        output = remove(img_byte_arr)
        result_image = Image.open(io.BytesIO(output))
        
        return result_image
    except Exception as e:
        st.warning(f"⚠️ Background removal failed: {e}. Using original image.")
        return image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array


def calculate_severity(confidence: float) -> dict:
    """Calculate severity level based on confidence score"""
    if confidence < 0.70:
        return {
            "level": "mild",
            "badge": "🟢 Mild",
            "color": "#4CAF50",
            "description": "Minor issue - early detection"
        }
    elif confidence < 0.85:
        return {
            "level": "moderate",
            "badge": "🟡 Moderate",
            "color": "#FF9800",
            "description": "Needs attention soon"
        }
    else:
        return {
            "level": "severe",
            "badge": "🔴 Severe",
            "color": "#F44336",
            "description": "Urgent - take action now"
        }


def detect_plant_species(image: Image.Image) -> dict:
    """Use Gemini to detect plant species"""
    if not GEMINI_API_KEY or not genai:
        return {
            "species_name": "Unknown",
            "common_name": "Species detection unavailable",
            "confidence": 0.0,
            "plant_family": "N/A",
            "care_level": "N/A"
        }
    
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Identify this houseplant species. Provide a JSON response with:
        {
            "species_name": "Scientific name",
            "common_name": "Common name",
            "plant_family": "Family name",
            "care_level": "Easy/Moderate/Difficult",
            "confidence": 0.0-1.0
        }
        Only return JSON, no other text.
        """
        
        response = model_gemini.generate_content([prompt, image])
        species_info = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        
        return species_info
        
    except Exception as e:
        st.warning(f"⚠️ Species detection error: {e}")
        return {
            "species_name": "Unknown",
            "common_name": "Could not identify species",
            "confidence": 0.0,
            "plant_family": "N/A",
            "care_level": "N/A"
        }


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Plant Whisperer</h1>
        <p>AI-Powered Plant Health Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show loading progress
    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.info("🔄 **Loading AI model...** This takes 1-3 minutes on first run, then it's instant!")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing...")
        progress_bar.progress(20)
        
        model = download_and_load_model()
        
        progress_bar.progress(100)
        status_text.text("Model loaded!")
    
    # Clear progress indicators
    progress_placeholder.empty()
    
    if model is None:
        st.error("❌ **Failed to load model**")
        st.info("""
        **What happened?**
        - Model download from Hugging Face may have timed out
        - Network connection issue
        
        **What to do:**
        1. **Refresh the page** (press F5 or reload button)
        2. Wait 30 seconds and try again
        3. If problem persists, check [Hugging Face Status](https://status.huggingface.co/)
        
        The model is ~85MB and downloads on first run only.
        """)
        
        if st.button("🔄 Try Loading Model Again"):
            st.rerun()
        
        return
    
    # Success message
    st.success("✅ **Model ready!** Upload a plant image to get started.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        remove_bg = st.checkbox(
            "Remove Background",
            value=True,
            help="Automatically remove background from uploaded image"
        )
        
        detect_species = st.checkbox(
            "Detect Plant Species",
            value=bool(GEMINI_API_KEY and genai),
            disabled=not bool(GEMINI_API_KEY and genai),
            help="Use Gemini AI to identify plant species (requires API key)"
        )
        
        st.divider()
        
        st.subheader("📊 About")
        st.info("""
        **Plant Whisperer** diagnoses:
        
        - 🚰 Under-watering
        - 💡 Low light
        - ☀️ High light
        - ✅ Healthy plants
        
        Upload a clear photo of plant leaves for best results!
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Plant Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your plant"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            if remove_bg:
                with st.spinner("🎨 Removing background..."):
                    processed_image = remove_background(image)
                st.image(processed_image, caption="Background Removed", use_container_width=True)
            else:
                processed_image = image
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis")
            
            if st.button("🌱 Diagnose Plant Health", type="primary"):
                with st.spinner("🤖 Analyzing plant health..."):
                    # Preprocess
                    img_array = preprocess_image(processed_image)
                    
                    # Predict
                    predictions = model.predict(img_array, verbose=0)
                    
                    # Get results
                    class_idx = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][class_idx])
                    predicted_class = CLASS_NAMES[class_idx]
                    
                    # Calculate severity
                    severity_info = calculate_severity(confidence)
                    
                    # Get treatment
                    treatment = TREATMENTS[predicted_class]
                    
                    # Store results
                    st.session_state.diagnosis = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "severity": severity_info,
                        "treatment": treatment,
                        "predictions": predictions[0]
                    }
                    
                    # Detect species if enabled
                    if detect_species:
                        with st.spinner("🔬 Identifying plant species..."):
                            species_info = detect_plant_species(processed_image)
                            st.session_state.species = species_info
    
    # Display results
    if hasattr(st.session_state, 'diagnosis'):
        st.divider()
        st.subheader("📋 Diagnosis Results")
        
        diagnosis = st.session_state.diagnosis
        
        # Species info (if available)
        if hasattr(st.session_state, 'species') and st.session_state.species:
            species = st.session_state.species
            
            with st.expander("🌿 Plant Species Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Common Name", species.get("common_name", "Unknown"))
                
                with col2:
                    st.metric("Scientific Name", species.get("species_name", "Unknown"))
                
                with col3:
                    st.metric("Care Level", species.get("care_level", "N/A"))
        
        # Health diagnosis
        predicted_class = diagnosis["predicted_class"]
        confidence = diagnosis["confidence"]
        severity = diagnosis["severity"]
        treatment = diagnosis["treatment"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Diagnosis", predicted_class.replace("_", " ").title())
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.markdown(f"""
            <div style="padding: 1rem; background-color: {severity['color']}20; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0; color: {severity['color']};">{severity['badge']}</h3>
                <p style="margin: 0; font-size: 0.9rem;">{severity['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Treatment recommendations
        st.subheader(treatment["title"])
        st.markdown(f"**{treatment['description']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚨 Immediate Action")
            st.info(treatment["immediate_action"])
            
            st.markdown("### 🔄 Ongoing Care")
            st.info(treatment["ongoing_care"])
        
        with col2:
            st.markdown("### ⏱️ Recovery Time")
            st.success(treatment["recovery_time"])
            
            st.markdown("### 🛡️ Prevention Tips")
            for tip in treatment["prevention"]:
                st.markdown(f"- {tip}")
        
        # All predictions
        with st.expander("📊 Detailed Prediction Scores"):
            import pandas as pd
            
            predictions_data = {
                CLASS_NAMES[i]: float(diagnosis["predictions"][i])
                for i in range(len(CLASS_NAMES))
            }
            
            df = pd.DataFrame({
                "Condition": [name.replace("_", " ").title() for name in predictions_data.keys()],
                "Probability": list(predictions_data.values())
            })
            
            st.bar_chart(df.set_index("Condition"))


if __name__ == "__main__":
    main()
