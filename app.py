"""
Plant Whisperer - Streamlit App
AI-powered plant health diagnosis with species detection and background removal
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
from datetime import datetime
import os
from rembg import remove
from huggingface_hub import hf_hub_download
import json

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="🌱 Plant Whisperer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .diagnosis-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .healthy-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================

# Class names
CLASS_NAMES = ["healthy", "high_light", "low_light", "under_water"]

# Image preprocessing config
IMG_SIZE = (224, 224)

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def download_model_from_hf():
    """Download model from Hugging Face"""
    model_filename = "plant_whisperer_final.keras"
    
    if os.path.exists(model_filename):
        return model_filename
    
    try:
        with st.spinner("📥 Downloading model from Hugging Face..."):
            downloaded_path = hf_hub_download(
                repo_id="Senuda2004/plant-whisperer-keras",
                filename=model_filename,
                local_dir=".",
                local_dir_use_symlinks=False
            )
        return downloaded_path
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        raise


@st.cache_resource
def load_model():
    """Load the trained Keras model"""
    try:
        model_path = download_model_from_hf()
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background from image using rembg
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image with background removed
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Remove background
        output = remove(img_byte_arr)
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output))
        
        return result_image
    except Exception as e:
        st.warning(f"⚠️ Background removal failed: {e}. Using original image.")
        return image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    
    # Convert to array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # ResNet50 preprocessing
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
    """
    Use Gemini to detect plant species from image
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with plant species information
    """
    if not GEMINI_API_KEY:
        return {
            "species_name": "Unknown",
            "common_name": "Plant species detection unavailable",
            "confidence": 0.0,
            "plant_family": "N/A",
            "care_level": "N/A"
        }
    
    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        Identify this houseplant species. Provide a JSON response with:
        {
            "species_name": "Scientific name (e.g., Monstera deliciosa)",
            "common_name": "Common name (e.g., Swiss Cheese Plant)",
            "plant_family": "Family name (e.g., Araceae)",
            "care_level": "Easy/Moderate/Difficult",
            "confidence": 0.0-1.0
        }
        
        Focus on common indoor houseplants. If uncertain, provide best guess with lower confidence.
        Only return the JSON, no other text.
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
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the console for errors.")
        return
    
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
            value=bool(GEMINI_API_KEY),
            disabled=not bool(GEMINI_API_KEY),
            help="Use Gemini AI to identify plant species (requires API key)"
        )
        
        st.divider()
        
        st.subheader("📊 About")
        st.info("""
        **Plant Whisperer** uses AI to diagnose common plant health issues:
        
        - 🚰 Under-watering
        - 💡 Low light
        - ☀️ High light
        - ✅ Healthy plants
        
        Upload a clear photo of your plant leaves for best results!
        """)
        
        st.divider()
        
        st.subheader("🔑 API Status")
        if GEMINI_API_KEY:
            st.success("✅ Gemini API: Enabled")
        else:
            st.warning("⚠️ Gemini API: Disabled\n\nSet GEMINI_API_KEY environment variable to enable species detection.")
    
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
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Process image
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
                    # Preprocess for model
                    img_array = preprocess_image(processed_image)
                    
                    # Get prediction
                    predictions = model.predict(img_array, verbose=0)
                    
                    # Get results
                    class_idx = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][class_idx])
                    predicted_class = CLASS_NAMES[class_idx]
                    
                    # Calculate severity
                    severity_info = calculate_severity(confidence)
                    
                    # Get treatment
                    treatment = TREATMENTS[predicted_class]
                    
                    # Store in session state
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
        
        diagnosis = st.session_state.diagnosis
        
        # Results header
        st.subheader("📋 Diagnosis Results")
        
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
                
                st.caption(f"Family: {species.get('plant_family', 'N/A')} | Confidence: {species.get('confidence', 0):.1%}")
        
        # Health diagnosis
        predicted_class = diagnosis["predicted_class"]
        confidence = diagnosis["confidence"]
        severity = diagnosis["severity"]
        treatment = diagnosis["treatment"]
        
        # Metrics row
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
            
            st.markdown("### ⏰ Frequency")
            st.info(treatment["frequency"])
        
        with col2:
            st.markdown("### ⏱️ Recovery Time")
            st.success(treatment["recovery_time"])
            
            st.markdown("### 🛡️ Prevention Tips")
            for tip in treatment["prevention"]:
                st.markdown(f"- {tip}")
            
            st.markdown("### ⚠️ Warning Signs")
            st.warning(treatment["warning_signs"])
        
        # All predictions
        with st.expander("📊 Detailed Prediction Scores"):
            predictions_data = {
                CLASS_NAMES[i]: float(diagnosis["predictions"][i])
                for i in range(len(CLASS_NAMES))
            }
            
            # Create bar chart
            import pandas as pd
            df = pd.DataFrame({
                "Condition": [name.replace("_", " ").title() for name in predictions_data.keys()],
                "Probability": list(predictions_data.values())
            })
            
            st.bar_chart(df.set_index("Condition"))
            
            # Show raw scores
            for condition, score in predictions_data.items():
                st.write(f"**{condition.replace('_', ' ').title()}:** {score:.2%}")


if __name__ == "__main__":
    main()
