# 🌱 Plant Whisperer - Streamlit App

AI-powered plant health diagnosis with automatic background removal and species detection.

## Features

✅ **Plant Health Diagnosis** - Detects 4 conditions:
- Under-watered plants
- Low light exposure
- High light/sun damage
- Healthy plants

✅ **Automatic Background Removal** - Uses `rembg` to focus on the plant

✅ **Species Detection** - Identifies plant species using Google Gemini AI

✅ **Treatment Recommendations** - Detailed care instructions for each condition

✅ **Beautiful UI** - Modern, responsive interface with charts and metrics

## Installation

### 1. Clone or Download

Download the files:
- `app.py` - Main application
- `requirements.txt` - Dependencies

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables (Optional)

For species detection with Gemini AI:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

**Note:** The app works without Gemini API key, but species detection will be disabled.

## Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Add `GEMINI_API_KEY` in "Advanced settings" → "Secrets"
   - Click "Deploy"

**Secrets format in Streamlit Cloud:**
```toml
GEMINI_API_KEY = "your_api_key_here"
```

### Option 2: Render.com

1. **Create `render.yaml`:**
```yaml
services:
  - type: web
    name: plant-whisperer
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: GEMINI_API_KEY
        sync: false
```

2. **Deploy:**
   - Push to GitHub
   - Connect repository to Render
   - Add environment variables
   - Deploy

### Option 3: Hugging Face Spaces

1. **Create a Space** on [huggingface.co/spaces](https://huggingface.co/spaces)

2. **Add files:**
   - `app.py`
   - `requirements.txt`

3. **Add API key** in Space settings → Variables

## Usage Guide

### 1. Upload Image
- Click "Browse files" or drag & drop
- Supported formats: JPG, JPEG, PNG
- Best results: Clear photos of plant leaves

### 2. Configure Settings (Sidebar)
- **Remove Background** - Toggle automatic background removal
- **Detect Plant Species** - Enable/disable species detection (requires API key)

### 3. Diagnose
- Click "🌱 Diagnose Plant Health"
- Wait for analysis (5-10 seconds)

### 4. View Results
- **Plant Species** - Scientific and common names (if detected)
- **Health Diagnosis** - Condition with confidence score
- **Severity Level** - Mild/Moderate/Severe
- **Treatment Plan** - Detailed care instructions
- **Prevention Tips** - How to avoid the issue
- **Prediction Scores** - All model predictions with chart

## Model Information

- **Source:** Hugging Face Hub (`Senuda2004/plant-whisperer-keras`)
- **Architecture:** ResNet50-based transfer learning
- **Input:** 224x224 RGB images
- **Classes:** 4 (healthy, high_light, low_light, under_water)

## Troubleshooting

### Background Removal Issues

If `rembg` fails:
- The app will use the original image automatically
- Check if image format is supported (RGB)

### Model Download Fails

```bash
# Manual download
huggingface-cli download Senuda2004/plant-whisperer-keras plant_whisperer_final.keras
```

### Gemini API Errors

- Verify API key is correct
- Check [AI Studio](https://aistudio.google.com/app/apikey) for quota
- Species detection will be disabled if API fails

### Port Already in Use

```bash
streamlit run app.py --server.port=8502
```

## Performance Tips

1. **Image Size:** Resize large images before upload (< 5MB)
2. **Background:** Dark or neutral backgrounds work best
3. **Lighting:** Good lighting for clear leaf details
4. **Angle:** Take photos straight-on, not at extreme angles

## Project Structure

```
plant-whisperer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── plant_whisperer_final.keras  # Model (auto-downloaded)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No | Google Gemini API key for species detection |

## Dependencies

- `streamlit` - Web framework
- `tensorflow` - Deep learning model
- `rembg` - Background removal
- `google-generativeai` - Species detection
- `huggingface-hub` - Model download
- `Pillow` - Image processing
- `numpy` - Array operations
- `pandas` - Data visualization

## Contributing

Feel free to:
- Report issues
- Suggest features
- Submit pull requests

## License

MIT License - feel free to use for personal or commercial projects

## Support

For issues or questions:
1. Check troubleshooting section
2. Review [Streamlit docs](https://docs.streamlit.io)
3. Check [rembg documentation](https://github.com/danielgatis/rembg)

## Credits

- Model: Custom trained ResNet50
- Background Removal: [rembg](https://github.com/danielgatis/rembg)
- Species Detection: Google Gemini 1.5 Flash
- Framework: Streamlit

---

Made with 🌱 by Plant Whisperer Team
