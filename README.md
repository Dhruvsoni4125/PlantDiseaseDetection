# Plant Disease Detection

A deep learning web application that detects plant diseases from leaf images using a pre-trained CNN model.

## Features

- ✅ Upload plant leaf images for disease detection
- ✅ Real-time predictions with confidence scores
- ✅ Cure/treatment recommendations
- ✅ Live camera feed support (desktop only)
- ✅ Mobile-friendly web interface

## Local Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/Dhruvsoni4125/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

5. Open browser and visit `http://localhost:5000`

## Deployment

### Option 1: Deploy on Render (Recommended)

1. Push code to GitHub
```bash
git add .
git commit -m "Plant disease detector"
git push origin main
```

2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name:** plant-disease-detection
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
6. Deploy!

### Option 2: Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway auto-detects Flask and configures it
5. Deploy!

### Option 3: Deploy on PythonAnywhere

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files via web interface
3. Configure Web app with Flask
4. Reload and access your URL

## Project Structure

```
├── app.py                    # Flask application
├── predict.py               # Prediction utility
├── class_names.json         # Disease labels
├── plant_disease_model.h5   # Pre-trained model
├── requirements.txt         # Dependencies
├── Procfile                 # Deployment config
├── runtime.txt              # Python version
├── templates/
│   └── index.html          # Web interface
└── static/
    └── styles.css          # CSS styles
```

## Model Details

- **Model:** MobileNetV2 with custom top layers
- **Input Size:** 224×224 pixels
- **Classes:** Plant diseases + Healthy leaves
- **Framework:** TensorFlow/Keras

## Technologies Used

- Flask - Web framework
- TensorFlow/Keras - Deep learning
- OpenCV - Image processing
- NumPy - Numerical computing
- Pillow - Image handling

## Usage

1. **Upload Image:** Choose a plant leaf image from your device
2. **Get Prediction:** Model predicts disease and confidence
3. **View Treatment:** Get cure recommendations

## Limitations

- Webcam/live feed only works on desktop browsers
- Large model files (~200MB+) may slow initial load
- Mobile file upload supported (but not live camera)

## License

MIT

## Author

Megha's Project
