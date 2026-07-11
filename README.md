# 🌾 Agro World

**Agro World** is a Flask-based web application that helps farmers make smarter agricultural decisions using machine learning. It combines three tools in one platform: **crop disease detection** from leaf images, **crop recommendation** based on soil and climate conditions, and **fertilizer recommendation** based on soil nutrients and crop type.

---

## ✨ Features

### 🌿 Crop Disease Detection
Upload a photo of a plant leaf and get an instant diagnosis using crop-specific CNN (deep learning) models. Supports disease detection for:

| Crop | Detectable Conditions |
|------|------------------------|
| Rice | Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Narrow Brown Spot, Neck Blast, Rice Hispa, Sheath Blight, Tungro, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Corn | Cercospora/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Strawberry | Leaf Scorch, Healthy |

### 🌱 Crop Recommendation
Suggests the most suitable crop to grow based on soil and environmental parameters (N, P, K values, temperature, humidity, pH, rainfall, etc.) using a trained ML model.

### 🧪 Fertilizer Recommendation
Recommends the ideal fertilizer (e.g., Urea, DAP, 10-26-26, 14-35-14, 17-17-17, 20-20, 28-28) based on soil type, crop type, and nutrient levels.

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow / Keras (CNN models for disease classification, `.h5` format)
- **Machine Learning:** scikit-learn (crop & fertilizer recommendation models, `.pkl` format)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **Image Handling:** Werkzeug, Keras image preprocessing

---

## 📁 Project Structure

```
Agro_World/
├── app.py                     # Main Flask application & routes
├── functions.py                # Model loading, prediction & recommendation logic
├── Model_Loading.py            # Model loading utilities
├── dataset.py                  # Dataset handling utilities
├── train.py                    # Model training script
├── rice.py                     # Rice-specific processing/training script
├── models/
│   ├── DL_models/              # CNN disease-detection models (.h5) — per crop
│   └── ML_models/              # Crop & fertilizer recommendation models (.pkl)
├── static/                     # CSS, JS, and static assets
├── templates/                  # HTML templates (Jinja2)
├── uploads/                    # Uploaded leaf images for prediction
├── dataset/                    # Training dataset
├── Crop_Disease_Images/        # Sample/reference disease images
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Groot-PAS/Agro_World.git
cd Agro_World
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install flask tensorflow numpy werkzeug scikit-learn
```

> 💡 If you have a `requirements.txt`, use `pip install -r requirements.txt` instead.

### 4. Add the model files
Make sure the trained models are present in the `models/` folder:
- `models/DL_models/<crop>_model.h5` — one per crop (e.g., `rice_model.h5`, `tomato_model.h5`)
- `models/ML_models/crop_model.pkl`, `crop_scaler.pkl`
- `models/ML_models/fertilizer_model.pkl`, `fertilizer_scaler.pkl`

> ⚠️ **Note:** Some model files exceed GitHub's 100 MB upload limit and are **not included** in this repository. Download them from **[add your Google Drive / Hugging Face link here]** and place them in the corresponding folders above before running the app.

### 5. Run the app
```bash
python app.py
```

The app will start in debug mode at:
```
http://127.0.0.1:5000/
```

---

## 🚀 Usage

1. **Home page** — navigate to the tool you need.
2. **Crop Disease Detection** — select your crop, upload a leaf image, and get an instant diagnosis.
3. **Crop Recommendation** — enter soil nutrients (N, P, K) and climate values to get the best crop suggestion.
4. **Fertilizer Recommendation** — enter soil type, crop type, and nutrient levels to get a fertilizer suggestion.

---

## 🧠 Model Training

- `train.py` and `rice.py` contain the training pipeline used to build the CNN models on the image dataset (`dataset/`, `Crop_Disease_Images/`).
- `Model_Loading.py` handles loading trained models for inference.
- Disease classification models were trained per crop, so each crop has its own dedicated `.h5` model file.

---

## 📌 Future Improvements

- [ ] Add a `requirements.txt` for easier dependency management
- [ ] Deploy the app on a cloud platform (Render, Railway, Heroku, etc.)
- [ ] Add authentication and user history/dashboard
- [ ] Improve UI/UX with a modern frontend framework
- [ ] Add API endpoints for mobile app integration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

This project currently has no license specified. Consider adding one (e.g., MIT) if you plan to share or open-source it.

---

*Built to help farmers make data-driven decisions about crop health, crop selection, and fertilization.*
