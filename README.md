# 🌾 Agro World

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/DL-TensorFlow%20%2F%20Keras-FF6F00.svg)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

**Agro World** is a production-ready agricultural decision-support web platform. It integrates deep learning computer vision and machine learning models into a unified interface for:
1. **🌿 Crop Disease Detection:** Leaf image diagnosis with confidence estimation and non-leaf / low-confidence rejection across 10 crops.
2. **🌱 Crop Recommendation:** Multi-class classification suggesting optimal crops based on N, P, K soil nutrients and climatic factors (temperature, humidity, pH, rainfall).
3. **🧪 Fertilizer Recommendation:** Targeted fertilizer suggestion based on soil characteristics, crop type, and nutrient composition.

---

## 🛠️ Architecture & Model Specifications

### 1. Crop Disease Detection (Deep Learning)
Crop leaf diagnosis uses crop-specific models tailored to each plant's disease spectrum:

| Crop | Architecture | Strategy | Classes / Outputs | Loss & Activation |
|:---|:---|:---|:---:|:---:|
| **Rice** | Custom 2-Conv CNN | Trained from scratch | 10 classes | Categorical Crossentropy, Softmax |
| **Apple** | MobileNetV2 | Transfer Learning | 4 classes | Categorical Crossentropy, Softmax |
| **Corn** | MobileNetV2 | Transfer Learning | 4 classes | Categorical Crossentropy, Softmax |
| **Grape** | MobileNetV2 | Transfer Learning | 4 classes | Categorical Crossentropy, Softmax |
| **Potato** | MobileNetV2 | Transfer Learning | 3 classes | Categorical Crossentropy, Softmax |
| **Tomato** | MobileNetV2 | Transfer Learning | 10 classes | Categorical Crossentropy, Softmax |
| **Cherry** | MobileNetV2 | Transfer Learning | 2 classes (Binary) | Binary Crossentropy, Sigmoid |
| **Peach** | MobileNetV2 | Transfer Learning | 2 classes (Binary) | Binary Crossentropy, Sigmoid |
| **Pepper** | MobileNetV2 | Transfer Learning | 2 classes (Binary) | Binary Crossentropy, Sigmoid |
| **Strawberry** | MobileNetV2 | Transfer Learning | 2 classes (Binary) | Binary Crossentropy, Sigmoid |

- **Preprocessing:** Images are resized to `(224, 224, 3)` and normalized via `rescale=1./255`.
- **Confidence Scoring & Unknown Rejection:** Softmax probabilities (and sigmoid margins for binary models) are extracted. If the highest class confidence is below **50%**, the application flags a warning to alert users that the image may be blurry, poorly illuminated, or not a leaf from the selected crop (e.g. a shoe or non-plant object).
- **In-Memory Caching:** Deep learning models are cached via an `@lru_cache(maxsize=3)` mechanism, avoiding repeated disk I/O while preventing memory exhaustion on low-RAM hosting tiers.

### 2. Crop Recommendation (Machine Learning)
- **Dataset:** 2,200 real agricultural soil & climate records across 22 crops (100 samples per crop).
- **Features (7):** Nitrogen (`N`), Phosphorus (`P`), Potassium (`K`), Temperature (`temperature`), Humidity (`humidity`), Soil pH (`ph`), Rainfall (`rainfall`).
- **Pipeline:** `StandardScaler` + `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Performance:** **99.55%** test accuracy on an 80/20 stratified split. Classes are 0-indexed (`0..21`) in alphabetical order.

### 3. Fertilizer Recommendation (Machine Learning)
- **Dataset:** 99 records across 7 fertilizer formulations (`10-26-26`, `14-35-14`, `17-17-17`, `20-20`, `28-28`, `DAP`, `Urea`).
- **Features (8):**
  - Numerical (6): `Temparature`, `Humidity `, `Moisture`, `Nitrogen`, `Potassium`, `Phosphorous` (scaled with `StandardScaler`).
  - Categorical (2): `Soil Type` (5 categories), `Crop Type` (11 categories).
- **Feature Order Integrity:** The scaler was fit using `[Temperature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous]` ($N, K, P$). All web requests parse parameters explicitly by field name to guarantee proper placement.
- **Evaluation & Dataset Reality:**
  - With 99 samples (~14 samples/class), an isolated train/test split can yield 100% accuracy due to lucky splits and tree memorization.
  - Using **5-Fold Stratified Cross-Validation**, the model achieves **96.00% ± 4.90%** accuracy, providing an honest baseline metric for tabular fertilizer prediction.

---

## 📁 Repository Structure

```
Agro_World/
├── app.py                     # Flask application routes, input validation & error handling
├── functions.py               # Model loading, LRU caching, image inference & feature transformations
├── train.py                   # CLI training pipeline for Crop and Fertilizer ML models
├── rice.py                    # Training pipeline for the custom Rice CNN architecture
├── requirements.txt           # Production & development dependencies
├── notebooks/
│   └── Model_Training.ipynb   # Interactive EDA, feature distributions, CV & model evaluation
├── dataset/
│   ├── Crop_recommendation.csv    # 2,200 samples for crop selection
│   └── Fertilizer Prediction.csv  # 99 samples for fertilizer recommendation
├── models/
│   ├── DL_models/             # CNN leaf disease classification models (.h5)
│   └── ML_models/             # Random Forest classifiers and StandardScaler pickles (.pkl)
├── templates/                 # Jinja2 HTML responsive templates
├── static/                    # CSS stylesheets, JavaScript, and UI icons
├── tests/
│   └── test_agro_world.py     # Pytest test suite covering ML inference, routes & validation
└── uploads/                   # Temporary directory for uploaded inference images (.gitkeep)
```

---

## ⚙️ Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Groot-PAS/Agro_World.git
cd Agro_World/Agro_World
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application

#### Development:
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

#### Production (Gunicorn):
```bash
gunicorn --workers 2 --threads 2 --timeout 120 -b 0.0.0.0:5000 app:app
```

> 💡 **Production Deployment Tip:** Keras and TensorFlow models have substantial memory footprints. Set `--workers 2` and `--timeout 120` to avoid worker timeouts or out-of-memory errors on 512MB-1GB RAM instances.

---

## 🧠 Model Training & Re-generation

You can retrain and regenerate all machine learning artifacts at any time using `train.py`:

```bash
# Train both Crop and Fertilizer models and update models/ML_models/
python train.py

# Train only Crop Recommendation
python train.py --crop

# Train only Fertilizer Recommendation
python train.py --fertilizer
```

To explore the EDA and cross-validation analysis interactively, open the Jupyter notebook:
```bash
jupyter notebook notebooks/Model_Training.ipynb
```

To train the custom Rice CNN model:
```bash
python rice.py --train_dir path/to/train --val_dir path/to/val --epochs 15
```

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/test_agro_world.py -v
```

The test suite covers:
- Verification of crop predictions on known rows from the dataset
- Fertilizer prediction feature ordering ($N, K, P$ order verification)
- Disease metadata integrity and typo verification (`potato_model.h5`)
- Flask web routes (`GET` and `POST`)
- Image upload validation (invalid crop names, unpermitted file types, empty files)

---

## 🔒 Security & Robustness Highlights

- **Form Parsing by Field Name:** Replaced fragile `request.form.to_dict().values()` unpacking with explicit field-by-field lookup and casting to avoid feature order inversion.
- **Upload Hardening:** Enforces allowed image MIME types (`.jpg`, `.jpeg`, `.png`, `.webp`), rejects empty uploads, and generates unique random filenames (`uuid.uuid4()`) to prevent filesystem collision.
- **Crop Input Validation:** Guarantees crop parameters match supported keys before disk interaction or model loading.
- **Out-of-Distribution Warning:** Flags low confidence predictions (<50%) to guard against false diagnoses on random non-leaf images.
- **Git Hygiene:** Untracked large test uploads via `.gitignore` while maintaining `.gitkeep` directory anchors.

---

## 👨‍💻 Author
**Prakhar Anand Singh**
- GitHub: [@Groot-PAS](https://github.com/Groot-PAS)
- Email: [prakharanand0967@gmail.com](mailto:prakharanand0967@gmail.com)

