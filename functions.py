import os
import pickle
from functools import lru_cache
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Disease classes mapping for each supported crop
crop_diseases_classes = {
    'strawberry': [
        (0, 'Leaf_scorch'),
        (1, 'healthy')
    ],
    'potato': [
        (0, 'Early_blight'),
        (1, 'Late_blight'),
        (2, 'healthy')
    ],
    'rice': [
        (0, 'bacterial_Leaf_blight'),
        (1, 'brown_spot'),
        (2, 'healthy'),
        (3, 'leaf_blast'),
        (4, 'leaf_scald'),
        (5, 'narrow_brown_spot'),
        (6, 'Neck_Blast'),
        (7, 'Rice Hispa'),
        (8, 'Sheath Blight'),
        (9, 'Tungro')
    ],
    'corn': [
        (0, 'Cercospora_leaf_spot Gray_leaf_spot'),
        (1, 'Common_rust_'),
        (2, 'Northern_Leaf_Blight'),
        (3, 'healthy')
    ],
    'apple': [
        (0, 'Apple_scab'),
        (1, 'Black_rot'),
        (2, 'Cedar_apple_rust'),
        (3, 'healthy')
    ],
    'cherry': [
        (0, 'Powdery_mildew'),
        (1, 'healthy')
    ],
    'grape': [
        (0, 'Black_rot'),
        (1, 'Esca_(Black_Measles)'),
        (2, 'Leaf_blight_(Isariopsis_Leaf_Spot)'),
        (3, 'healthy')
    ],
    'peach': [
        (0, 'Bacterial_spot'),
        (1, 'healthy')
    ],
    'pepper': [
        (0, 'Bacterial_spot'),
        (1, 'healthy')
    ],
    'tomato': [
        (0, 'Bacterial_spot'),
        (1, 'Early_blight'),
        (2, 'Late_blight'),
        (3, 'Leaf_Mold'),
        (4, 'Septoria_leaf_spot'),
        (5, 'Spider_mites Two-spotted_spider_mite'),
        (6, 'Target_Spot'),
        (7, 'Tomato_Yellow_Leaf_Curl_Virus'),
        (8, 'Tomato_mosaic_virus'),
        (9, 'healthy')
    ]
}

crop_list = sorted(list(crop_diseases_classes.keys()))

# Alphabetical list of crops matching the 0-indexed ML model predictions
CROP_CLASSES = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
    'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]
crops = CROP_CLASSES  # Backward compatibility alias

soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
Crop_types = [
    'Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets',
    'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat'
]
fertilizer_classes = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']

CONFIDENCE_THRESHOLD = 0.50


@lru_cache(maxsize=3)
def get_model(path):
    """
    Loads Keras deep learning model with LRU cache (maxsize=3) to prevent
    repeated disk reads while keeping memory bounded on resource-constrained servers.
    """
    from tensorflow.keras.models import load_model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return load_model(path, compile=False)


def img_predict(path, crop):
    """
    Preprocesses the leaf image, runs model inference, and calculates confidence.
    Returns:
        predicted: int class index
        confidence: float probability (0.0 to 1.0)
    """
    from tensorflow.keras.preprocessing.image import load_img

    if crop not in crop_diseases_classes:
        raise ValueError(f"Invalid crop '{crop}'. Must be one of {crop_list}")

    model_path = os.path.join(BASE_DIR, 'models', 'DL_models', f'{crop}_model.h5')
    model = get_model(model_path)

    img = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(img).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255.0

    class_count = len(crop_diseases_classes[crop])
    if class_count > 2:
        # Multi-class model (softmax output)
        probabilities = model.predict(data)[0]
        predicted = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
    else:
        # Binary classification model (sigmoid output, units=1)
        raw_pred = model.predict(data)[0]
        p = float(raw_pred[0]) if hasattr(raw_pred, '__len__') else float(raw_pred)
        predicted = int(np.round(p))
        confidence = p if predicted == 1 else (1.0 - p)

    return predicted, confidence


def get_diseases_classes(crop, prediction):
    """Returns human-readable disease class name from prediction index."""
    crop_classes = crop_diseases_classes[crop]
    if 0 <= prediction < len(crop_classes):
        return crop_classes[prediction][1].replace("_", " ")
    return "Unknown Condition"


def diagnose_crop_image(file_path, crop, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Full pipeline helper: loads image, predicts, checks confidence.
    Returns:
        disease_name: str
        confidence_pct: float (0.0 - 100.0)
        is_confident: bool (False if below threshold, e.g. non-leaf image)
    """
    pred_idx, conf = img_predict(file_path, crop)
    disease_name = get_diseases_classes(crop, pred_idx)
    conf_pct = round(conf * 100.0, 1)
    is_confident = conf >= confidence_threshold
    return disease_name, conf_pct, is_confident


@lru_cache(maxsize=1)
def load_crop_artifacts():
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return scaler, model


@lru_cache(maxsize=1)
def load_fertilizer_artifacts():
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return scaler, model


def get_crop_recommendation(item):
    """
    Predicts optimal crop from 7 features:
    item = [N, P, K, temperature, humidity, ph, rainfall]
    """
    scaler, model = load_crop_artifacts()
    features = np.array(item, dtype=float).reshape(1, -1)
    scaled_item = scaler.transform(features)
    prediction = int(model.predict(scaled_item)[0])
    if 0 <= prediction < len(CROP_CLASSES):
        return CROP_CLASSES[prediction]
    raise IndexError(f"Model prediction index {prediction} out of bounds for {len(CROP_CLASSES)} crops.")


def get_fertilizer_recommendation(num_features, cat_features):
    """
    Predicts optimal fertilizer from:
    num_features = [temperature, humidity, moisture, nitrogen, potassium, phosphorous] (N, K, P order)
    cat_features = [soil_type_index, crop_type_index]
    """
    scaler, model = load_fertilizer_artifacts()
    num_arr = np.array(num_features, dtype=float).reshape(1, -1)
    cat_arr = np.array(cat_features, dtype=float).reshape(1, -1)

    scaled_num = scaler.transform(num_arr)
    item = np.concatenate([scaled_num, cat_arr], axis=1)
    prediction = int(model.predict(item)[0])
    if 0 <= prediction < len(fertilizer_classes):
        return fertilizer_classes[prediction]
    raise IndexError(f"Fertilizer prediction index {prediction} out of bounds.")
