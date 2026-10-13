"""
Comprehensive Test Suite for Agro World
Tests:
- Crop recommendation inference and accuracy
- Fertilizer recommendation feature ordering and predictions
- Disease classification metadata, confidence scoring, and potato rename
- Flask web routes and upload validation
"""

import os
import sys
import io
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
from functions import (
    get_crop_recommendation,
    get_fertilizer_recommendation,
    crop_diseases_classes,
    crop_list,
    CROP_CLASSES,
    soil_types,
    Crop_types,
    fertilizer_classes,
    BASE_DIR
)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestCropRecommendation:
    def test_known_crops_match(self):
        """Test with known rows from Crop_recommendation.csv to ensure no off-by-one errors."""
        csv_path = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
        df = pd.read_csv(csv_path)

        test_crops = ['rice', 'apple', 'maize', 'chickpea', 'watermelon', 'coffee', 'cotton']
        for crop_name in test_crops:
            row = df[df['label'] == crop_name].iloc[0]
            features = [
                row['N'], row['P'], row['K'],
                row['temperature'], row['humidity'], row['ph'], row['rainfall']
            ]
            predicted = get_crop_recommendation(features)
            assert predicted.lower() == crop_name.lower(), f"Expected {crop_name}, got {predicted}"

    def test_crops_list_integrity(self):
        """Ensure crop classes count and ordering matches dataset."""
        assert len(CROP_CLASSES) == 22
        assert CROP_CLASSES[0] == 'apple'
        assert CROP_CLASSES[-1] == 'watermelon'


class TestFertilizerRecommendation:
    def test_feature_order_integrity(self):
        """Test with known rows from Fertilizer Prediction.csv ensuring N, K, P ordering."""
        csv_path = os.path.join(BASE_DIR, "dataset", "Fertilizer Prediction.csv")
        df = pd.read_csv(csv_path)

        # Test first 10 rows
        for _, row in df.head(10).iterrows():
            temp = float(row['Temparature'])
            hum = float(row['Humidity '])
            moist = float(row['Moisture'])
            n = float(row['Nitrogen'])
            k = float(row['Potassium'])
            p = float(row['Phosphorous'])
            soil_idx = soil_types.index(row['Soil Type'])
            crop_idx = Crop_types.index(row['Crop Type'])
            actual_fert = row['Fertilizer Name']

            # num_features order MUST be [Temp, Hum, Moist, N, K, P]
            num_features = [temp, hum, moist, n, k, p]
            cat_features = [soil_idx, crop_idx]

            predicted = get_fertilizer_recommendation(num_features, cat_features)
            assert predicted in fertilizer_classes
            assert predicted == actual_fert, f"Expected {actual_fert}, got {predicted}"


class TestDiseaseMetadataAndTypo:
    def test_potato_naming(self):
        """Ensure 'potato' typo is resolved and 'patato' is gone."""
        assert 'potato' in crop_diseases_classes
        assert 'patato' not in crop_diseases_classes
        assert 'potato' in crop_list

        model_path = os.path.join(BASE_DIR, 'models', 'DL_models', 'potato_model.h5')
        assert os.path.exists(model_path), f"potato_model.h5 not found at {model_path}"

    def test_all_10_crops_present(self):
        expected_crops = {
            'apple', 'cherry', 'corn', 'grape', 'peach',
            'pepper', 'potato', 'rice', 'strawberry', 'tomato'
        }
        assert set(crop_list) == expected_crops


class TestFlaskRoutes:
    def test_index_route(self, client):
        res = client.get('/')
        assert res.status_code == 200
        assert b"Agro World" in res.data

    def test_crop_recommendation_get(self, client):
        res = client.get('/crop-recommendation')
        assert res.status_code == 200
        assert b"Crop Recommendation" in res.data

    def test_crop_recommendation_post(self, client):
        # Known rice sample: N=90, P=42, K=43, temp=20.8, hum=82.0, ph=6.5, rainfall=202.9
        data = {
            'N': '90',
            'P': '42',
            'K': '43',
            'temperature': '20.8',
            'humidity': '82.0',
            'ph': '6.5',
            'rainfall': '202.9'
        }
        res = client.post('/crop-recommendation', data=data)
        assert res.status_code == 200
        assert b"Rice" in res.data

    def test_fertilizer_recommendation_get(self, client):
        res = client.get('/fertilizer-recommendation')
        assert res.status_code == 200
        assert b"Fertilizer Recommendation" in res.data

    def test_fertilizer_recommendation_post(self, client):
        # Known Urea sample: Sandy (index 4), Maize (index 3), Temp=26, Hum=52, Moist=38, N=37, P=0, K=0
        data = {
            'temperature': '26',
            'humidity': '52',
            'moisture': '38',
            'N': '37',
            'P': '0',
            'K': '0',
            'soil': '4',
            'crop': '3'
        }
        res = client.post('/fertilizer-recommendation', data=data)
        assert res.status_code == 200
        assert b"Urea" in res.data

    def test_crop_disease_validation(self, client):
        # Invalid crop rejected
        res = client.post('/crop-disease', data={
            'crop': 'invalid_crop',
            'file': (io.BytesIO(b"fake image content"), 'test.jpg')
        }, follow_redirects=True)
        assert b"Invalid crop selected" in res.data

        # Invalid file extension rejected
        res = client.post('/crop-disease', data={
            'crop': 'potato',
            'file': (io.BytesIO(b"fake executable"), 'malicious.exe')
        }, follow_redirects=True)
        assert b"Unsupported file format" in res.data
